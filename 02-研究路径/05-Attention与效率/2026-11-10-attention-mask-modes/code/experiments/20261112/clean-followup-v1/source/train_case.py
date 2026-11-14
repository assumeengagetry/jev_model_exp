"""Explicit world1 full-case training CLI. Importing this module runs nothing."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from bindings import Context, DOMAINS, commit_json, fsync_tree, offline_environment, require, stat, write_json
from case_runtime import CaseStore, Exposure, micro_plan, slots, slot_manifest
from checkpoints import (MacroCosine, Run, cpu_tree, rng_state, save_tensor_file)
from fresh_init import adapter_state, audit_model, load_fresh, validate_init
from model_runtime import FullRow, configure_cuda
from objectives import coefficients
from state_boundaries import BoundaryStore


def bitwise_equal(a, b):
    import torch
    return (a.shape == b.shape and a.dtype == b.dtype and
            torch.equal(a.detach().cpu().contiguous().view(torch.uint8),
                        b.detach().cpu().contiguous().view(torch.uint8)))


def tree_equal(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and bitwise_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def gradients(model):
    import torch
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    require(len(params) == 393, 'Expected all393 trainable leaves')
    stats = {}
    for name, p in sorted(params.items()):
        require(p.dtype == torch.float32 and p.grad is not None and p.grad.dtype == torch.float32 and
                torch.isfinite(p.grad).all().item(), 'Missing/nonfinite FP32 gradient: ' + name)
        stats[name] = {'shape': list(p.grad.shape), 'dtype': str(p.grad.dtype),
                       'nonzero': int(torch.count_nonzero(p.grad).item()),
                       'l2_float64': float(torch.linalg.vector_norm(p.grad.double()).item())}
    return params, stats


def optimizer_finite(optimizer, model):
    import torch
    params = [p for _, p in sorted(model.named_parameters()) if p.requires_grad]
    require(len(optimizer.state) == 393 and len(params) == 393, 'AdamW must hold all393 parameter states')
    for p in params:
        require(p in optimizer.state and torch.isfinite(p).all().item(), 'Missing state/nonfinite updated trainable')
        state = optimizer.state[p]
        require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'} and
                all(torch.isfinite(v).all().item() for v in state.values()), 'Nonfinite/unexpected AdamW state')


def train_macro(run, runtime, optimizer, scheduler, cases, exposures, macro_index):
    import torch
    runtime.train()
    pairs, fields = slots(cases, boundary_store=runtime.boundary_store)
    plan = micro_plan(pairs)
    require(len(cases) == 16 and sum(p.length for p in pairs) ==
            sum(e['logical_pair_tokens'] for e in exposures), 'Macro input/exposure accounting differs')
    raw_known = sum(field['known'] for case in cases for field in case['fields'])
    raw_fields = sum(len(case['fields']) for case in cases)
    coverage = {
        'case_occurrences': len(cases), 'raw_fields': raw_fields, 'known_fields': raw_known,
        'unknown_fields_retained_no_forward': raw_fields - raw_known,
        'cases_with_known_fields': sum(any(f['known'] for f in case['fields']) for case in cases),
        'cases_with_unknown_fields': sum(any(not f['known'] for f in case['fields']) for case in cases),
        'forwarded_complete_candidates': len(pairs),
    }
    require(raw_known == len(fields), 'Macro lost a known field or forwarded an unknown field')
    first = macro_index == 0
    artifact = None
    if first:
        artifact = run.root / 'first-actual-macro'
        artifact.mkdir(mode=0o700)
    mapping = slot_manifest(pairs, cases, exposures)
    if first:
        write_json(artifact / 'SLOTS.json', {'macro_index': macro_index, 'exposures': exposures,
                                            'slots': mapping, 'fields': fields,
                                            'micros': [[p.slot for p in micro] for micro in plan],
                                            'not_an_extra_smoke': True})
    optimizer.zero_grad(set_to_none=True)
    logits_cache = torch.empty((len(pairs), 2), dtype=torch.float32, device='cpu')
    margin_cache = torch.empty(len(pairs), dtype=torch.float32, device='cpu')
    replay_logits_cache = torch.empty_like(logits_cache) if first else None
    replay_margin_cache = torch.empty_like(margin_cache) if first else None
    start_rng = rng_state()
    before_parameters = cpu_tree(adapter_state(runtime.model)) if first else None
    completed_first_slots, completed_replay_slots = [], []
    try:
        if first:
            runtime.begin_macro0_dispatch()
        runtime.phase = 'train_first'
        # The no-grad scope contains no outer autocast. FullRow.forward opens one per wave.
        with torch.no_grad():
            for micro in plan:
                logits, margins = runtime.forward(micro)
                indices = [p.slot for p in micro]
                logits_cache[indices] = logits.detach().cpu()
                margin_cache[indices] = margins.detach().cpu()
                completed_first_slots.extend(indices)
                del logits, margins
        require(sorted(completed_first_slots) == list(range(len(pairs))), 'First-pass slot coverage incomplete')
        require(tree_equal(start_rng, rng_state()), 'Zero-dropout first pass consumed RNG; exact replay not admitted')
        coef, loss_value, bookkeeping = coefficients(margin_cache, fields, len(cases), run.ctx.arm['loss'])
        require(bookkeeping['cases'] == len(cases) and bookkeeping['known_fields'] == raw_known and
                bookkeeping['known_candidates'] == len(pairs), 'Actual objective/coverage scope differs')
        if first:
            save_tensor_file(artifact / 'actual-cache-and-coefficients.pt',
                             {'logits': logits_cache, 'margins': margin_cache, 'coefficients': coef,
                              'objective': loss_value, 'bookkeeping': bookkeeping, 'rng_before': start_rng})
        runtime.phase = 'train_replay'
        for wave, micro in enumerate(plan):
            logits, margins = runtime.forward(micro)
            indices = [p.slot for p in micro]
            actual_margins, actual_logits = margins.detach().cpu(), logits.detach().cpu()
            if first:
                replay_logits_cache[indices] = actual_logits
                replay_margin_cache[indices] = actual_margins
            if not (bitwise_equal(actual_margins, margin_cache[indices]) and
                    bitwise_equal(actual_logits, logits_cache[indices])):
                failure_dir = artifact if first else run.root
                save_tensor_file(failure_dir / f'replay-mismatch-m{macro_index:07d}-w{wave:05d}.pt',
                                 {'actual_margins': actual_margins, 'expected_margins': margin_cache[indices],
                                  'actual_logits': actual_logits, 'expected_logits': logits_cache[indices],
                                  'coefficients': coef[indices], 'slots': indices,
                                  'pass': False, 'tolerance_relaxed': False})
                raise RuntimeError('First/second logits or margins are not bitwise identical; update forbidden')
            # Coef ALREADY has every complete case and known-field/candidate denominator.
            surrogate = (margins * coef[indices].to(device=margins.device)).sum()
            surrogate.backward()
            completed_replay_slots.extend(indices)
            runtime.counts['train_replay.completed_backward_waves'] += 1
            runtime.counts['train_replay.backward_real_tokens'] += sum(p.length for p in micro)
            del logits, margins, surrogate
        require(sorted(completed_replay_slots) == list(range(len(pairs))), 'Replay coverage incomplete')
        require(tree_equal(start_rng, rng_state()), 'Zero-dropout replay consumed RNG')
        if first:
            save_tensor_file(artifact / 'actual-replay-logits.pt',
                             {'logits': replay_logits_cache, 'margins': replay_margin_cache,
                              'bitwise_equal_to_first': True, 'completed_slots': completed_replay_slots})
        params, grad_stats = gradients(runtime.model)
        if first:
            save_tensor_file(artifact / 'all393-gradients-unclipped.pt',
                             {n: p.grad for n, p in sorted(params.items())})
        norm = torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0,
                                             error_if_nonfinite=True, foreach=False)
        _, clipped_stats = gradients(runtime.model)
        if first:
            save_tensor_file(artifact / 'all393-gradients-clipped.pt',
                             {n: p.grad for n, p in sorted(params.items())})
        used_lr = optimizer.param_groups[0]['lr']
        require(used_lr == scheduler.next_lr(), 'Macro LR differs from frozen actual-step schedule')
        optimizer.step()
        optimizer_finite(optimizer, runtime.model)
        scheduler.step()
        record = {'macro_index': macro_index, 'cases': 16, 'logical_pair_tokens': sum(p.length for p in pairs),
                  'objective': loss_value, 'bookkeeping': bookkeeping, 'coverage': coverage,
                  'micro_waves_per_pass': len(plan),
                  'first_replay_logits_and_margins_bitwise_equal': True,
                  'RNG_unchanged_zero_dropout': True, 'gradient_keys': len(grad_stats),
                  'all393_gradients_finite': True, 'preclip_l2': float(norm.detach().cpu()),
                  'clip_norm': 1.0, 'clip_factor': min(1.0, 1.0 / (float(norm.detach().cpu()) + 1e-6)),
                  'used_lr': used_lr, 'optimizer_steps': scheduler.completed,
                  'trainable_parameters_after_update_finite': True,
                  'actual_completed_forward_real_tokens_this_macro': 2 * sum(p.length for p in pairs),
                  'actual_completed_backward_real_tokens_this_macro': sum(p.length for p in pairs)}
        if first:
            dispatch = runtime.finish_macro0_dispatch()
            write_json(artifact / 'ACTUAL_NATIVE_DISPATCH.json', dispatch)
            del dispatch
            after_parameters = cpu_tree(adapter_state(runtime.model))
            require(set(after_parameters) == set(before_parameters) and len(after_parameters) == 393,
                    'First actual update changed saved parameter set')
            save_tensor_file(artifact / 'all393-actual-update-and-optimizer.pt',
                             {'before': before_parameters, 'after': after_parameters,
                              'delta': {k: after_parameters[k] - before_parameters[k] for k in after_parameters},
                              'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                              'optimizer_parameter_names': [n for n, _ in sorted(params.items())]})
            write_json(artifact / 'ACTUAL_MACRO.json', {**record, 'gradients': grad_stats,
                                                       'clipped_gradients': clipped_stats,
                                                       'runtime_after': runtime.receipt()})
            fsync_tree(artifact)
            commit_json(artifact / 'COMPLETE.json', {'complete': True, 'is_formal_exposure_macro0': True,
                                                  'scope': run.ctx.snapshot,
                                                  'files': [stat(p) for p in sorted(artifact.iterdir()) if p.is_file()]})
            fsync_tree(artifact)
        optimizer.zero_grad(set_to_none=True)
        return record
    except BaseException:
        # Preserve actual caches/partial gradients for later review, never label a partial macro resumable.
        path = run.root / f'failed-actual-macro-m{macro_index:07d}.pt'
        if not path.exists():
            save_tensor_file(path, {'scope': run.ctx.snapshot, 'exposures': exposures, 'slots': mapping,
                                    'first_completed_slots': completed_first_slots,
                                    'replay_completed_slots': completed_replay_slots,
                                    'actual_logits_completed_first': logits_cache[completed_first_slots],
                                    'actual_margins_completed_first': margin_cache[completed_first_slots],
                                    'coefficients_if_available': locals().get('coef'),
                                    'partial_gradients': {n: p.grad for n, p in runtime.model.named_parameters()
                                                          if p.requires_grad and p.grad is not None},
                                    'current_adapter': adapter_state(runtime.model),
                                    'current_optimizer': optimizer.state_dict(),
                                     'current_scheduler': scheduler.state_dict(),
                                     'runtime': runtime.receipt(), 'resumable': False})
        if first and runtime.dispatch_waves is not None:
            try:
                write_json(artifact / 'ACTUAL_NATIVE_DISPATCH.partial.json',
                           runtime.finish_macro0_dispatch(complete=False))
            except BaseException:
                pass  # Keep the original training error and already-saved failure cache.
        raise
    finally:
        if first:
            runtime.stop_macro0_dispatch()


def finish_observations(run, runtime, store, exposure, cursor, dev_done, model,
                        optimizer, scheduler, identity):
    from evaluate_native_dev import evaluate_select
    changed = False
    for node in exposure.schedule['observations']:
        if node['macro_updates'] == cursor['macro_updates'] and node['cap'] not in dev_done:
            require(cursor['cases'] == node['cases'] and cursor['logical_pair_tokens'] == node['logical_pair_tokens'],
                    'Observation is not its largest complete frozen macro prefix')
            output = run.root / f"D_select-cap{node['cap']}-m{cursor['macro_updates']:07d}"
            require(not output.exists(), 'Existing partial/complete dev attempt preserved; do not silently reevaluate')
            run.event('dev_started', node=node, output=str(output))
            evaluate_select(runtime, store, output, scope=run.ctx.snapshot, node=node,
                            checkpoint_identity={'macro_updates': cursor['macro_updates'], 'initialization': identity})
            dev_done.append(node['cap'])
            run.event('dev_completed', node=node, complete=stat(output / 'COMPLETE.json'))
            changed = True
    if changed:
        return run.checkpoint(model, optimizer, scheduler, runtime, cursor, dev_done, identity,
                              ['observation_evaluations_committed'])
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--resume-checkpoint', type=Path)
    parser.add_argument('--resume-complete-pin', type=Path, help='Reviewed JSON stat of own checkpoint COMPLETE')
    parser.add_argument('--pause-after-macros', type=int, help='Absolute complete-macro cursor; exits75 after durable checkpoint')
    parser.add_argument('--evaluate-only', action='store_true', help='Resume only a pending frozen D_select node, then exit75')
    args = parser.parse_args(argv)
    require(not args.evaluate_only or (args.resume_checkpoint is not None and args.pause_after_macros is None),
            'Evaluate-only requires explicit own checkpoint, not a fresh model/arbitrary cursor')
    offline_environment()
    ctx = Context(args.config, args.arm)
    init_identity = validate_init(ctx)
    train = select = exposure = run = runtime = boundary_store = None
    cursor = {'macro_updates': 0, 'cases': 0, 'logical_pair_tokens': 0}
    try:
        train = CaseStore(Path(ctx.cfg['data_root']) / 'data' / 'train.jsonl', 'train')
        require(all(train.domains.get(d, 0) >= 2500 for d in DOMAINS), 'Actual train inventory under2500/domain')
        exposure = Exposure(ctx.cfg['data_root'], ctx.cfg['seed'], train)
        select = CaseStore(Path(ctx.cfg['data_root']) / 'data' / 'select.jsonl', 'validation')
        require(select.domains == {d: 75 for d in DOMAINS}, 'D_select actual quota differs')
        require(not set(train.meta) & set(select.meta), 'Train/select overlap')
        boundary_store = (BoundaryStore(ctx.state_boundary, train, select)
                          if ctx.arm_name == 'STATE_NL_PB_P_CE' else None)
        total_macros = exposure.schedule['macro_updates']
        require(args.pause_after_macros is None or 0 <= args.pause_after_macros <= total_macros, 'Bad pause boundary')
        run = Run(ctx, args.output, resume=args.resume_checkpoint, resume_pin=args.resume_complete_pin)
        environment = configure_cuda(ctx)
        import torch
        import numpy as np
        np.random.seed(ctx.cfg['seed'])
        model, compared = load_fresh(ctx, init_identity)
        runtime = FullRow(ctx, model)
        runtime.boundary_store = boundary_store
        names = sorted(n for n, p in model.named_parameters() if p.requires_grad)
        named = dict(model.named_parameters())
        optimizer = torch.optim.AdamW([{'params': [named[n] for n in names], 'param_names': names}],
                                       lr=2e-5, weight_decay=0.01, fused=True, betas=(0.9, 0.999), eps=1e-8)
        scheduler = MacroCosine(optimizer, total_macros)
        identity = {'fresh_init': init_identity, 'per_key_comparison': compared, 'environment': environment,
                    'fresh_before_any_resume': True, 'audit_initial': audit_model(model, require_fresh=True)}
        dev_done = []
        if run.resume is not None:
            cursor, dev_done, identity = run.restore(model, optimizer, scheduler, runtime)
            require(cursor['cases'] == cursor['macro_updates'] * 16 and
                    cursor['logical_pair_tokens'] == exposure.prefix_tokens(cursor['macro_updates']),
                    'Own-resume exposure cursor differs')
            require(all(cap in (10_000_000, 30_000_000, 100_000_000) for cap in dev_done), 'Invalid dev cursor')
            run.event('resumed_environment', environment=environment, cursor=cursor)
        else:
            run.event('initialized_no_forward', environment=environment, per_key_comparison=compared,
                      exposure_schedule=exposure.schedule, fresh_init=ctx.cfg['fresh_init'])
            run.checkpoint(model, optimizer, scheduler, runtime, cursor, dev_done, identity, ['step0_no_forward'])
        require(args.pause_after_macros is None or args.pause_after_macros >= cursor['macro_updates'],
                'Absolute pause cursor is behind restored completed exposure; no implicit reset')
        if run.should_stop() or args.pause_after_macros == cursor['macro_updates']:
            run.checkpoint(model, optimizer, scheduler, runtime, cursor, dev_done, identity, ['pause_at_full_boundary'])
            run.event('paused', cursor=cursor, exit_code=75, signal=run.stop_signal)
            return 75
        if args.evaluate_only:
            require(any(n['macro_updates'] == cursor['macro_updates'] and n['cap'] not in dev_done
                        for n in exposure.schedule['observations']), 'No pending frozen D_select node at this checkpoint')
        finish_observations(run, runtime, select, exposure, cursor, dev_done, model,
                            optimizer, scheduler, identity)
        if args.evaluate_only or run.should_stop():
            run.checkpoint(model, optimizer, scheduler, runtime, cursor, dev_done, identity,
                           ['evaluate_only_pause' if args.evaluate_only else 'stop_after_dev'])
            run.event('paused', cursor=cursor, exit_code=75, signal=run.stop_signal)
            return 75
        for macro_index in range(cursor['macro_updates'], total_macros):
            ctx.check()
            rows = exposure.macro(macro_index)
            cases = [train.get(row['case_id']) for row in rows]
            run.event('macro_started', macro_index=macro_index, exposures=rows)
            result = train_macro(run, runtime, optimizer, scheduler, cases, rows, macro_index)
            cursor = {'macro_updates': macro_index + 1, 'cases': 16 * (macro_index + 1),
                      'logical_pair_tokens': cursor['logical_pair_tokens'] + result['logical_pair_tokens']}
            run.event('macro_updated', cursor=cursor, result=result, runtime=runtime.receipt())
            node = any(n['macro_updates'] == cursor['macro_updates'] for n in exposure.schedule['observations'])
            pause = run.should_stop() or args.pause_after_macros == cursor['macro_updates']
            reasons = []
            if cursor['macro_updates'] == 1:
                reasons.append('formal_first_actual_macro')
            if cursor['macro_updates'] % ctx.cfg['checkpoint_every_macros'] == 0:
                reasons.append('controlled_periodic')
            if node:
                reasons.append('predeclared_observation_before_dev')
            if pause:
                reasons.append('pause_requested_at_full_macro')
            if cursor['macro_updates'] == total_macros:
                reasons.append('final_exposure_macro')
            if reasons:
                run.checkpoint(model, optimizer, scheduler, runtime, cursor, dev_done, identity, reasons)
            # STOP priority: pause before evaluation, pending node is handled on explicit own-resume.
            if pause:
                run.event('paused', cursor=cursor, exit_code=75, signal=run.stop_signal)
                return 75
            finish_observations(run, runtime, select, exposure, cursor, dev_done, model,
                                optimizer, scheduler, identity)
            if run.should_stop():
                run.checkpoint(model, optimizer, scheduler, runtime, cursor, dev_done, identity, ['stop_after_dev'])
                run.event('paused', cursor=cursor, exit_code=75, signal=run.stop_signal)
                return 75
            run.budget()
        require(cursor['logical_pair_tokens'] == exposure.schedule['logical_pair_tokens'] and
                cursor['cases'] == exposure.schedule['cases'] and scheduler.completed == total_macros and
                sorted(dev_done) == [10_000_000, 30_000_000, 100_000_000], 'Final exposure/dev accounting incomplete')
        ctx.check()
        commit_json(run.root / 'COMPLETE.json', {'complete': True, 'scope': ctx.snapshot, 'cursor': cursor,
                                               'actual_runtime': runtime.receipt(), 'observations': sorted(dev_done),
                                               'final_checkpoint': stat(run.last_checkpoint / 'COMPLETE.json'),
                                               'checkpoints': [stat(p) for p in sorted(run.root.glob('checkpoint-*/COMPLETE.json'))],
                                               'first_actual_macro': stat(run.root / 'first-actual-macro' / 'COMPLETE.json'),
                                               'D_select': [stat(p) for p in sorted(run.root.glob('D_select-*/COMPLETE.json'))],
                                               'all_checkpoints_retained': True, 'public_run': False,
                                               'D_confirm_opened': False, 'schedule': exposure.schedule})
        fsync_tree(run.root)
        run.event('arm_completed', complete=stat(run.root / 'COMPLETE.json'), cursor=cursor)
        return 0
    except BaseException as error:
        if run is not None:
            run.failure(error, cursor, runtime)
        raise
    finally:
        # A failed close must not prevent closing independent caller-owned stores.
        unwinding = sys.exc_info()[0] is not None
        close_error = None
        for handle in (run, boundary_store, select, exposure, train):
            if handle is not None:
                try:
                    handle.close()
                except BaseException as error:
                    if close_error is None:
                        close_error = error
        if close_error is not None and not unwinding:
            raise close_error


if __name__ == '__main__':
    raise SystemExit(main())
