"""Explicit request-local, mode-aware inference adapter; not a serving engine.

No renderer/common-prefix heuristic, loader, trainer, checkpoint writer, device
selection side effect, or launcher. All token rows are supplied by the caller.
"""

from collections import Counter
from contextlib import contextmanager
import operator

import torch

from .layout import _integers, make_batch


class MaskedRuntime:
    def __init__(self, core, pad, eos, context, device, layout='packed'):
        if layout not in ('packed', 'dense'):
            raise ValueError('Explicit packed/dense layout required')
        if core.training or core.backbone.training or core.head.training:
            raise ValueError('Call core.eval() explicitly before constructing an inference runtime')
        context = operator.index(context)
        if not 0 < context <= core.backbone.config.max_position_embeddings:
            raise ValueError('Invalid context limit')
        self.core = core
        self.pad, self.eos = operator.index(pad), operator.index(eos)
        self.context, self.device, self.layout = context, torch.device(device), layout
        if self.device.type != 'cuda':
            raise ValueError('Native FA3 CUDA device required')
        self.prefix = None
        self._active = False
        self.counts = Counter()

    @property
    def mode(self):
        return self.core.mode

    def begin_request(self):
        if self._active:
            raise RuntimeError('Finish the previous request first; runtime is not concurrent')
        self.prefix = None
        self._active = True

    def finish_request(self):
        self.prefix = None
        self._active = False

    @contextmanager
    def request(self):
        self.begin_request()
        try:
            yield self
        finally:
            self.finish_request()

    def _require_request(self):
        if not self._active:
            raise RuntimeError('An explicit request scope is required')
        if self.core.training or self.core.backbone.training or self.core.head.training:
            raise RuntimeError('Runtime cannot execute a training model')

    def _rows(self, rows, *, eos_required):
        rows = tuple(_integers(row, 'token IDs') for row in rows)
        if not rows or any(not row for row in rows):
            raise ValueError('Actual nonempty token rows required')
        if eos_required and any(row[-1] != self.eos for row in rows):
            raise ValueError('Every scored row must end in the actual EOS; no automatic insertion')
        return rows

    def batch(self, rows, prefix_length=0):
        self._require_request()
        if any(len(row) + prefix_length > self.context for row in rows):
            raise ValueError('Context exceeded; no truncation')
        batch = make_batch(rows, self.device, self.layout, self.pad, prefix_length)
        # Attempts are named as such: a later execution failure must not look
        # like a successfully completed wave or a measured per-layer trace.
        self.counts['attempted_tokenwise_rows_per_operator'] += batch.ids.numel()
        self.counts['attempted_real_execution_tokens'] += batch.total
        self.counts['attempted_waves'] += 1
        return batch

    def _prefix_identity(self, prefix_tokens, rendered_prefix_length):
        tokens = _integers(prefix_tokens, 'prefix token IDs')
        if not tokens:
            raise ValueError('An actual nonempty prefix is required')
        if self.mode == 'bidirectional':
            raise ValueError('Bidirectional mode has no reusable immutable prefix')
        if self.mode == 'prefix_block':
            if rendered_prefix_length is None:
                raise ValueError('Supply the original complete Candidate:\\n prefix boundary')
            if operator.index(rendered_prefix_length) != len(tokens):
                raise ValueError('A partial/common prefix is invalid for prefix_block')
        elif rendered_prefix_length is not None and operator.index(rendered_prefix_length) < len(tokens):
            raise ValueError('Chosen causal common prefix exceeds the rendered prefix boundary')
        return tokens

    @torch.inference_mode()
    def prefill(self, prefix_tokens, *, rendered_prefix_length=None):
        """Always recompute one owner; no cross-request or A/B dictionary cache.

        causal may prefill the caller's exact-token common prefix. prefix_block
        must pass the COMPLETE original rendered prefix and its original length.
        This adapter never selects a common prefix across fields by itself.
        """
        self._require_request()
        self.prefix = None  # A failed switch cannot leave a usable old owner.
        tokens = self._prefix_identity(prefix_tokens, rendered_prefix_length)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, captured = self.core(self.batch((tokens,)), (len(tokens),), capture_prefix=True)
        if captured.token_ids != tokens:
            raise RuntimeError('Captured owner identity differs from actual prefix tokens')
        self.prefix = captured
        self.counts['completed_prefix_owners'] += 1
        self.counts['completed_prefix_tokens_once'] += len(tokens)
        return captured

    @torch.inference_mode()
    def branched(self, prefix_tokens, rows, *, rendered_prefix_length=None):
        """Score suffix-only rows; exact prefix token identity is mandatory.

        Passing only a length (the old runtime API) is intentionally unsupported.
        Each row contains its own entire suffix+EOS, never a sibling's tokens.
        """
        self._require_request()
        tokens = self._prefix_identity(prefix_tokens, rendered_prefix_length)
        if self.prefix is None or self.prefix.token_ids != tokens:
            raise ValueError('Wrong complete prefix owner, even if lengths happen to match')
        rows = self._rows(rows, eos_required=True)
        batch = self.batch(rows, len(tokens))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = self.core(batch, (len(tokens),) * len(rows), prefix=self.prefix)
        margins = self._margins(logits, len(rows))
        self.counts['completed_fork_waves'] += 1
        return margins

    @torch.inference_mode()
    def full(self, rows, prefix_lengths=None):
        """Complete real P+S+EOS rows; mandatory path for bidirectional mode."""
        self._require_request()
        rows = self._rows(rows, eos_required=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = self.core(self.batch(rows), prefix_lengths)
        margins = self._margins(logits, len(rows))
        self.counts['completed_full_waves'] += 1
        return margins

    @staticmethod
    def _margins(logits, rows):
        if logits.shape != (rows, 2):
            raise ValueError('Frozen two-label classifier output required')
        return logits[:, 1].float() - logits[:, 0].float()

    def runtime(self):
        """Structural counters only, not a claim of numerical/performance admission."""
        return {'mode': self.mode, 'layout': self.layout,
                'execution': 'dynamic eager real-shape',
                'prefix_policy': 'disabled' if self.mode == 'bidirectional' else
                                 'complete exact-token block only' if self.mode == 'prefix_block' else
                                 'caller-selected exact-token common prefix',
                'cuda_graph_dummy_rows': 0, 'concurrent_requests_supported': False,
                'counts': dict(self.counts), 'numerical_admission': 'not executed'}
