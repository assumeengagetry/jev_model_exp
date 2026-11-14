"""Actual-margin complete-field objectives; no micro/worker renormalization."""
from __future__ import annotations

from bindings import require


def _hard_rank_components(margins, fields, cases):
    import torch
    import torch.nn.functional as F
    require(margins.ndim == 1 and cases > 0 and fields, 'Invalid objective scope')
    require(bool(torch.isfinite(margins).all()), 'Nonfinite actual margins')
    bookkeeping = {
        'cases': cases,
        'known_fields': len(fields),
        'known_candidates': 0,
        'K1_fields': 0,
        'hard_positive_candidates': 0,
        'hard_negative_candidates': 0,
        'soft_candidates': 0,
        'rank_active_fields': 0,
        'rank_zero_fields': 0,
        'rank_zero_no_positive_fields': 0,
        'rank_zero_no_negative_fields': 0,
        'rank_zero_K1_fields': 0,
        'rank_zero_soft_only_fields': 0,
    }
    losses, bce_losses, rank_losses, active_cases = [], [], [], set()
    for field in fields:
        m = margins[field['start']:field['end']]
        targets = field['target']
        require(m.numel() == len(targets) and m.numel() > 0 and field['known_fields_in_case'] > 0,
                'Complete field or fixed known-field denominator missing')
        case_index = field['case_index']
        require(isinstance(case_index, int) and 0 <= case_index < cases, 'Invalid case occurrence index')
        require(all(isinstance(q, (float, int)) and 0 <= q <= 1 for q in targets),
                'Targets must be original finite Python probabilities')
        # Classify ORIGINAL host values: float32 rounding must never harden a soft q.
        positive = [index for index, q in enumerate(targets) if q == 1]
        negative = [index for index, q in enumerate(targets) if q == 0]
        q = torch.tensor(targets, dtype=m.dtype, device=m.device)
        bce = F.binary_cross_entropy_with_logits(m, q, reduction='mean')
        active = bool(positive) and bool(negative)
        if active:
            rank = torch.logsumexp(m[positive + negative], dim=0) - torch.logsumexp(m[positive], dim=0)
            active_cases.add(case_index)
        else:
            # Multiply first so a sum of large, finite margins cannot overflow.
            # Every known slot remains graph-connected, including K1/soft-only fields.
            rank = (m * 0.0).sum()
        denominator = field['known_fields_in_case']
        # Preserve field-wise (B + R) / F_known / C; do not sum component gradients.
        losses.append((bce + rank) / denominator / cases)
        bce_losses.append(bce / denominator / cases)
        rank_losses.append(rank / denominator / cases)
        k1 = m.numel() == 1
        bookkeeping['known_candidates'] += m.numel()
        bookkeeping['K1_fields'] += k1
        bookkeeping['hard_positive_candidates'] += len(positive)
        bookkeeping['hard_negative_candidates'] += len(negative)
        bookkeeping['soft_candidates'] += m.numel() - len(positive) - len(negative)
        bookkeeping['rank_active_fields'] += active
        bookkeeping['rank_zero_fields'] += not active
        # These reasons intentionally overlap; they are not a partition.
        bookkeeping['rank_zero_no_positive_fields'] += not positive
        bookkeeping['rank_zero_no_negative_fields'] += not negative
        bookkeeping['rank_zero_K1_fields'] += k1
        bookkeeping['rank_zero_soft_only_fields'] += not positive and not negative
    total, bce, rank = (torch.stack(values).sum() for values in (losses, bce_losses, rank_losses))
    require(all(bool(torch.isfinite(value)) for value in (total, bce, rank)), 'Nonfinite objective component')
    bookkeeping['no_rank_cases'] = cases - len(active_cases)
    bookkeeping['macro_bce_loss'] = float(bce.detach())
    bookkeeping['macro_rank_loss'] = float(rank.detach())
    bookkeeping['macro_total_loss'] = float(total.detach())
    return total, bce, rank, bookkeeping


def _hard_rank_coefficients(margins, total, bce, rank, bookkeeping, retain_graph=False):
    import torch
    from math import isfinite
    (bce_coef,) = torch.autograd.grad(bce, margins, retain_graph=True)
    (rank_coef,) = torch.autograd.grad(rank, margins, retain_graph=True)
    # The returned coefficient is from the actual complete total loss, independently.
    (coef,) = torch.autograd.grad(total, margins, retain_graph=retain_graph)
    for name, component in (('bce', bce_coef), ('rank', rank_coef), ('total', coef)):
        require(bool(torch.isfinite(component).all()), 'Nonfinite objective/VJP coefficient')
        values = component.detach().double()
        l2, maxabs = float(torch.linalg.vector_norm(values)), float(values.abs().max())
        require(isfinite(l2) and isfinite(maxabs), 'Nonfinite margin-coefficient summary')
        bookkeeping['margin_' + name + '_l2'] = l2
        bookkeeping['margin_' + name + '_maxabs'] = maxabs
    for name in ('rank', 'total'):
        for norm in ('l2', 'maxabs'):
            denominator = bookkeeping['margin_bce_' + norm]
            relative = bookkeeping['margin_' + name + '_' + norm] / denominator if denominator else None
            require(relative is None or isfinite(relative), 'Nonfinite relative margin-coefficient scale')
            bookkeeping['margin_' + name + '_over_bce_' + norm] = relative
    return coef.detach()


def objective(margins, fields, cases, kind):
    import torch
    import torch.nn.functional as F
    if kind == 'BCE_RANK_HARD':
        if torch.is_grad_enabled() and margins.requires_grad:
            loss, bce, rank, bookkeeping = _hard_rank_components(margins, fields, cases)
            # Keep the returned loss usable for the caller's own backward pass.
            _hard_rank_coefficients(margins, loss, bce, rank, bookkeeping, retain_graph=True)
            return loss, bookkeeping
        # Diagnostics still describe these actual margins when callers request no graph.
        # No model is run; preserve objective input dtype rather than forcing float32.
        with torch.enable_grad():
            m = margins.detach().clone().requires_grad_(True)
            loss, bce, rank, bookkeeping = _hard_rank_components(m, fields, cases)
            _hard_rank_coefficients(m, loss, bce, rank, bookkeeping)
        return loss.detach(), bookkeeping
    require(kind in ('BCE', 'CE') and margins.ndim == 1 and cases > 0 and fields, 'Invalid objective scope')
    require(bool(torch.isfinite(margins).all()), 'Nonfinite actual margins')
    losses, bookkeeping = [], {'cases': cases, 'known_fields': len(fields), 'known_candidates': 0, 'K1_fields': 0}
    for field in fields:
        m = margins[field['start']:field['end']]
        q = torch.tensor(field['target'], dtype=m.dtype, device=m.device)
        require(m.numel() == q.numel() and m.numel() > 0 and field['known_fields_in_case'] > 0,
                'Complete field or fixed known-field denominator missing')
        if kind == 'BCE':
            loss = F.binary_cross_entropy_with_logits(m, q, reduction='mean')
        else:
            # Native q is deliberately not renormalized. K=1 remains in denominator.
            loss = -(q * torch.log_softmax(m, dim=0)).sum()
        losses.append(loss / field['known_fields_in_case'] / cases)
        bookkeeping['known_candidates'] += m.numel()
        bookkeeping['K1_fields'] += m.numel() == 1
    return torch.stack(losses).sum(), bookkeeping


def coefficients(actual_margins, fields, cases, kind):
    import torch
    # This graph has ONLY actual cached model margins, not synthetic model input.
    m = actual_margins.detach().float().clone().requires_grad_(True)
    if kind == 'BCE_RANK_HARD':
        loss, bce, rank, bookkeeping = _hard_rank_components(m, fields, cases)
        coef = _hard_rank_coefficients(m, loss, bce, rank, bookkeeping)
        return coef, float(loss.detach()), bookkeeping
    loss, bookkeeping = objective(m, fields, cases, kind)
    (coef,) = torch.autograd.grad(loss, m)
    require(bool(torch.isfinite(coef).all()) and bool(torch.isfinite(loss)), 'Nonfinite objective/VJP coefficient')
    return coef.detach(), float(loss.detach()), bookkeeping
