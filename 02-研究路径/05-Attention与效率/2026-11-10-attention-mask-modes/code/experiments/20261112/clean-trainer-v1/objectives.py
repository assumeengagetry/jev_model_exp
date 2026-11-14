"""Actual-margin complete-field objectives; no micro/worker renormalization."""
from __future__ import annotations

from bindings import require


def objective(margins, fields, cases, kind):
    import torch
    import torch.nn.functional as F
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
    loss, bookkeeping = objective(m, fields, cases, kind)
    (coef,) = torch.autograd.grad(loss, m)
    require(bool(torch.isfinite(coef).all()) and bool(torch.isfinite(loss)), 'Nonfinite objective/VJP coefficient')
    return coef.detach(), float(loss.detach()), bookkeeping
