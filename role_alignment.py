"""Role-exchange consistency from fixed training targets.

A random cycle estimates the mean loss over distinct ordered pairs of valid users in the batch.
Sparse target products and sampled latent pairs require O(B*d + nnz(R_batch))
extra work, reusing the available dense neighborhood targets.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F


class RoleExchangePairPlan:
    """Cache valid-user indices and sample fresh cycles with one device transfer."""

    def __init__(self, valid_users, batch_size, device):
        self.valid_users = np.asarray(valid_users, dtype=np.int64).copy()
        self.batch_size = int(batch_size)
        self.device = device
        self.count = len(self.valid_users)
        self.row_ids = np.arange(self.batch_size, dtype=np.int64)
        self.pair_count_tensor = torch.tensor(
            self.count if self.count >= 2 else 0, device=device,
        )
        self.valid_count_tensor = torch.tensor(self.count, device=device)

    def sample(self, generator):
        if self.count < 2:
            return None
        permutation = torch.randperm(self.count, generator=generator, device="cpu").numpy()
        packed = np.empty(2 * self.count + 2 * self.batch_size, dtype=np.int64)
        left = packed[:self.count]
        right = packed[self.count:2 * self.count]
        successor = packed[2 * self.count:2 * self.count + self.batch_size]
        predecessor = packed[2 * self.count + self.batch_size:]
        left[:] = self.valid_users[permutation]
        right[:-1], right[-1] = left[1:], left[0]
        successor[:] = predecessor[:] = self.row_ids
        successor[left] = right
        predecessor[right] = left
        indices = torch.from_numpy(packed).to(self.device)
        return indices.split((self.count, self.count, self.batch_size, self.batch_size))


class _BernoulliRenyiDivergence(torch.autograd.Function):
    """Compute the divergence in log space and differentiate the prediction logit."""

    @staticmethod
    def forward(ctx, target, prediction, order):
        log_overlap = torch.logaddexp(
            order * F.logsigmoid(target) + (1.0 - order) * F.logsigmoid(prediction),
            order * F.logsigmoid(-target) + (1.0 - order) * F.logsigmoid(-prediction),
        )
        raw = log_overlap / (order - 1.0)
        ctx.save_for_backward(target, prediction, raw >= 0.0)
        ctx.order = order
        return raw.clamp_min(0.0)

    @staticmethod
    def backward(ctx, grad_output):
        target, prediction, nonnegative = ctx.saved_tensors
        mixture = ctx.order * target + (1.0 - ctx.order) * prediction
        # Exact logit derivative; the mask follows the forward round-off clamp.
        derivative = torch.sigmoid(prediction) - torch.sigmoid(mixture)
        return None, grad_output * derivative * nonnegative, None


def bernoulli_renyi_from_logits(target_logits, predicted_logits, order):
    """D_q(Bern(sigmoid(target)) || Bern(sigmoid(prediction))), 0 < q < 1."""
    if not math.isfinite(order) or not 0.0 < order < 1.0:
        raise ValueError("Role alignment requires 0 < renyi_order < 1.")
    target_logits = target_logits.detach()
    if predicted_logits.dtype in (torch.float16, torch.bfloat16):
        predicted_logits = predicted_logits.float()
        target_logits = target_logits.float()
    target_logits, predicted_logits = torch.broadcast_tensors(target_logits, predicted_logits)
    return _BernoulliRenyiDivergence.apply(target_logits, predicted_logits, order)


def role_exchange_consistency_loss(
    interaction_latent,
    social_latent,
    *,
    interaction_entries,
    social_targets,
    interaction_norms,
    social_norms,
    target_valid=None,
    order=0.3,
    temperature=0.2,
    generator=None,
    pair_plan=None,
    sampled_pairs=None,
    normalized_latents=None,
):
    """Match training-derived and latent role-exchange differences on a random cycle.

    interaction_entries contains batch-local (row, item, value) arrays.
    Cached norms must use these interactions and the same neighborhood targets.
    A dedicated CPU generator separates pair sampling from model randomness.
    """
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("role_exchange_temperature must be finite and positive.")
    if not math.isfinite(order) or not 0.0 < order < 1.0:
        raise ValueError("Role alignment requires 0 < renyi_order < 1.")
    batch_size = interaction_latent.shape[0]
    if social_latent.shape != interaction_latent.shape:
        raise ValueError("Both latent batches must have identical shapes.")
    if social_targets.shape[0] != batch_size:
        raise ValueError("Neighborhood targets must follow the latent batch order.")
    if interaction_norms.shape != (batch_size,) or social_norms.shape != (batch_size,):
        raise ValueError("Target norms must contain one value per batch user.")

    if pair_plan is None:
        with torch.no_grad():
            valid = (interaction_norms > 1e-8) & (social_norms > 1e-8)
            if target_valid is not None:
                valid = valid & target_valid.bool()
            valid_users = torch.nonzero(valid, as_tuple=False).flatten()
        count = valid_users.numel()
    else:
        if pair_plan.batch_size != batch_size:
            raise ValueError("Pair plan must follow the current batch order and size.")
        count = pair_plan.count
    zero = interaction_latent.new_zeros(())
    diagnostics = {
        "role_pairs": (count if count >= 2 else 0) if pair_plan is None else pair_plan.pair_count_tensor,
        "role_valid_users": count if pair_plan is None else pair_plan.valid_count_tensor,
        "role_target_gap_abs": zero.detach(),
        "role_predicted_gap_abs": zero.detach(),
    }
    if count < 2:
        zero = (interaction_latent.sum() + social_latent.sum()) * 0.0
        return zero, diagnostics

    # Each valid user's successor is uniform among the other valid users in this batch.
    if pair_plan is None:
        order_indices = torch.randperm(count, generator=generator, device="cpu")
        left = valid_users[order_indices.to(valid_users.device)]
        right = left.roll(-1)
        successor = torch.arange(batch_size, device=left.device)
        predecessor = successor.clone()
        successor[left] = right
        predecessor[right] = left
    else:
        if sampled_pairs is None:
            sampled_pairs = pair_plan.sample(generator)
        left, right, successor, predecessor = sampled_pairs
    with torch.no_grad():
        rows, items, values = interaction_entries
        # Invalid rows only accumulate into their own slots, which are never
        # read below. No dynamic boolean-index filtering is needed on the GPU.
        targets = social_targets.detach()
        values = values.to(targets.dtype)
        neighbor_rows = torch.stack((successor, predecessor))[:, rows]
        contributions = values * targets[neighbor_rows, items]
        dots = targets.new_zeros((2, batch_size))
        dots.scatter_add_(1, rows.expand(2, -1), contributions)
        forward_dot, reverse_dot = dots.unbind(0)
        c_uv = forward_dot[left] / (interaction_norms[left] * social_norms[right])
        c_vu = reverse_dot[right] / (interaction_norms[right] * social_norms[left])
        # Preserve how preference agreement changes when user roles are exchanged.
        target_gap = c_uv.clamp(-1.0, 1.0) - c_vu.clamp(-1.0, 1.0)
        target_logits = target_gap / temperature

    if normalized_latents is None:
        normalized_i = F.normalize(interaction_latent, p=2, dim=1)
        normalized_s = F.normalize(social_latent, p=2, dim=1)
    else:
        normalized_i, normalized_s = normalized_latents
    s_uv = (normalized_i[left] * normalized_s[right]).sum(dim=1)
    s_vu = (normalized_i[right] * normalized_s[left]).sum(dim=1)
    predicted_gap = s_uv - s_vu
    # Compare Bernoulli probabilities induced by the two directional gaps.
    loss = bernoulli_renyi_from_logits(
        target_logits, predicted_gap / temperature, order
    ).mean()
    diagnostics["role_target_gap_abs"] = target_gap.abs().mean().detach()
    diagnostics["role_predicted_gap_abs"] = predicted_gap.abs().mean().detach()
    return loss, diagnostics
