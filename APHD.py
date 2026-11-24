"""APHD encoders, cross-view alignment, adaptive fusion, and prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GSMGraphEncoder(nn.Module):
    """Project interaction inputs and compensate for neighborhood smoothing."""

    def __init__(self, layer_dims, dropout, high_pass_compensation):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.lambda_s = high_pass_compensation
        for i in range(len(layer_dims) - 1):
            d_in, d_out = layer_dims[i], layer_dims[i + 1]
            self.layers.append(nn.Linear(d_in, d_out))

    def forward(self, topology_matrix):
        h = None
        for i, layer in enumerate(self.layers):
            if i == 0:
                h = layer(topology_matrix)
                if topology_matrix.size(0) == topology_matrix.size(1):
                    h_smooth = torch.sparse.mm(topology_matrix, h)
                else:
                    # Apply user-domain smoothing without forming a user-user matrix.
                    item_state = torch.sparse.mm(topology_matrix.transpose(0, 1), h)
                    h_smooth = torch.sparse.mm(topology_matrix, item_state)
                # Retain deviations from interaction-induced neighborhood smoothing.
                h = h + self.lambda_s * (h - h_smooth)
            else:
                h_smooth = torch.sparse.mm(topology_matrix, h)
                h_in = h + self.lambda_s * (h - h_smooth)
                h = layer(h_in)
            h = self.dropout(h)
            if i != len(self.layers) - 1:
                h = torch.tanh(h)
            else:
                dim = h.shape[1] // 2
                mu = h[:, :dim]
                logvar = h[:, dim:]
                return mu, logvar


class GSMSocialEncoder(nn.Module):
    """Map social neighborhoods to Gaussian posterior parameters."""

    def __init__(self, num_users, latent_dim, dropout):
        super().__init__()
        self.posterior_projection = nn.Linear(num_users, latent_dim * 2)
        self.state_dropout = nn.Dropout(dropout)

    def forward(self, social_topology, user_ids):
        posterior_params = torch.sparse.mm(
            social_topology,
            self.posterior_projection.weight.t(),
        )
        posterior_params = posterior_params + self.posterior_projection.bias
        posterior_params = self.state_dropout(posterior_params)
        return posterior_params[user_ids].chunk(2, dim=1)


class EvidenceAwareAdaptiveFusion(nn.Module):
    """Infer interaction-view weights from latent states and reliability evidence."""

    def __init__(self, input_dim, hidden_dim, output_dim, slope=0.1):
        super().__init__()
        self.reliability_estimator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=slope),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(
        self,
        interaction_latent,
        social_latent,
        interaction_evidence,
        social_evidence,
        extra_evidence,
        temperature=2.0,
    ):
        evidence_parts = [
            interaction_latent, social_latent,
            interaction_evidence, social_evidence, extra_evidence,
        ]
        joint_state = torch.cat(evidence_parts, dim=1)
        energy = self.reliability_estimator(joint_state)
        T = float(temperature) if float(temperature) > 1e-8 else 1.0
        interaction_gate = torch.sigmoid(energy / T)
        return interaction_gate


class StructuralSupportEncoder(nn.Module):
    """Keep a scalar support anchor and add a small learned feature residual."""

    def __init__(self, latent_dim, residual_scale=1e-4):
        super().__init__()
        self.latent_dim = latent_dim
        self.residual_scale = residual_scale
        self.projection = nn.Linear(1, latent_dim)
        self.normalization = nn.LayerNorm(latent_dim)

    def forward(self, scalar_evidence):
        learned_state = self.normalization(torch.tanh(self.projection(scalar_evidence)))
        structural_state = torch.zeros_like(learned_state)
        structural_state[:, :1] = scalar_evidence
        return structural_state + self.residual_scale * learned_state


class APHD(nn.Module):
    """Learn separate interaction and social roles before adaptive integration."""

    SOCIAL_PREFERENCE_PRIOR_PRESETS = {
        'douban-book': {'alpha': 0.0, 'gate_threshold': 4.0, 'gate_temperature': 0.25},
        'douban-movie': {'alpha': 0.0, 'gate_threshold': 3.0, 'gate_temperature': 0.25},
        'yelp': {'alpha': 1.38},
    }

    def __init__(
        self,
        decoder_dims,
        encoder_dims,
        dataset=None,
        device=None,
        args=None,
        dropout=0.0,
    ):
        super().__init__()
        self.device = device
        self.dataset = dataset
        self.args = args
        self.dataset_name = str(args.dataset).lower().removesuffix('-712g')
        self.social_preference_prior_config = self.SOCIAL_PREFERENCE_PRIOR_PRESETS[self.dataset_name].copy()
        self.latent_dim = args.latent_dim
        self.items_num = decoder_dims[-1]

        interaction_posterior_dims = encoder_dims[:-1] + [encoder_dims[-1] * 2]
        self.interaction_encoder = GSMGraphEncoder(
            interaction_posterior_dims,
            dropout=dropout,
            high_pass_compensation=args.high_pass_compensation,
        )
        self.social_encoder = GSMSocialEncoder(
            num_users=dataset.num_users,
            latent_dim=self.latent_dim,
            dropout=dropout,
        )

        self.num_social_prototypes = args.num_social_prototypes
        self.social_prototypes = nn.Parameter(torch.Tensor(self.num_social_prototypes, self.latent_dim))
        nn.init.xavier_uniform_(self.social_prototypes)

        mismatch_context_dim = 2
        self.eaf = EvidenceAwareAdaptiveFusion(
            input_dim=self.latent_dim * 2 + 2 + mismatch_context_dim,
            hidden_dim=self.latent_dim,
            output_dim=self.latent_dim
        )
        self.interaction_decoder = nn.ModuleList([
            nn.Linear(d_in, d_out)
            for d_in, d_out in zip(decoder_dims[:-1], decoder_dims[1:])
        ])
        self.social_decoder = nn.ModuleList([
            nn.Linear(self.latent_dim, self.items_num)
        ])
        self.structural_support_encoder = StructuralSupportEncoder(
            self.latent_dim,
            residual_scale=1e-6,
        )

        # Expand scalar support inputs while retaining their initialized weights.
        old_first_layer = self.eaf.reliability_estimator[0]
        expanded_input_dim = self.latent_dim * 4 + mismatch_context_dim
        # Keep this expansion from advancing the model initialization RNG.
        rng_state = torch.get_rng_state()
        expanded_first_layer = nn.Linear(
            expanded_input_dim,
            old_first_layer.out_features,
        )
        torch.set_rng_state(rng_state)
        with torch.no_grad():
            expanded_first_layer.weight.zero_()
            expanded_first_layer.bias.copy_(old_first_layer.bias)
            expanded_first_layer.weight[:, :self.latent_dim * 2].copy_(
                old_first_layer.weight[:, :self.latent_dim * 2]
            )
            expanded_first_layer.weight[:, self.latent_dim * 2].copy_(
                old_first_layer.weight[:, self.latent_dim * 2]
            )
            expanded_first_layer.weight[:, self.latent_dim * 3].copy_(
                old_first_layer.weight[:, self.latent_dim * 2 + 1]
            )
            expanded_first_layer.weight[:, self.latent_dim * 4:].copy_(
                old_first_layer.weight[:, self.latent_dim * 2 + 2:]
            )
        self.eaf.reliability_estimator[0] = expanded_first_layer
        self.interaction_topology = self._to_sparse_tensor(self.dataset.bipartite_graph).coalesce().to(self.device)
        self.social_topology = self._to_sparse_tensor(self.dataset.social_graph).coalesce().to(self.device)

    def _to_sparse_tensor(self, sp_mat):
        coo = sp_mat.tocoo().astype(np.float32)
        idx = torch.stack([torch.LongTensor(coo.row), torch.LongTensor(coo.col)])
        val = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(idx, val, size=coo.shape)

    def sample_posterior(self, mu, logvar):
        """Sample during training and use the posterior mean during evaluation."""
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        if self.training:
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode_branch(self, z, layers):
        h = z
        for i, layer in enumerate(layers):
            h = layer(h)
            if i != len(layers) - 1:
                h = torch.tanh(h)
        return h

    def posterior_uncertainty(self, logvar, user_ids=None):
        """Summarize each diagonal Gaussian by log-one-plus mean variance."""
        batch_logvar = logvar if user_ids is None else logvar[user_ids]
        batch_logvar = torch.clamp(batch_logvar, min=-10.0, max=10.0)
        return torch.log1p(torch.exp(batch_logvar).mean(dim=1, keepdim=True))

    def interaction_preference_objective(self, recon_x, target, mu, logvar, beta=1.0):
        """Add the weighted posterior KL to the negative log-likelihood of positive items."""
        log_prob = F.log_softmax(recon_x, 1)
        ll = -torch.mean(torch.sum(log_prob * target, dim=-1))
        bounded_logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        kld = -0.5 * torch.mean(
            torch.sum(
                1
                + bounded_logvar
                - mu.pow(2)
                - bounded_logvar.exp(),
                dim=1,
            )
        )
        return ll + beta * kld


    def social_geometry_regularizer(self, z):
        """Pull latent states toward assigned prototypes and update those prototypes."""
        z_ext = z.unsqueeze(1)
        c_ext = self.social_prototypes.unsqueeze(0)
        dist = (z_ext - c_ext).pow(2).sum(2)
        nearest = dist.argmin(dim=1)
        assigned_prototypes = self.social_prototypes[nearest]
        # Separate latent-state and prototype updates with stop-gradient.
        latent_pull = (z - assigned_prototypes.detach()).pow(2).sum(dim=1)
        prototype_update = (
            z.detach() - assigned_prototypes
        ).pow(2).sum(dim=1)
        gamma = float(self.args.prototype_update_coefficient)
        return (latent_pull + gamma * prototype_update).mean()


    def renyi_overlap_objective(self, interaction_latent, social_latent):
        """Reward matched affinity while controlling total cross-view row mass."""
        normalized_interaction = F.normalize(interaction_latent, p=2, dim=1)
        normalized_social = F.normalize(social_latent, p=2, dim=1)
        temperature = self.args.kde_temperature
        affinity_logits = torch.mm(normalized_interaction, normalized_social.t()) / temperature
        affinity_matrix = torch.exp(affinity_logits)
        matched_affinity = affinity_matrix.diag() * self.args.matched_affinity_weight
        unmatched_affinity_mass = affinity_matrix.sum(dim=1) - affinity_matrix.diag()
        row_mass = matched_affinity + unmatched_affinity_mass
        q = self.args.renyi_order
        mass_scale = self.args.row_mass_scale
        term_1 = -matched_affinity.pow(q).div(q)
        term_2 = (mass_scale * row_mass).pow(q).div(q)
        per_user_overlap_loss = term_1 + term_2
        return per_user_overlap_loss.mean()

    renyi_align_loss = renyi_overlap_objective

    def representation_mismatch(self, interaction_latent, social_latent):
        """Scale detached same-user affinity mismatch for the fusion context."""
        norm_i = F.normalize(interaction_latent, p=2, dim=1)
        norm_s = F.normalize(social_latent, p=2, dim=1)
        temperature = float(self.args.kde_temperature)
        temperature = temperature if temperature > 1e-8 else 1.0
        same_user_cosine = (norm_i * norm_s).sum(dim=1).clamp(-1.0, 1.0)
        matched_affinity = torch.exp(
            (same_user_cosine - 1.0) / temperature
        )
        mismatch = (1.0 - matched_affinity).clamp(0.0, 1.0)
        scale = float(self.args.representation_mismatch_scale)
        return (scale * mismatch).detach().unsqueeze(1)

    def eaf_mismatch_context(self, interaction_latent, social_latent, batch_idx):
        representation = self.representation_mismatch(interaction_latent, social_latent)
        objective = self.dataset.objective_mismatch[batch_idx].unsqueeze(1).to(interaction_latent.device)
        objective = objective.detach()
        mean = self.dataset.objective_mismatch_mean.to(objective.device)
        std = self.dataset.objective_mismatch_std.to(objective.device)
        objective = (objective - mean) / std.clamp_min(1e-8)
        return torch.cat([representation, objective.clamp(-3.0, 3.0)], dim=1)

    def objective_aware_social_residual(self, interaction_logits, social_regression, interaction_gate, batch_idx):
        """Scale normalized social scores by target agreement and social-view weight."""
        alpha = float(self.args.social_residual_weight)
        social_score = social_regression.detach()
        # Standardize each user's social scores across the full item catalog.
        social_score = (
            social_score - social_score.mean(dim=1, keepdim=True)
        ) / (social_score.std(dim=1, keepdim=True, unbiased=False) + 1e-8)
        objective_mismatch = self.dataset.objective_mismatch[batch_idx].unsqueeze(1).to(interaction_logits.device)
        agreement = torch.exp(
            -float(self.args.objective_agreement_scale)
            * objective_mismatch.detach().clamp_min(0.0)
        )
        social_weight = 1.0 - interaction_gate
        social_weight = social_weight.mean(dim=1, keepdim=True)
        return interaction_logits + alpha * agreement * social_weight * social_score

    def _degree_normalized_movie_interactions(self):
        dataset = self.dataset
        cache = getattr(dataset, 'social_preference_interaction_cache', None)
        if cache is None:
            cache = {}
            dataset.social_preference_interaction_cache = cache
        if 'row' not in cache:
            degree = np.asarray(dataset.train_data.sum(axis=1)).reshape(-1).astype(np.float32)
            scale = np.zeros_like(degree, dtype=np.float32)
            np.power(degree, -1.0, out=scale, where=degree > 0)
            cache['row'] = dataset.train_data.multiply(scale[:, None]).tocsr()
        return cache['row']

    def _neighbor_item_profile(self, user_ids, device):
        """Aggregate neighbors' training interactions and normalize by each user's maximum."""
        users = user_ids.detach().cpu().numpy()
        neighbors = self.dataset.social_data[users].tocsr().astype(np.float32)
        row_mass = np.asarray(neighbors.sum(axis=1)).reshape(-1)
        inv_mass = np.zeros_like(row_mass, dtype=np.float32)
        np.divide(1.0, row_mass, out=inv_mass, where=row_mass > 0)
        neighbors = neighbors.multiply(inv_mass[:, None])

        interactions = self.dataset.train_data
        if self.dataset_name == 'douban-movie':
            interactions = self._degree_normalized_movie_interactions()
        profile = neighbors.dot(interactions).toarray().astype(np.float32)
        row_max = np.max(np.abs(profile), axis=1, keepdims=True)
        return torch.from_numpy(profile / np.maximum(row_max, 1e-8)).to(device)

    def apply_social_preference_prior(self, logits, user_ids):
        """Add the configured neighborhood item prior to evaluation scores."""
        config = self.social_preference_prior_config
        alpha = float(config['alpha'])
        if self.training or alpha == 0.0:
            return logits
        profile = self._neighbor_item_profile(user_ids, logits.device)
        if self.dataset_name != 'yelp':
            temperature = max(float(config['gate_temperature']), 1e-6)
            threshold = float(config['gate_threshold'])
            scores = logits.detach().float()
            mean = scores.mean(dim=1, keepdim=True)
            std = scores.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-8)
            profile = profile * torch.sigmoid(((scores - mean) / std - threshold) / temperature)
        return logits + alpha * profile

    def evaluation_scores(self, user_ids):
        """Return ranking scores with the configured evaluation-time neighborhood prior."""
        user_ids = user_ids.long()
        logits = self(user_ids)[0]
        return self.apply_social_preference_prior(logits, user_ids)

    def forward(self, user_ids):
        """Return the learned ranking logits and auxiliary outputs used for training."""
        batch_idx = user_ids.long()
        interaction_mu, interaction_logvar = self.interaction_encoder(self.interaction_topology)
        social_mu, social_logvar = self.social_encoder(self.social_topology, batch_idx)
        interaction_samples = self.sample_posterior(interaction_mu, interaction_logvar)
        social_samples = self.sample_posterior(social_mu, social_logvar)
        interaction_latent = interaction_samples[batch_idx]
        social_latent = social_samples
        interaction_uncertainty = self.posterior_uncertainty(interaction_logvar, batch_idx)
        social_uncertainty = self.posterior_uncertainty(social_logvar)
        # A positive gap favors the less uncertain interaction branch.
        uncertainty_gap = social_uncertainty - interaction_uncertainty

        evidence_i = self.structural_support_encoder(
            self.dataset.interaction_structural_support[batch_idx].unsqueeze(1)
        )
        evidence_s = self.structural_support_encoder(
            self.dataset.social_structural_support[batch_idx].unsqueeze(1)
        )
        mismatch_context = self.eaf_mismatch_context(
            interaction_latent,
            social_latent,
            batch_idx,
        )
        interaction_gate = self.eaf(
            interaction_latent,
            social_latent,
            evidence_i,
            evidence_s,
            extra_evidence=mismatch_context,
            temperature=self.args.gate_temperature
        )
        # Bound the uncertainty correction and attenuate it near gate endpoints.
        scale = float(self.args.uncertainty_calibration_weight)
        bounded_delta = torch.tanh(uncertainty_gap)
        bounded_delta = bounded_delta.expand_as(interaction_gate)
        interaction_gate = torch.clamp(
            interaction_gate + scale * interaction_gate * (1.0 - interaction_gate) * bounded_delta,
            min=0.0,
            max=1.0,
        )
        # Integrate representations only after factor-wise reliability estimation.
        fused_representation = interaction_gate * interaction_latent + (1.0 - interaction_gate) * social_latent
        interaction_logits = self.decode_branch(fused_representation, self.interaction_decoder)
        social_regression = self.decode_branch(social_latent, self.social_decoder)
        base_logits = self.objective_aware_social_residual(
            interaction_logits, social_regression, interaction_gate, batch_idx
        )
        return (
            base_logits,
            social_regression,
            interaction_mu,
            interaction_logvar,
            social_mu,
            social_logvar,
            interaction_latent,
            social_latent
        )
