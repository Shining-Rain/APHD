# APHD: Asymmetric Personalized Heterogeneous Dual-Graph Recommendation.
import argparse
import os
import time
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.optim as optim

import batch_test
import data as d
from APHD import APHD


DATASETS = ("douban-book", "douban-movie", "yelp")
MAX_EPOCHS = {
    "douban-book": 2000,
    "douban-movie": 10000,
    "yelp": 250,
}

MONITOR_PATIENCE = {
    "douban-book": 300,
    "douban-movie": 500,
    "yelp": 5,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=DATASETS + ("all",),
        default="douban-book",
    )
    parser.add_argument("--path", default="./Data/")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--monitor_interval", type=int, default=10)
    parser.add_argument("--monitor_patience", type=int, default=None)
    parser.add_argument("--monitor_min_epochs", type=int, default=0)
    return parser.parse_args()


def model_config(dataset):
    """Default training and model hyperparameters."""
    name = dataset.removesuffix("-712g")
    is_yelp = name == "yelp"
    return SimpleNamespace(
        dataset=dataset,
        learning_rate=0.004 if is_yelp else 0.005,
        latent_dim=320 if is_yelp else 140,
        weight_decay=0.0,
        batch_size=1024,
        dropout=0.7 if name == "douban-book" else 0.6,
        interaction_kl_weight=1e-3,
        high_pass_compensation=0.2,
        num_social_prototypes=10,
        social_geometry_weight=0.01 if is_yelp else 0.1,
        prototype_update_coefficient=0.1,
        alignment_weight=2e-3,
        renyi_order=0.3,
        row_mass_scale=0.1,
        kde_temperature=2.0,
        matched_affinity_weight=1.4,
        social_kl_weight=0.003 if is_yelp else 0.1,
        gate_temperature=1.5,
        uncertainty_calibration_weight=0.1,
        social_residual_weight=0.16,
        objective_agreement_scale=2.0,
        representation_mismatch_scale=1e-6,
    )


def normalize_rows(matrix):
    """Normalize nonempty rows while preserving zero rows."""
    matrix = matrix.tocsr().astype(np.float32)
    row_sum = np.asarray(matrix.sum(axis=1)).reshape(-1)
    inverse = np.zeros_like(row_sum, dtype=np.float32)
    np.divide(1.0, row_sum, out=inverse, where=row_sum > 0)
    return sp.diags(inverse).dot(matrix).tocsr()


def structural_support(matrix, device):
    """Scale log-transformed row sums by their maximum within the graph."""
    degree = np.asarray(matrix.sum(axis=1)).reshape(-1)
    log_degree = np.log1p(degree)
    support = log_degree / (log_degree.max() + 1e-8)
    return torch.from_numpy(support).float().to(device)


def weight_social_edges_by_support(matrix):
    """Downweight encoder edges to neighbors with high incoming support."""
    matrix = matrix.tocoo().astype(np.float32)
    incoming = np.asarray(
        sp.csr_matrix(matrix).sum(axis=0)
    ).reshape(-1).astype(np.float32)
    mean_incoming = max(float(incoming.mean()), 1e-8)
    weights = np.power(
        1.0 + incoming[matrix.col] / mean_incoming,
        -0.5,
    ).astype(np.float32)
    return sp.csr_matrix(
        (weights, (matrix.row, matrix.col)),
        shape=matrix.shape,
        dtype=np.float32,
    )


def build_neighborhood_target_adjacency(train_matrix, social_matrix):
    """Build binary social adjacency for uniformly averaged neighbor targets."""
    adjacency = social_matrix.tocsr().astype(np.float32, copy=True)
    adjacency.sum_duplicates()
    adjacency.eliminate_zeros()
    adjacency.data.fill(1.0)
    # A nonempty neighborhood is valid even when its target is zero.
    valid = np.diff(adjacency.indptr) > 0
    return adjacency, valid


def social_regression_objective(prediction, target, valid):
    """Average squared errors over valid users and items; return zero if empty."""
    if not torch.any(valid):
        return prediction.sum() * 0.0
    return F.mse_loss(prediction[valid], target[valid])


def compute_objective_mismatch(train_matrix, social_matrix):
    """Compute cosine mismatch between interaction and neighborhood targets."""
    social_target = normalize_rows(social_matrix).dot(train_matrix).tocsr().astype(np.float32)
    train_matrix = train_matrix.tocsr().astype(np.float32)
    dot = np.asarray(train_matrix.multiply(social_target).sum(axis=1)).reshape(-1)
    train_norm = np.sqrt(
        np.asarray(train_matrix.multiply(train_matrix).sum(axis=1)).reshape(-1)
    )
    target_norm = np.sqrt(
        np.asarray(social_target.multiply(social_target).sum(axis=1)).reshape(-1)
    )
    denominator = train_norm * target_norm
    cosine = np.zeros_like(dot, dtype=np.float32)
    np.divide(dot, denominator, out=cosine, where=denominator > 1e-8)
    mismatch = 1.0 - np.clip(cosine, 0.0, 1.0)
    # A missing target supplies no evidence of objective agreement.
    mismatch[train_norm <= 1e-8] = 1.0
    mismatch[target_norm <= 1e-8] = 1.0
    return mismatch.astype(np.float32)


def dense_tensor(matrix, device):
    return torch.from_numpy(matrix.toarray()).float().to(device)


def sparse_tensor(matrix, device):
    matrix = matrix.tocoo()
    indices = torch.from_numpy(
        np.vstack((matrix.row, matrix.col))
    ).long()
    values = torch.from_numpy(matrix.data).float()
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=matrix.shape,
        device=device,
    ).coalesce()


def prepare_stage(dataset, path, seed, *, refit=True):
    """Build the stage from training interactions, adding validation for refit."""
    batch_test.fix_random_seed(seed)
    name = dataset.removesuffix("-712g")
    data_dir = os.path.join(path, name)
    if not os.path.isfile(os.path.join(data_dir, "val.txt")) and os.path.isdir(data_dir + "-712g"):
        data_dir += "-712g"
    loader = d.DataLoader(data_dir, include_validation=refit and name == dataset)
    # Uniform neighbor targets and weighted encoder edges use separate matrices.
    target_social_data, target_valid = build_neighborhood_target_adjacency(
        loader.train_data, loader.social_data
    )
    loader.social_data = weight_social_edges_by_support(loader.social_data)
    loader.social_graph = normalize_rows(loader.social_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = loader.train_data
    social_data = loader.social_data
    users_num, items_num = train_data.shape
    config = model_config(dataset)
    # Preserve the batch count associated with the original user ID range.
    raw_batch_count = max(
        1,
        int(np.ceil(loader.raw_num_users / float(config.batch_size))),
    )
    train_batch_size = max(
        1,
        int(np.ceil(users_num / float(raw_batch_count))),
    )
    # Structural support uses interaction counts and weighted social row sums.
    loader.interaction_structural_support = structural_support(train_data, device)
    loader.social_structural_support = structural_support(social_data, device)
    objective_mismatch = compute_objective_mismatch(train_data, target_social_data)
    loader.objective_mismatch = torch.from_numpy(objective_mismatch).float().to(device)
    loader.objective_mismatch_mean = loader.objective_mismatch.mean()
    loader.objective_mismatch_std = loader.objective_mismatch.std(
        unbiased=False
    ).clamp_min(1e-8)
    model = APHD(
        decoder_dims=[config.latent_dim, items_num],
        encoder_dims=[items_num, config.latent_dim],
        dataset=loader,
        device=device,
        args=config,
        dropout=config.dropout,
    ).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100] if name == "yelp" else [], gamma=0.25,
    )
    interaction_matrix = sparse_tensor(train_data, device)
    # Reuse this permutation throughout the training stage.
    user_order = np.arange(users_num)
    np.random.shuffle(user_order)
    return SimpleNamespace(
        loader=loader,
        device=device,
        train_data=train_data,
        social_data=social_data,
        target_social_data=target_social_data,
        target_valid=torch.from_numpy(target_valid).to(device),
        users_num=users_num,
        items_num=items_num,
        train_batch_size=train_batch_size,
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        interaction_matrix=interaction_matrix,
        user_order=user_order,
    )


def train_epoch(stage):
    stage.model.train()
    objective_sum = 0.0
    batch_count = 0
    started = time.time()
    for start in range(0, stage.users_num, stage.train_batch_size):
        end = min(start + stage.train_batch_size, stage.users_num)
        batch_users = stage.user_order[start:end]
        users = torch.from_numpy(batch_users).long().to(stage.device)
        batch_interactions = stage.train_data[batch_users]
        stage.optimizer.zero_grad()
        (
            interaction_logits,
            social_regression,
            interaction_mu,
            interaction_logvar,
            social_mu,
            social_logvar,
            interaction_latent,
            social_latent,
        ) = stage.model(users)
        interaction_loss = stage.model.interaction_preference_objective(
            interaction_logits,
            dense_tensor(batch_interactions, stage.device),
            interaction_mu,
            interaction_logvar,
            stage.config.interaction_kl_weight,
        )
        # Average neighbors' training interactions with uniform social weights.
        batch_social = stage.target_social_data[batch_users].tocsr()
        social_row_sum = np.asarray(batch_social.sum(axis=1)).reshape(-1)
        social_inverse = np.zeros_like(social_row_sum, dtype=np.float32)
        np.divide(
            1.0,
            social_row_sum,
            out=social_inverse,
            where=social_row_sum > 0,
        )
        normalized_social = batch_social.multiply(
            social_inverse[:, None]
        ).tocsr()
        social_target = torch.sparse.mm(
            sparse_tensor(normalized_social, stage.device),
            stage.interaction_matrix,
        ).to_dense()
        social_regression_loss = social_regression_objective(
            social_regression,
            social_target,
            stage.target_valid[users],
        )
        bounded_social_logvar = torch.clamp(
            social_logvar,
            min=-10.0,
            max=10.0,
        )
        social_kl = -0.5 * torch.mean(
            torch.sum(
                1
                + bounded_social_logvar
                - social_mu.pow(2)
                - bounded_social_logvar.exp(),
                dim=1,
            )
        )
        geometry_loss = stage.model.social_geometry_regularizer(
            social_latent
        )
        alignment_loss = stage.model.renyi_align_loss(interaction_latent, social_latent)
        loss = (
            interaction_loss
            + social_regression_loss
            + stage.config.social_kl_weight * social_kl
            + stage.config.social_geometry_weight * geometry_loss
            + stage.config.alignment_weight * alignment_loss
        )
        if not torch.isfinite(loss):
            values = {
                "interaction": float(interaction_loss.detach().cpu()),
                "social_regression": float(social_regression_loss.detach().cpu()),
                "social_kl": float(social_kl.detach().cpu()),
                "geometry": float(geometry_loss.detach().cpu()),
                "alignment": float(alignment_loss.detach().cpu()),
            }
            raise FloatingPointError(f"Non-finite APHD objective: {values}")
        loss.backward()
        if stage.config.dataset.removesuffix("-712g") == "douban-book":
            torch.nn.utils.clip_grad_norm_(
                stage.model.parameters(),
                max_norm=5.0,
            )
        stage.optimizer.step()
        objective_sum += float(loss.item())
        batch_count += 1
    stage.scheduler.step()
    return objective_sum / max(batch_count, 1), time.time() - started


def evaluate(stage, evaluation_dict):
    """Evaluate held-out targets while masking the stage's training items."""
    original_dict = stage.loader.test_dict
    stage.loader.test_dict = evaluation_dict
    try:
        return batch_test.evaluate_allranking(
            stage.loader,
            stage.model,
            stage.device,
            [10, 20, 30, 50, 100],
            stage.config.batch_size,
        )
    finally:
        stage.loader.test_dict = original_dict


def select_epoch(
    dataset,
    path,
    seed,
    max_epochs,
    monitor_interval,
    monitor_patience,
    monitor_min_epochs,
):
    """Select an epoch on the validation split using Recall@20 and NDCG@20."""
    stage = prepare_stage(dataset, path, seed, refit=False)
    validation_dict = stage.loader.load_eval_dict("val.txt")
    best_epoch = 0
    best_score = -float("inf")
    best_recall = -float("inf")
    best_ndcg = -float("inf")
    # Patience counts validation checks, not training epochs.
    bad_checks = 0
    print(
        f"[Selection] dataset={dataset} "
        f"users={stage.users_num} batch={stage.train_batch_size} "
        f"max_epochs={max_epochs}"
    )
    for epoch in range(1, max_epochs + 1):
        objective, epoch_time = train_epoch(stage)
        should_monitor = (
            epoch % max(monitor_interval, 1) == 0
            or epoch == max_epochs
        )
        if epoch == 1 or epoch % 100 == 0:
            print(
                f"[Selection {epoch:04d}/{max_epochs:04d}] "
                f"objective={objective:.6f} time={epoch_time:.2f}s"
            )
        if not should_monitor:
            continue
        result = evaluate(stage, validation_dict)
        recall20 = float(result["recall"][1])
        ndcg20 = float(result["ndcg"][1])
        score = (recall20 * ndcg20) ** 0.5
        print(
            f"[Validation {epoch:04d}] "
            f"R@20={recall20:.6f} N@20={ndcg20:.6f}"
        )
        if score > best_score:
            best_score = score
            best_recall = recall20
            best_ndcg = ndcg20
            best_epoch = epoch
            bad_checks = 0
        else:
            bad_checks += 1
        if (
            epoch >= monitor_min_epochs
            and bad_checks >= monitor_patience
        ):
            break
    if best_epoch <= 0:
        raise RuntimeError("Validation did not produce a selected epoch.")
    print(
        f"[Selection complete] dataset={dataset} "
        f"best_epoch={best_epoch} best_R@20={best_recall:.6f} "
        f"best_N@20={best_ndcg:.6f}"
    )
    return best_epoch


def refit_and_test(dataset, path, seed, selected_epoch):
    """Train a fresh model on the merged split, then evaluate the test set once."""
    stage = prepare_stage(dataset, path, seed, refit=True)
    print(
        f"[Refit] dataset={dataset} users={stage.users_num} "
        f"batch={stage.train_batch_size} epochs={selected_epoch}"
    )
    for epoch in range(1, selected_epoch + 1):
        objective, epoch_time = train_epoch(stage)
        if epoch == 1 or epoch % 100 == 0 or epoch == selected_epoch:
            print(
                f"[Refit {epoch:04d}/{selected_epoch:04d}] "
                f"objective={objective:.6f} time={epoch_time:.2f}s"
            )
    result = evaluate(stage, stage.loader.test_dict)
    print(f"[Test once] dataset={dataset}")
    for index, cutoff in enumerate((10, 20, 30, 50, 100)):
        print(
            f"@{cutoff}: Recall={result['recall'][index]:.5f} "
            f"NDCG={result['ndcg'][index]:.5f}"
        )
    return result


def run_dataset(dataset, cli_args):
    max_epochs = (
        cli_args.max_epochs
        if cli_args.max_epochs is not None
        else MAX_EPOCHS[dataset]
    )
    monitor_patience = (
        cli_args.monitor_patience
        if cli_args.monitor_patience is not None
        else MONITOR_PATIENCE[dataset]
    )
    selected_epoch = select_epoch(
        dataset,
        cli_args.path,
        cli_args.seed,
        max_epochs,
        cli_args.monitor_interval,
        monitor_patience,
        cli_args.monitor_min_epochs,
    )
    return refit_and_test(
        dataset,
        cli_args.path,
        cli_args.seed,
        selected_epoch,
    )


def main():
    cli_args = parse_args()
    datasets = DATASETS if cli_args.dataset == "all" else (cli_args.dataset,)
    results = {}
    for dataset in datasets:
        results[dataset] = run_dataset(dataset, cli_args)
    return results


if __name__ == "__main__":
    main()
