import torch
import numpy as np
from data import DataLoader


def fix_random_seed(seed: int) -> None:
    """Seed NumPy and PyTorch random number generators."""
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def iter_minibatch(*tensors, batch_size: int = 1024):
    if len(tensors) == 1:
        t = tensors[0]
        for i in range(0, len(t), batch_size):
            yield t[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def evaluate_allranking(dataset: DataLoader,
                        model,
                        device,
                        Ks,
                        batch_size_eval: int):
    """Rank the full item catalog after masking each user's training items."""
    model.eval()
    users = list(dataset.test_dict)
    if not users:
        raise ValueError("Ranking evaluation requires at least one held-out user.")
    results = {name: np.zeros(len(Ks)) for name in ('precision', 'recall', 'ndcg')}
    max_k = max(Ks)

    with torch.no_grad():
        for batch_users in iter_minibatch(users, batch_size=batch_size_eval):
            exclude_users, exclude_items = [], []
            all_positive = dataset.get_user_pos_items(batch_users)
            ground_true = [dataset.test_dict[u] for u in batch_users]

            batch_users_device = torch.as_tensor(batch_users, dtype=torch.long, device=device)
            rating = model.evaluation_scores(batch_users_device)
            for i, items in enumerate(all_positive):
                exclude_users.extend([i] * len(items))
                exclude_items.extend(items)
            rating[exclude_users, exclude_items] = -1e9
            top_items = torch.topk(rating, k=max_k).indices.cpu()
            del rating

            batch_metrics = eval_one_batch((top_items, ground_true), Ks)
            for name in results:
                results[name] += batch_metrics[name]

    # Each batch contributes per-user sums, including the final partial batch.
    for name in results:
        results[name] /= len(users)
    return results


def eval_one_batch(enum_item, cutoffs=(10, 20, 30, 50, 100)):
    """Sum per-user metrics at each requested cutoff."""
    recommender_items = enum_item[0].numpy()
    ground_truth_items = enum_item[1]
    r = build_label(ground_truth_items, recommender_items)
    precision, recall, ndcg = [], [], []
    for k in cutoffs:
        recall.append(recall_at_k(r, k, ground_truth_items))
        precision.append(precision_at_k(r, k, ground_truth_items))
        ndcg.append(ndcg_at_k(r, k, ground_truth_items))
    return {
        'recall':   np.array(recall),
        'precision':np.array(precision),
        'ndcg':     np.array(ndcg)
    }


def ndcg_at_k(r, k, test_data):
    assert len(r) == len(test_data)
    prediction_data = r[:, :k]
    test_matrix = np.zeros((len(prediction_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = np.sum(prediction_data * (1. / np.log2(np.arange(2, k + 2))), axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def recall_at_k(r, k, test_data):
    right_prediction = r[:, :k].sum(1)
    recall_num = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_prediction / recall_num)
    return recall


def precision_at_k(r, k, test_data):
    right_prediction = r[:, :k].sum(1)
    precision_num = k
    precision = np.sum(right_prediction) / precision_num
    return precision


def build_label(true_data, pred_data):
    """Mark recommended items that belong to each user's held-out targets."""
    r = []
    for i in range(len(true_data)):
        ground_true = true_data[i]
        pred_top_k = pred_data[i]
        pred = list(map(lambda x: x in ground_true, pred_top_k))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype("float")

