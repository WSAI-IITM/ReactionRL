"""Ranking metrics for evaluating action prediction quality.

Used during validation to measure how well the model's predicted action
embedding ranks compared to the true action among all applicable actions.
"""
import torch


def get_ranking(pred, emb_for_comparison, correct_index, distance="euclidean", k=None):
    """Compute the rank of the correct action among candidates.

    Args:
        pred: Predicted embedding, shape (dim,).
        emb_for_comparison: Candidate embeddings, shape (N, dim).
        correct_index: Index of the correct action in emb_for_comparison.
        distance: Distance metric - "euclidean" or "cosine".
        k: If provided, return top-k indices instead of rank.

    Returns:
        If k is None: (rank, distances_up_to_rank) where rank is 1-indexed.
        If k is not None: top-k sorted indices by distance.
    """
    if distance == "euclidean":
        dist = ((emb_for_comparison - pred) ** 2).sum(axis=1)
    elif distance == "cosine":
        dist = 1 - torch.mm(emb_for_comparison, pred.view(-1, 1)).view(-1) / (
            torch.linalg.norm(emb_for_comparison, axis=1) * torch.linalg.norm(pred)
        )

    sorted_idx = dist.argsort()
    rank = (dist[sorted_idx] == dist[correct_index]).nonzero()[0] + 1
    list_of_distances = dist[sorted_idx[:rank]]

    if k is not None:
        return sorted_idx[:k]
    return rank, list_of_distances


def get_top_k_indices(pred, emb_for_comparison, correct_index, distance="euclidean", k=1):
    """Return the top-k nearest action indices by distance.

    Args:
        pred: Predicted embedding, shape (dim,).
        emb_for_comparison: Candidate embeddings, shape (N, dim).
        correct_index: Index of the correct action (unused but kept for API consistency).
        distance: Distance metric - "euclidean" or "cosine".
        k: Number of top candidates to return.

    Returns:
        Tensor of k indices sorted by distance to pred.
    """
    return get_ranking(pred, emb_for_comparison, correct_index, distance, k)
