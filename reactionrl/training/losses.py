import torch
import torch.nn as nn

from reactionrl.training.embedding_helpers import get_action_embedding_from_packed_molecule


def compute_mse_actor_loss(actor_actions, embedding_model, train_rsigs, train_psigs, i, batch_size):
    """MSE loss between predicted and true action embeddings."""
    return nn.MSELoss()(
        actor_actions,
        get_action_embedding_from_packed_molecule(
            embedding_model, train_rsigs[i:i + batch_size], train_psigs[i:i + batch_size]
        ),
    )


def compute_pg_actor_loss(actor_actions, embedding_model, train_rsigs, train_psigs,
                          action_embeddings, negative_indices, actor_log_std, topk, i, batch_size):
    """Policy gradient actor loss with negative sampling."""
    normal_dist = torch.distributions.Normal(actor_actions, actor_log_std.exp())
    positives = get_action_embedding_from_packed_molecule(
        embedding_model, train_rsigs[i:i + batch_size], train_psigs[i:i + batch_size]
    )
    positive_log_pi = normal_dist.log_prob(positives)
    negative_log_pi = []
    for _i, _indices in enumerate(negative_indices):
        normal_dist = torch.distributions.Normal(actor_actions[_i], actor_log_std.exp())
        negative_log_pi.append(normal_dist.log_prob(action_embeddings[_indices]))
    negative_log_pi = torch.concatenate(negative_log_pi, axis=0)

    # Using R = 1 for positives, and R = -1/2topk for negatives
    actor_loss = torch.concatenate(
        [-positive_log_pi, (1 / (topk * 2)) * negative_log_pi], axis=0
    ).sum(-1, keepdim=True).mean()
    return actor_loss


def compute_critic_loss(model, batch_reactants, batch_products, batch_rsigs, batch_psigs,
                        batch_q_targets, criterion, device):
    """Binary classification loss for critic between correct and negative actions."""
    critic_qs = model(
        batch_reactants.to(device), batch_products.to(device),
        batch_rsigs.to(device), batch_psigs.to(device), "critic"
    )
    return criterion(critic_qs, batch_q_targets.to(device))
