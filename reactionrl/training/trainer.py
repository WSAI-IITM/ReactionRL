"""Offline RL trainer for action prediction models.

Handles the training loop, validation, metric tracking, and model checkpointing
for actor, critic, and actor-critic model types.
"""
import copy
import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tabulate import tabulate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from reactionrl.config import TrainingConfig
from reactionrl.training.embedding_helpers import (
    get_action_dataset_embeddings,
    get_action_embedding_from_packed_molecule,
)
from reactionrl.training.ranking import get_ranking
from reactionrl.training.losses import compute_mse_actor_loss, compute_pg_actor_loss, compute_critic_loss


class OfflineRLTrainer:
    """Trains actor/critic/actor-critic models for offline RL action prediction.

    The trainer manages:
    - Training loop with actor and/or critic updates
    - Negative sampling for PG loss and critic training
    - Validation with ranking metrics (actor) or classification metrics (critic)
    - Best model tracking and checkpointing
    - Embedding model synchronization with the GIN backbone

    Note: Both actor and critic optimizers update all model parameters (including
    the shared GIN backbone). This is intentional for the shared-backbone
    actor-critic architecture.

    Args:
        model: An ActorNetwork, CriticNetwork, or ActorCritic instance.
        dataset: An OfflineRLDataset that has been prepared (prepare() called).
        config: TrainingConfig with hyperparameters.
    """
    def __init__(self, model, dataset, config: TrainingConfig):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = torch.device(config.device)

        self.train_actor = config.model_type in ["actor", "actor-critic"]
        self.train_critic = config.model_type in ["critic", "actor-critic"]

        # Optimizers - both use all model.parameters() because the GIN backbone
        # is shared and should be updated by both actor and critic gradients.
        if self.train_actor:
            self.actor_optimizer = torch.optim.Adam(model.parameters(), lr=config.actor_lr)
        if self.train_critic:
            self.critic_optimizer = torch.optim.Adam(model.parameters(), lr=config.critic_lr)
        self.critic_loss_criterion = nn.MSELoss()

        # For actor log prob calc (PG loss)
        if self.train_actor:
            self.actor_log_std = nn.Parameter(
                torch.zeros(model.actor.last_layer.out_features, dtype=torch.float32)
            ).to(self.device)

        # Embedding model for computing action embeddings - synced with model's GIN
        gin_path = config.get_gin_model_path()
        self.embedding_model = torch.load(gin_path).to(self.device)
        self.embedding_model.load_state_dict(model.GIN.state_dict())
        self.action_embeddings = get_action_dataset_embeddings(
            self.embedding_model, dataset.action_rsigs, dataset.action_psigs
        )
        self.action_embeddings_norm = torch.linalg.norm(self.action_embeddings, axis=1)

        # Best model tracking
        self.best_rank = float('inf')
        self.best_metric = -float('inf')
        self.best_model = None
        self.best_epoch = 0

        # Metric dicts
        self.actor_metric_dict = {
            "cos_rank_mean": [], "euc_rank_mean": [], "cos_rank_std": [], "euc_rank_std": [],
            "cos_rank_tot": [], "euc_rank_tot": [], "rmse": [], "cos_sim": [], "time(epoch_start-now)": [],
        }
        self.critic_metric_dict = {
            "GT_acc": [], "GT_rec": [], "GT_prec": [], "GT_f1": [],
            "others_acc": [], "others_rec": [], "others_prec": [], "others_f1": [],
            "mean_acc": [], "mean_rec": [], "mean_prec": [], "mean_f1": [], "time(epoch_start-now)": [],
        }

    def train_epoch(self, epoch, train_split, valid_split):
        """Run one epoch of training."""
        batch_size = self.config.batch_size
        topk = self.config.topk
        device = self.device

        train_reactants = train_split.reactants
        train_products = train_split.products
        train_rsigs = train_split.rsigs
        train_psigs = train_split.psigs
        train_idx = train_split.idx

        action_rsigs = self.dataset.action_rsigs
        action_psigs = self.dataset.action_psigs
        correct_action_dataset_indices = self.dataset.correct_action_dataset_indices
        action_embedding_indices = self.dataset.action_embedding_indices

        self.model.train()
        actor_loss = None
        critic_loss = None
        last_batch_i = 0

        for i in range(0, train_reactants.batch_size - batch_size, batch_size):
            last_batch_i = i
            # Forward pass
            actor_actions = self.model(
                train_reactants[i:i + batch_size], train_products[i:i + batch_size],
                train_rsigs[i:i + batch_size], train_psigs[i:i + batch_size], "actor"
            )

            if self.train_critic or (self.train_actor and self.config.actor_loss == "PG"):
                # Calc negatives - mix of random applicable + closest in embedding space
                negative_indices = []
                for _i in range(actor_actions.shape[0]):
                    correct_action_dataset_index = correct_action_dataset_indices[train_idx[i + _i]]
                    temp_list = []
                    # Add some random applicable actions
                    size = min(topk, action_embedding_indices[train_idx[i + _i]].shape[0])
                    temp_list.append(np.random.choice(action_embedding_indices[train_idx[i + _i]], size=(size // 2,), replace=False))
                    # Add closest (hardest) negatives
                    if self.train_actor:
                        curr_out = actor_actions[_i].detach()
                    else:
                        curr_out = self.action_embeddings[correct_action_dataset_index]
                    dist = torch.linalg.norm(self.action_embeddings - curr_out, axis=1)
                    sorted_idx = torch.argsort(dist)[:topk]
                    sorted_idx = sorted_idx[sorted_idx != correct_action_dataset_index]
                    temp_list.append(sorted_idx[:sorted_idx.shape[0] // 2].cpu().numpy())
                    negative_indices.append(np.concatenate(temp_list))

            # Critic update
            if self.train_critic:
                batch_reactants = train_reactants[sum([[i + _i] * (1 + negative_indices[_i].shape[0]) for _i in range(actor_actions.shape[0])], [])]
                batch_products = train_products[sum([[i + _i] * (1 + negative_indices[_i].shape[0]) for _i in range(actor_actions.shape[0])], [])]
                batch_rsigs = action_rsigs[sum([[correct_action_dataset_indices[train_idx[i + _i]]] + negative_indices[_i].tolist() for _i in range(actor_actions.shape[0])], [])]
                batch_psigs = action_psigs[sum([[correct_action_dataset_indices[train_idx[i + _i]]] + negative_indices[_i].tolist() for _i in range(actor_actions.shape[0])], [])]
                batch_q_targets = torch.Tensor(sum([[1] + [0] * negative_indices[_i].shape[0] for _i in range(actor_actions.shape[0])], [])).view(-1, 1)

                critic_loss = compute_critic_loss(
                    self.model, batch_reactants, batch_products, batch_rsigs, batch_psigs,
                    batch_q_targets, self.critic_loss_criterion, device
                )
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

            # Actor update
            if self.train_actor:
                actor_actions = self.model(
                    train_reactants[i:i + batch_size], train_products[i:i + batch_size],
                    train_rsigs[i:i + batch_size], train_psigs[i:i + batch_size], "actor"
                )
                if self.config.actor_loss == "mse":
                    actor_loss = compute_mse_actor_loss(
                        actor_actions, self.embedding_model, train_rsigs, train_psigs, i, batch_size
                    )
                elif self.config.actor_loss == "PG":
                    actor_loss = compute_pg_actor_loss(
                        actor_actions, self.embedding_model, train_rsigs, train_psigs,
                        self.action_embeddings, negative_indices, self.actor_log_std, topk, i, batch_size
                    )
                else:
                    raise ValueError(f"Unknown actor_loss: {self.config.actor_loss!r}. Use 'mse' or 'PG'.")

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

            torch.cuda.empty_cache()

        result_string = f'Epoch {epoch}/{self.config.epochs}. Batch {last_batch_i}/{train_reactants.batch_size - batch_size}.'
        if self.train_actor and actor_loss is not None:
            result_string += f' Actor loss = {actor_loss.item():.6f}'
        if self.train_critic and critic_loss is not None:
            result_string += f' || critic loss = {critic_loss.item():.6f}'
        print(result_string)

    def validate_actor(self, epoch, valid_split, start_time):
        """Validate actor: compute ranking metrics (euclidean + cosine)."""
        batch_size = self.config.batch_size
        valid_reactants = valid_split.reactants
        valid_products = valid_split.products
        valid_rsigs = valid_split.rsigs
        valid_psigs = valid_split.psigs
        valid_idx = valid_split.idx
        action_embedding_indices = self.dataset.action_embedding_indices
        correct_applicable_indices = self.dataset.correct_applicable_indices

        # Predictions
        pred = torch.concatenate([
            self.model(
                valid_reactants[i:i + batch_size], valid_products[i:i + batch_size],
                valid_rsigs[i:i + batch_size], valid_psigs[i:i + batch_size], "actor"
            ).detach()
            for i in range(0, valid_reactants.batch_size - batch_size, batch_size)
        ], axis=0)
        true = get_action_embedding_from_packed_molecule(
            self.embedding_model, valid_rsigs[:pred.shape[0]], valid_psigs[:pred.shape[0]]
        )

        metric_df = pd.DataFrame(columns=[
            "rmse", "cos_sim", "euc_rank_mean", "euc_rank_std", "euc_rank_tot",
            "cos_rank_mean", "cos_rank_std", "cos_rank_tot", "time(epoch_start-now)",
        ])

        self.actor_metric_dict["rmse"].append((((pred - true) ** 2).sum(axis=1) ** 0.5).mean().item())
        self.actor_metric_dict["cos_sim"].append(
            ((pred * true).sum(axis=1) / torch.linalg.norm(pred, axis=1) / torch.linalg.norm(true, axis=1)).mean().item()
        )

        for dist in ["euclidean", "cosine"]:
            l = []
            total = []
            for i in range(pred.shape[0]):
                pred_for_i = pred[i]
                act_emb_for_i = self.action_embeddings[action_embedding_indices[valid_idx[i]]]
                correct_applicable_index = correct_applicable_indices[valid_idx[i]]
                rank, _ = get_ranking(pred_for_i, act_emb_for_i, correct_applicable_index, distance=dist)
                l.append(rank.item())
                total.append(act_emb_for_i.shape[0])
            self.actor_metric_dict[f"{dist[:3]}_rank_mean"].append(np.mean(l))
            self.actor_metric_dict[f"{dist[:3]}_rank_std"].append(np.std(l))
            self.actor_metric_dict[f"{dist[:3]}_rank_tot"].append(np.mean(total))

        self.actor_metric_dict["time(epoch_start-now)"].append(f"{(time.time() - start_time) / 60:.2f} min")
        for col in metric_df.columns:
            metric_df[col] = [self.actor_metric_dict[col][-1]]
        metric_df.index = [epoch]
        print(tabulate(metric_df, headers='keys', tablefmt='fancy_grid'))
        print()

        # Update best model
        if self.actor_metric_dict["euc_rank_mean"][-1] < self.best_rank:
            self.best_rank = self.actor_metric_dict["euc_rank_mean"][-1]
            self.best_model = copy.deepcopy(self.model).cpu()
            self.best_epoch = epoch
            print(f"BEST MODEL UPDATED! BEST RANK = {self.best_rank}")

    def validate_critic(self, epoch, valid_split, start_time):
        """Validate critic: compute GT/others/mean classification metrics."""
        batch_size = self.config.batch_size
        device = self.device
        valid_reactants = valid_split.reactants
        valid_products = valid_split.products
        valid_rsigs = valid_split.rsigs
        valid_psigs = valid_split.psigs
        valid_idx = valid_split.idx
        action_rsigs = self.dataset.action_rsigs
        action_psigs = self.dataset.action_psigs
        correct_action_dataset_indices = self.dataset.correct_action_dataset_indices

        # Predict for ground truth actions
        GT_pred_qs = (torch.concatenate([
            self.model(
                valid_reactants[i:i + batch_size], valid_products[i:i + batch_size],
                valid_rsigs[i:i + batch_size], valid_psigs[i:i + batch_size], "critic"
            ).detach()
            for i in range(0, valid_reactants.batch_size - batch_size, batch_size)
        ], axis=0).cpu().numpy() > 0.5).astype(int)
        GT_true_qs = np.ones_like(GT_pred_qs)

        # Predict for closest negative actions
        negative_indices = []
        for i in valid_idx:
            correct_action_dataset_index = correct_action_dataset_indices[i]
            curr_out = self.action_embeddings[correct_action_dataset_index]
            dist = torch.linalg.norm(self.action_embeddings - curr_out, axis=1)
            sorted_idx = torch.argsort(dist)[:2]
            sorted_idx = sorted_idx[sorted_idx != correct_action_dataset_index]
            sorted_idx = sorted_idx[:1]
            negative_indices.append(sorted_idx)

        valid_batch_reactants = valid_reactants[sum([[i] * negative_indices[i].shape[0] for i in range(valid_idx.shape[0])], [])].to(device)
        valid_batch_products = valid_products[sum([[i] * negative_indices[i].shape[0] for i in range(valid_idx.shape[0])], [])].to(device)
        valid_batch_rsigs = action_rsigs[torch.concatenate(negative_indices)].to(device)
        valid_batch_psigs = action_psigs[torch.concatenate(negative_indices)].to(device)

        others_pred_qs = (torch.concatenate([
            self.model(
                valid_batch_reactants[i:i + batch_size], valid_batch_products[i:i + batch_size],
                valid_batch_rsigs[i:i + batch_size], valid_batch_psigs[i:i + batch_size], "critic"
            ).detach()
            for i in range(0, valid_batch_reactants.batch_size - batch_size, batch_size)
        ], axis=0).cpu().numpy() > 0.5).astype(int)
        others_true_qs = np.zeros_like(others_pred_qs)

        # GT metrics
        acc, (prec, rec, f1, _) = accuracy_score(GT_true_qs, GT_pred_qs), precision_recall_fscore_support(GT_true_qs, GT_pred_qs, average="binary")
        self.critic_metric_dict["GT_acc"].append(acc)
        self.critic_metric_dict["GT_rec"].append(rec)
        self.critic_metric_dict["GT_prec"].append(prec)
        self.critic_metric_dict["GT_f1"].append(f1)

        # Others metrics (invert labels for sklearn)
        acc, (prec, rec, f1, _) = accuracy_score(others_true_qs, others_pred_qs), precision_recall_fscore_support(1 - others_true_qs, 1 - others_pred_qs, average="binary")
        self.critic_metric_dict["others_acc"].append(acc)
        self.critic_metric_dict["others_rec"].append(rec)
        self.critic_metric_dict["others_prec"].append(prec)
        self.critic_metric_dict["others_f1"].append(f1)

        # Combined metrics
        mean_pred_qs = np.concatenate([GT_pred_qs, others_pred_qs], axis=0)
        mean_true_qs = np.concatenate([GT_true_qs, others_true_qs], axis=0)
        acc, (prec, rec, f1, _) = accuracy_score(mean_true_qs, mean_pred_qs), precision_recall_fscore_support(mean_true_qs, mean_pred_qs, average="binary")
        self.critic_metric_dict["mean_acc"].append(acc)
        self.critic_metric_dict["mean_rec"].append(rec)
        self.critic_metric_dict["mean_prec"].append(prec)
        self.critic_metric_dict["mean_f1"].append(f1)

        metric_df = pd.DataFrame(columns=[
            "GT_acc", "GT_rec", "GT_prec", "GT_f1", "others_acc", "others_rec", "others_prec", "others_f1",
            "mean_acc", "mean_rec", "mean_prec", "mean_f1", "time(epoch_start-now)",
        ])
        self.critic_metric_dict["time(epoch_start-now)"].append(f"{(time.time() - start_time) / 60:.2f} min")
        for col in metric_df.columns:
            metric_df[col] = [self.critic_metric_dict[col][-1]]
        metric_df.index = [epoch]
        print(tabulate(metric_df, headers='keys', tablefmt='fancy_grid'))
        print()

        # Update best model (by GT F1 - we want the critic that best identifies correct actions)
        curr_metric = self.critic_metric_dict["GT_f1"][-1]
        if curr_metric > self.best_metric:
            self.best_metric = curr_metric
            self.best_model = copy.deepcopy(self.model).cpu()
            self.best_epoch = epoch
            print(f"BEST MODEL UPDATED! BEST GT_f1 = {self.best_metric}")

    def train(self, train_split, valid_split):
        """Run the full training loop.

        Args:
            train_split: OfflineRLSplit namedtuple with training data.
            valid_split: OfflineRLSplit namedtuple with validation data.
        """
        config = self.config

        print(config)
        for epoch in range(1, config.epochs + 1):
            start_time = time.time()

            self.train_epoch(epoch, train_split, valid_split)

            # VALIDATION
            self.model.eval()
            with torch.no_grad():
                d = {
                    "steps": config.steps, "model_type": config.model_type,
                    "actor_loss": config.actor_loss, "negative_selection": config.negative_selection,
                    "seed": config.seed, "device": config.device,
                }
                margin_string = "# " + " || ".join([f"{k}--{d[k]}" for k in d]) + " #"
                print("#" * len(margin_string))
                print(margin_string)
                print("#" * len(margin_string))

                if self.train_actor:
                    self.validate_actor(epoch, valid_split, start_time)

                # Critic-only validation (when actor is also trained, actor
                # validation already covers the key ranking metrics)
                if self.train_critic and not self.train_actor:
                    self.validate_critic(epoch, valid_split, start_time)

                # Sync embedding model with updated GIN backbone
                self.embedding_model.load_state_dict(self.model.GIN.state_dict())
                self.action_embeddings = get_action_dataset_embeddings(
                    self.embedding_model, self.dataset.action_rsigs, self.dataset.action_psigs
                )
                self.action_embeddings_norm = torch.linalg.norm(self.action_embeddings, axis=1)

    def save(self, folder=None):
        """Save model, metrics, and config to disk.

        Args:
            folder: Output directory. Defaults to output/supervised/{model_type}/...
        """
        config = self.config
        if folder is None:
            folder = os.path.join(
                "output", "supervised", config.model_type,
                f"steps={config.steps}_actor_loss={config.actor_loss}_neg={config.negative_selection}_seed={config.seed}"
            )
        os.makedirs(folder, exist_ok=True)

        if self.train_actor:
            metric_dict = self.actor_metric_dict

            fig = plt.figure(figsize=(8, 8))
            for dist in filter(lambda x: "mean" in x, metric_dict.keys()):
                plt.plot(metric_dict[dist], label=dist)
            plt.title(f"Offline RL (steps={config.steps})")
            plt.xlabel("epoch")
            plt.ylabel("ranking")
            plt.legend()
            fig.savefig(os.path.join(folder, "plot.png"))
            plt.close(fig)
        else:
            metric_dict = self.critic_metric_dict

        torch.save(self.model, os.path.join(folder, "model.pth"))
        pd.DataFrame.from_dict(metric_dict).to_csv(os.path.join(folder, "metrics.csv"))
        with open(os.path.join(folder, "config.json"), 'w') as f:
            json.dump({
                "steps(trajectory length)": config.steps,
                "actor_lr": config.actor_lr,
                "critic_lr": config.critic_lr,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "topk": config.topk,
                "best_epoch": self.best_epoch,
                "best_rank": self.best_rank,
            }, f, indent=2)
        print("Saved model at", folder)
