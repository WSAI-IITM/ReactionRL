"""ReactionRL: Offline reinforcement learning for molecular lead optimization.

Key classes:
    - ActionSpace: Find and apply molecular transformation actions
    - OfflineRLTrainer: Train actor/critic models for action prediction
    - TrainingConfig: Hyperparameter configuration

Quick start:
    from reactionrl.config import TrainingConfig
    from reactionrl.actions import ActionSpace, get_applicable_actions, apply_action
    from reactionrl.data.dataset import OfflineRLDataset
    from reactionrl.training import OfflineRLTrainer
    from reactionrl.models import MODEL_REGISTRY
"""
__version__ = "1.0.0"
