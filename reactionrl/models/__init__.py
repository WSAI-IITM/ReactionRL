from reactionrl.models.networks import NeuralNet, ActorNetwork, CriticNetwork, ActorCritic

MODEL_REGISTRY = {
    "actor": ActorNetwork,
    "critic": CriticNetwork,
    "actor-critic": ActorCritic,
}
