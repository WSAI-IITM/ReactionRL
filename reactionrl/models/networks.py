"""Neural network architectures for offline RL action prediction.

Contains the Actor, Critic, and ActorCritic models that use a shared GIN
backbone for molecular graph encoding, with dense heads for action prediction
(actor) and action scoring (critic).
"""
import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    """Feed-forward network with batch normalization.

    Architecture: Linear -> BN -> ReLU -> [Hidden layers] -> Linear output.
    Each hidden layer is Linear -> BN -> ReLU.
    """
    def __init__(self, input_size, output_size, num_hidden=1, hidden_size=50):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_layers.append(nn.BatchNorm1d(hidden_size))
            self.hidden_layers.append(nn.ReLU())

        self.last_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        for layer in self.hidden_layers:
            out = layer(out)
        out = self.last_layer(out)
        return out


class ActorNetwork(nn.Module):
    """Actor network: predicts action embeddings from (reactant, product) pairs.

    Takes reactant and product molecular graphs, encodes them via a shared GIN,
    and passes the concatenated embeddings through a dense network to predict
    the action embedding (concatenated rsig + psig GIN embeddings).

    Args:
        gin_model_path: Path to pretrained GIN model (.pth file).
        num_hidden: Number of hidden layers in the dense head.
        hidden_size: Hidden layer width.
    """
    def __init__(self, gin_model_path, num_hidden=3, hidden_size=256):
        super(ActorNetwork, self).__init__()
        self.GIN = torch.load(gin_model_path, weights_only=False)
        self.DENSE = NeuralNet(self.GIN.output_dim * 2, self.GIN.output_dim * 2,
                               num_hidden=num_hidden, hidden_size=hidden_size)

    @property
    def actor(self):
        return self.DENSE

    def forward(self, x1, x2, *args):
        out1 = self.GIN(x1, x1.node_feature.float())["graph_feature"]
        out2 = self.GIN(x2, x2.node_feature.float())["graph_feature"]

        out = torch.concatenate([out1, out2], axis=1)
        out = self.DENSE(out)
        return out


class CriticNetwork(nn.Module):
    """Critic network: scores whether an action is correct for a transition.

    Takes reactant, product, rsig, and psig molecular graphs, encodes all four
    via a shared GIN, and passes the concatenated embeddings through a dense
    network to produce a scalar score.

    Args:
        gin_model_path: Path to pretrained GIN model (.pth file).
        num_hidden: Number of hidden layers in the dense head.
        hidden_size: Hidden layer width.
    """
    def __init__(self, gin_model_path, num_hidden=2, hidden_size=256):
        super(CriticNetwork, self).__init__()
        self.GIN = torch.load(gin_model_path, weights_only=False)
        self.DENSE = NeuralNet(self.GIN.output_dim * 4, 1,
                               num_hidden=num_hidden, hidden_size=hidden_size)

    def forward(self, x1, x2, x3, x4, *args):
        out1 = self.GIN(x1, x1.node_feature.float())["graph_feature"]
        out2 = self.GIN(x2, x2.node_feature.float())["graph_feature"]
        out3 = self.GIN(x3, x3.node_feature.float())["graph_feature"]
        out4 = self.GIN(x4, x4.node_feature.float())["graph_feature"]

        out = torch.concatenate([out1, out2, out3, out4], axis=1)
        out = self.DENSE(out)
        return out


class ActorCritic(nn.Module):
    """Combined actor-critic with shared GIN backbone.

    The actor head predicts action embeddings from (reactant, product).
    The critic head scores actions from (reactant, product, rsig, psig).
    Both share the same GIN encoder, so backbone gradients come from both heads.

    Args:
        gin_model_path: Path to pretrained GIN model (.pth file).
        actor_num_hidden: Number of hidden layers in actor head.
        critic_num_hidden: Number of hidden layers in critic head.
        hidden_size: Hidden layer width for both heads.
    """
    def __init__(self, gin_model_path, actor_num_hidden=3, critic_num_hidden=2, hidden_size=256):
        super(ActorCritic, self).__init__()
        self.GIN = torch.load(gin_model_path, weights_only=False)
        self.actor = NeuralNet(self.GIN.output_dim * 2, self.GIN.output_dim * 2,
                               num_hidden=actor_num_hidden, hidden_size=hidden_size)
        self.critic = NeuralNet(self.GIN.output_dim * 4, 1,
                                num_hidden=critic_num_hidden, hidden_size=hidden_size)

    def forward(self, reac, prod, rsig, psig, out_type="both"):
        """Forward pass with selectable output.

        Args:
            reac: Packed reactant molecules.
            prod: Packed product molecules.
            rsig: Packed reactant signature molecules (used by critic).
            psig: Packed product signature molecules (used by critic).
            out_type: "actor" returns action embeddings, "critic" returns Q-values,
                "both" returns [action_embeddings, q_values].
        """
        reac_out = self.GIN(reac, reac.node_feature.float())["graph_feature"]
        prod_out = self.GIN(prod, prod.node_feature.float())["graph_feature"]

        output = []
        if out_type in ["both", "actor"]:
            output.append(self.actor(torch.concatenate([reac_out, prod_out], axis=1)))

        if out_type in ["both", "critic"]:
            psig_out = self.GIN(psig, psig.node_feature.float())["graph_feature"]
            rsig_out = self.GIN(rsig, rsig.node_feature.float())["graph_feature"]
            output.append(self.critic(torch.concatenate([reac_out, prod_out, rsig_out, psig_out], axis=1)))

        if len(output) == 1:
            return output[0]
        return output
