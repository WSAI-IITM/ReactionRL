# PURE: Policy-guided Unbiased REpresentations for structure-constrained molecular generation

PURE uses:
1. A Graph Isomorphism Network (GIN) to encode molecular structures as graphs.
2. Template-based molecular simulations, extracted from the USPTO-MIT reaction database, which constrain exploration to chemically valid, synthesizable reactions.
3. A policy-guided actor-critic RL setup, where molecular similarity naturally emerges from the learned representations rather than being hard-coded through external metrics.

For more details, please refer to
**Gupta, A., Lenin, B., Current, S., Batra, R., Ravindran, B., Raman, K., & Parthasarathy, S. (2025). PURE: Policy-guided Unbiased REpresentations for structure-constrained molecular generation. bioRxiv, 2025-05.** ([PDF](https://www.biorxiv.org/content/10.1101/2025.05.21.655002v1.full.pdf)).

---

## Repository Structure

```
reactionrl/                  # Main package
├── actions/                 # ActionSpace: find & apply molecular transformations
├── embeddings/              # GIN and MPNN molecular embedders
├── models/                  # Actor, Critic, ActorCritic networks
├── rewards/                 # Property scorers (logP, QED, DRD2, SA, similarity)
├── data/                    # Dataset loading & trajectory generation
├── generation/              # Beam search molecule generation (Algorithm 2)
├── training/                # Trainer, losses, ranking metrics
├── evaluation/              # PURE vs baseline comparison metrics
├── preprocessing/           # Data pipeline scripts (USPTO extraction)
├── scripts/                 # CLI entry points (train, generate_data, generate_molecules, evaluate)
├── utils/                   # Shared molecule utilities
└── config.py                # Paths & TrainingConfig dataclass
pretrained_models/           # GIN and MPNN pretrained weights
datasets/                    # Action dataset & training data
notebooks/                   # Experimental notebooks
```

---

## Requirements

Python 3.8 with the following core dependencies:

```
torch
torchdrug
rdkit
pandas
numpy
networkx
scikit-learn
tabulate
tqdm
filehash
matplotlib
```

Optional (for MPNN embedder):
```
deepchem
dgl
```

Install via conda:
```bash
conda env create -f environment.yml
conda activate reactionrl
```

Or install manually:
```bash
conda create -n reactionrl python=3.8
conda activate reactionrl
pip install torch==2.1.1
pip install torchdrug==0.2.1 rdkit-pypi pandas numpy networkx scikit-learn tabulate tqdm filehash matplotlib
```

---

## Quick Start

### 1. Preprocess the action dataset (one-time setup)

Extract reaction templates from USPTO-MIT:

```bash
bash reactionrl/scripts/preprocess.sh
```

This runs the full pipeline: download transformations, extract reaction centres and signatures, generate and filter the action dataset, dump starting molecules, and compute action embeddings.

### 2. Generate training data

Roll out a random policy to collect (reactant, action, product) trajectories:

```bash
python -m reactionrl.scripts.generate_data --steps 5 --train-samples 100000
```

Options:
- `--steps`: Number of transformation steps per trajectory
- `--train-samples`: Number of training samples to generate (default: 100,000)
- `--processes`: Number of parallel workers (default: 80% of CPU cores)

### 3. Train an offline RL model

```bash
python -m reactionrl.scripts.train --steps 5 --model-type actor-critic --actor-loss PG --cuda 0
```

Options:
- `--steps`: Trajectory length (must match the generated data)
- `--model-type`: `actor`, `critic`, or `actor-critic`
- `--actor-loss`: `mse` or `PG` (policy gradient with negative sampling)
- `--epochs`: Number of training epochs (default: 50)
- `--cuda`: GPU index (-1 for CPU, default: -1)
- `--seed`: Random seed (default: 42)
- `--negative-selection`: `random`, `closest`, `e-greedy`, or `combined`
- `--num-workers`: Parallel workers for data preparation

### 4. Generate molecules (Algorithm 2 beam search)

Given a trained model, generate molecules similar to targets:

```bash
python -m reactionrl.scripts.generate_molecules \
    --model-path output/supervised/actor-critic/steps=5_actor_loss=PG_neg=combined_seed=42/model.pth \
    --source-smiles "c1ccc(-c2ccccc2)cc1" \
    --target-smiles "c1ccc(O)cc1" \
    --steps 5 --topk-actor 50 --topk-critic 5 \
    --output results/generated.pickle --cuda 0
```

Options:
- `--model-path`: Path to trained model checkpoint
- `--source-smiles` / `--source-file`: Starting molecule(s)
- `--target-smiles` / `--target-file`: Target molecule(s)
- `--steps`: Number of generation steps (default: 5)
- `--topk-actor`: Actor pre-filter count B_A (default: 50)
- `--topk-critic`: Beam width B after critic re-ranking (default: 5)
- `--num-workers`: Parallel workers (default: 8)

### 5. Evaluate on COMA benchmarks

Evaluate the trained model on QED, DRD2, pLogP04, or pLogP06 benchmarks:

```bash
python -m reactionrl.scripts.evaluate \
    --model-path output/supervised/actor-critic/steps=5_actor_loss=PG_neg=combined_seed=42/model.pth \
    --property qed --cuda 0
```

Options:
- `--property`: Benchmark to evaluate (`qed`, `drd2`, `logp04`, `logp06`)
- `--num-start-mols`: Starting molecules per target (default: 10, selects 3x this)
- `--num-decode`: Top molecules per target for metrics (default: 20)
- `--max-targets`: Limit number of test targets (useful for quick testing)
- `--steps`, `--topk-actor`, `--topk-critic`, `--num-workers`: Same as generation

**Benchmark test data** must be downloaded separately:

1. Clone: `git clone https://github.com/wengong-jin/iclr19-graph2graph`
2. Copy test files:
   ```bash
   for prop in qed drd2 logp04 logp06; do
       mkdir -p datasets/coma/$prop
       cp iclr19-graph2graph/data/$prop/test.txt datasets/coma/$prop/rdkit_test.txt
   done
   ```

The script will print download instructions if test data is missing.

---

## How to Experiment

The codebase is modular -- here's how to swap or modify components:

### Swap the embedding model

All models use a GIN backbone loaded from `pretrained_models/zinc2m_gin.pth`. To use a different pretrained model:

```python
from reactionrl.config import TrainingConfig

config = TrainingConfig(gin_model_path="/path/to/your/model.pth")
```

To implement a completely new embedder, subclass `BaseEmbeddingClass`:

```python
from reactionrl.embeddings.base import BaseEmbeddingClass

class MyEmbedder(BaseEmbeddingClass):
    def mol_to_embedding(self, mol):
        ...
    def atom_to_embedding(self, mol, idx):
        ...
```

### Change model architecture

Architecture parameters are configurable:

```python
from reactionrl.models import ActorCritic

model = ActorCritic(
    gin_model_path="pretrained_models/zinc2m_gin.pth",
    actor_num_hidden=4,      # default: 3
    critic_num_hidden=3,     # default: 2
    hidden_size=512,         # default: 256
)
```

Or use `TrainingConfig`:

```python
config = TrainingConfig(
    hidden_size=512,
    actor_num_hidden=4,
    critic_num_hidden=3,
)
```

### Add a new model type

1. Create a new `nn.Module` in `reactionrl/models/` with a `.GIN` attribute and `.actor` property
2. Register it in `reactionrl/models/__init__.py`:

```python
MODEL_REGISTRY["my-model"] = MyModel
```

### Try different reward functions

Property scorers are in `reactionrl/rewards/`:

```python
from reactionrl.rewards import logP, qed, drd2, SA, similarity
```

### Use the generation module programmatically

```python
from reactionrl.generation import generate_molecules, prepare_action_data
import torch

# Load trained model
model = torch.load("output/supervised/actor-critic/.../model.pth", weights_only=False)
model.eval()

# Prepare action data (one-time)
action_dataset, action_rsigs, action_psigs = prepare_action_data()

# Generate molecules
traj_dict, sim_dict = generate_molecules(
    model,
    source_smiles=["c1ccc(-c2ccccc2)cc1"],
    target_smiles=["c1ccc(O)cc1"],
    action_rsigs=action_rsigs,
    action_psigs=action_psigs,
    device=torch.device("cuda:0"),
    steps=5,
    topk_actor=50,
    topk_critic=5,
)
# traj_dict maps trajectory keys to SMILES
# sim_dict maps trajectory keys to Tanimoto similarity to target
```

### Use the action space programmatically

```python
from rdkit import Chem
from reactionrl.actions import get_applicable_actions, apply_action

mol = Chem.MolFromSmiles("c1ccc(-c2ccccc2)cc1")  # biphenyl
actions = get_applicable_actions(mol)
print(f"{actions.shape[0]} applicable actions")

# Apply the first action
product = apply_action(mol, *actions.iloc[0])
print(Chem.MolToSmiles(product))
```

### Customize training

```python
from reactionrl.config import TrainingConfig
from reactionrl.models import ActorCritic
from reactionrl.data.dataset import OfflineRLDataset
from reactionrl.training import OfflineRLTrainer

config = TrainingConfig(
    steps=5,
    model_type="actor-critic",
    actor_loss="PG",
    epochs=100,
    batch_size=256,
    actor_lr=1e-4,
    device="cuda:0",
)

dataset = OfflineRLDataset("datasets/offlineRL/5steps_train.csv", device="cuda:0")
dataset.prepare(num_workers=config.num_workers)
train_split, valid_split = dataset.split(train_frac=config.train_frac)

model = ActorCritic(gin_model_path=config.get_gin_model_path())
model = model.to("cuda:0")

trainer = OfflineRLTrainer(model, dataset, config)
trainer.train(train_split, valid_split)
trainer.save("output/my_experiment")
```

---

## Key Concepts

**Action Space**: Molecular transformations are defined as reaction signature pairs (rsig -> psig) extracted from USPTO. Given a molecule, the `ActionSpace` identifies which transformations are applicable by matching substructures via articulation point decomposition.

**Offline RL Training**: The actor network learns to predict the correct action embedding (concatenated rsig + psig GIN embeddings) given a (reactant, product) pair. The critic network learns to score whether a given action is correct for a transition. Policy gradient loss with negative sampling trains the actor to distinguish correct actions from similar but incorrect ones.

**Evaluation**: Ranking metrics (euclidean and cosine distance) measure how well the predicted action embedding ranks the correct action among all applicable actions. Lower rank = better prediction.

---

## Citation

```bibtex
@article{gupta2025pure,
  title={PURE: Policy-guided Unbiased REpresentations for structure-constrained molecular generation},
  author={Gupta, Abhor and Lenin, Bhargav and Current, Sean and Batra, Ritwik and Ravindran, Balaraman and Raman, Karthik and Parthasarathy, Srinivasan},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.05.21.655002}
}
```

