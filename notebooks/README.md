# Notebooks

These notebooks are from the original development of PURE and use old import
paths (`from action_utils import *`, `from offlineRL_utils import *`, etc.)
that no longer exist after the package restructure.

To use them, update imports to the new `reactionrl.*` module paths. For example:

```python
# Old
from action_utils import apply_action, get_applicable_actions
from offlineRL_utils import *
from rewards.properties import logP, qed

# New
from reactionrl.actions import apply_action, get_applicable_actions
from reactionrl.models import ActorNetwork, CriticNetwork, ActorCritic
from reactionrl.training.embedding_helpers import get_mol_embedding
from reactionrl.rewards import logP, qed
```
