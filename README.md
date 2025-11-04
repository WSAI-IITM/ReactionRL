# PURE: Policy-guided Unbiased REpresentations for structure-constrained molecular generation

PURE uses:
 1. A Graph Isomorphism Network (GIN) to encode molecular structures as graphs.  
2. Template-based molecular simulations, extracted from the USPTO-MIT reaction database, which constrain exploration to chemically valid, synthesizable reactions.  
3. A policy-guided actor–critic RL setup, where molecular similarity naturally emerges from the learned representations rather than being hard-coded through external metrics.  

For more details, please refer to  
**Gupta, A., Lenin, B., Current, S., Batra, R., Ravindran, B., Raman, K., & Parthasarathy, S. (2025). PURE: Policy-guided Unbiased REpresentations for structure-constrained molecular generation. bioRxiv, 2025-05.** ([[link](https://www.biorxiv.org/content/10.1101/2025.05.21.655002v1.full.pdf)](https://www.biorxiv.org/content/10.1101/2025.05.21.655002v1.full.pdf)). 

## Contents of repo
This repo contains two applications:
1. Drug discovery using a gymnasium-compatible RL simulator (Online RL)
2. Lead optimization without using similarity-based metrics (Goal-conditioned RL + offline RL)

_The code for PURE corresponds to the second application._


# Requirements
This repo was built using python=3.7.  
Common requirements:
```
deepchem
notebook
pandas
RDKit
filehash
pytorch
networkx
torchdrug
````

#### Requirements for drug discovery
```
gymnasium
tensorboard
stable_baselines3
```


#### Requirements for lead optimization
```
tabulate
matplotlib
```

# Usage
#### Extract action dataset from USPTO-MIT
```
./preprocess.sh
```

#### Molecular Discovery
The gymnasium environment for molecular discovery is contained in folder `molecular_discovery`. Folder `sb3` contains 4 example \<agents\> = [ppo/sac/td3/ddpg]. For molecular discovery using stable_baselines3, run 

```
python -m sb3.<agent> --timesteps 1000000 mode train --reward [logp\qed\drd2\SA]
```

#### Lead optimization
```
# generate some offline data by rolling out a random policy.
python lead_optimization.dump_data_for_offlineRL --train 100000 --steps 5

# train an offline RL agent
python lead_optimization.python offlineRL.py --steps 5 --model actor-critic --actor-loss PG --cuda 0 
```

