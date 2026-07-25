# AAAI-27 Experiment Code

Five research projects targeting AAAI-27 main conference. Each subdirectory contains standalone Python experiment scripts and a COMPUTE_CONTEXT.txt describing resource requirements.

## Projects

| Directory | Project | GPU Hours (est.) | Peak GPU Memory |
|-----------|---------|-----------------|-----------------|
| `behavfm/` | Identity-Token Adaptation for cross-species pose estimation | ~625 A100-eq | 16 GB (feature extraction) |
| `cell2circuit/` | Pareto-front audit of connectome graph generators | ~54 H100-eq | 16 GB (DiGress training) |
| `conflict/` | Matched-null audit of mechanistic interpretability claims | ~35 H100-eq | 32 GB (Llama-3.1-8B) |
| `intervenefm/` | Counterfactual audit of per-gene perturbation embeddings | ~15 H100-eq | 16 GB (GEARS/CPA training) |
| `reflex-rlvr/` | Discrimination premise in self-teacher RLVR | ~130 H100-eq | 80 GB (72B models) |

**Total estimated GPU compute: ~860 H100-equivalent hours across all projects.**

## Environment

- Python 3.10+
- PyTorch 2.x (CUDA or ROCm)
- See each project's COMPUTE_CONTEXT.txt for specific dependencies

## Running

Each script uses argparse. Example:

```bash
cd behavfm/
python precompute_all_backbones.py --dataset ap10k --backbone dinov2 --output-dir ./features
python run_cross_backbone.py --feature-dir ./features --output-dir ./results --seeds 10
```

All scripts save results as JSON files. No external orchestration (Modal, SLURM) required.

## Data

Data is not included in this repository. Each COMPUTE_CONTEXT.txt lists required datasets and approximate sizes. Most datasets are publicly available (AP-10K, MICrONS, FlyWire, Norman, Replogle, AIME).

## Contact

Ethan Wang (planetpastrywasnottakenew@gmail.com)
