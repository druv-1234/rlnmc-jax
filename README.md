# Reinforcement Learning Nonlocal Monte Carlo (RLNMC)
A JAX implementation of algorithms (SA/NMC/RLNMC) for the paper ["Nonlocal Monte Carlo via Reinforcement Learning"](https://arxiv.org/abs/2508.10520).

## Installation
Install dependencies from requirements.txt to your virtual environment (may install the CPU version of JAX, modify for CUDA, if available).
```
pip install -r requirements.txt
```
PS. The multi-device and/or large problem size use of Gurobi can require a special license. WandB logging requires an account.

## Running
Modify files in `scripts` for your needs. For example, benchmarking of Uniform Random 4-SAT problems of size `N=500` from the paper is carried out as:
```
chmod +x scripts/bench/uf500.sh
./scripts/bench/uf500.sh
```
(can use many "devices": GPUs recommended). Configuration files in `configs` are adjustable.


## Citation
The bibtex citation for the RLNMC paper is:
```
@misc{dobrynin2025nmc,
      title={Nonlocal Monte Carlo via Reinforcement Learning}, 
      author={Dmitrii Dobrynin and Masoud Mohseni and John Paul Strachan},
      year={2025},
      eprint={2508.10520},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.10520}, 
}
```
