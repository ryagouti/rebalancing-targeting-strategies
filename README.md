# Rebalancing Targeting Strategies for Multi-Treatment Experiments

This repository contains the implementation and simulation code accompanying the paper:

**On Response-Adaptive Targeting Strategies for Multi-Treatment Experiments**
*Redouane Yagouti, Rémy Degenne, and Emilie Kaufmann*
https://arxiv.org/abs/2606.17777

## Overview

This repository implements the family of **α-Rebalancing Targeting Strategies (αRTS)** introduced in the paper, together with their **Forced Exploration (αRTS-FE)** variants. The code reproduces the empirical studies presented in the manuscript, including:

* Convergence to target allocations;
* Comparison of different αRTS instantiations;
* Sparse target allocation experiments;
* Forced-exploration analysis;
* Hypothesis testing under adaptive sampling.

The repository provides a complete simulation framework for multi-treatment response-adaptive randomization experiments.

---

## Repository Structure

```text
.
├── adaptive_designs.py        # αRTS and αRTS-FE implementations
├── allocation_functions.py    # Target allocation functions
├── main_algo.py               # Core simulation engine
├── trajectory_manager.py      # Allocation trajectory tracking
├── table_manager.py           # Generation of summary tables
├── utils.py                   # Utility functions

├── empirical_study.ipynb      # Reproduces all experiments from the paper

├── tables/                    # Generated numerical results
└── plots/                     # Generated figures
```

---

## Implemented Designs

The repository includes several members of the αRTS family (and their corresponding forced exploration FE variants):

* Distance-Based Allocation
* ERADE2025
* Interpolated D-Tracking

---

## Reproducing the Experiments

The main experiments presented in Section 6 of the paper can be reproduced directly from the notebook:

```bash
jupyter notebook empirical_study.ipynb
```

The notebook contains:

1. Convergence experiments for Neyman allocation targets.
2. Sparse allocation experiments using Tymofyeyev targets.
3. Comparisons between αRTS and αRTS-FE.
4. Hypothesis testing experiments under adaptive sampling.
5. Generation of all figures and summary tables.

---

## Outputs

### Figures and Tables

All generated plots and tables are saved in:

```text
plots/
tables/
```
---

## License

This project is released under the MIT License. See the LICENSE file for details.
