

# Decentralized-Constrained-Optimization

This repository contains the code associated with the paper <a href="https://arxiv.org/abs/2210.03232">"Double Averaging and Gradient Projection: Convergence Guarantees for Decentralized Constrained Optimization,"</a> organized into distinct parts that correspond to different experiments of the paper. The codebase is divided into three main folders: `Feasible_parameters`, `LR_Synthetic`, and `OT`. Below is a detailed description of each part, including how to run the experiments and the structure of the folders.

## Repository Structure

### Feasible_parameters

This folder is dedicated to identifying a feasible region of parameters for the DAGP algorithm, ensuring guaranteed convergence. It includes:

- `main.m`: A MATLAB script designed to symbolically compute the determinant of the transfer matrix. This script rigorously checks the conditions necessary for matrices to be considered proper, thus establishing feasibility for the parameters.
- `detF4.mat`: This file contains the computed determinant of the transfer matrix when \(M=4\), as a result of executing the first cell in `main.m`. 


### LR_Synthetic

This folder contains the code for generating the first three figures in our paper and is structured as follows:

- `Final_synthetic.py`: Generates Figures 1 and 2 by simulating synthetic functions scenarios.
- `Final_LR.py`: Produces Figure 3, focusing on logistic regression problem.

Supporting folders within `LR_Synthetic` include:

- `utilities/`: Contains utility functions.
- `Problems/`: Includes problem definitions, gradients, and projections.
- `data/`: Stores datasets like MNIST used in experiments.
- `Optimizers/`: Holds implementations of algorithms.
- `graph/`: Contains scripts for generating gossip matrices.
- `analysis/`: Provides functions for performance metrics.

### OT

This section covers the experiments related to Decentralized Optimal Transport, including:

- `OT_Time.py`: A script essential for replicating Figure 4 in the paper. It is designed to be run multiple times with varying dimensions to simulate different scenarios. The script takes parameters for source and target dimensions (--n_source and --n_target) and the number of simulations (--n_sims). For example usage, refer to the run_vm.sh script, which automates the process of executing OT_Time.py with a range of dimension values to efficiently generate the necessary data.
- `plot_time.py`: Generates Figure 4 from the collected data.
- `OT_DA.ipynb`: A Jupyter notebook detailing the Optimal Transport for domain adaptation experiment.

## Dependencies and Environment Setup

The `requirements.txt` file in this repository lists all the libraries and packages installed in the Anaconda environment used for developing and testing the experiments. It is important to note that this list may include more packages than are strictly necessary for running the experiments, as it reflects the entire environment setup.

```bash
pip install -r requirements.txt
```
