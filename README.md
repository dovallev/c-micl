# Conformal Mixed-Integer Constraint Learning (C-MICL)

This repository contains the implementation of the experiments presented in the paper:

**Title:** *Conformal Mixed-Integer Constraint Learning with Feasibility Guarantees*  
**Authors:** Daniel Ovalle, Lorenz T. Biegler, Ignacio E. Grossmann, Carl D. Laird, and Mateo Dulce Rubio

## Abstract

We propose **Conformal Mixed-Integer Constraint Learning (C-MICL)**, a novel framework that provides probabilistic feasibility guarantees for data-driven constraints in optimization problems. While standard Mixed-Integer Constraint Learning methods often violate the true constraints due to model error or data limitations, our C-MICL approach leverages conformal prediction to ensure feasible solutions are ground-truth feasible with probability at least  $1{-}\alpha$, under a conditional independence assumption. The proposed framework supports both regression and classification tasks without requiring access to the true constraint function, while avoiding the scalability issues associated with ensemble-based heuristics. Experiments on real-world applications demonstrate that C-MICL consistently achieves target feasibility rates, maintains competitive objective performance, and significantly reduces computational cost compared to existing methods. Our work bridges mathematical optimization and machine learning, offering a principled approach to incorporate uncertainty-aware constraints into decision-making with rigorous statistical guarantees.

---

## Repository Structure

```plaintext
.
├── classification.py       # Runs the classification case study (Food basket)
├── classification_oracle       # Contains the neural network used as an oracle for the classification setting
├── regression.py           # Runs the regression case study (chemical reactor dataset)
├── requirements.txt        # Python dependencies
└── data/
    ├── unscaled_noisy_reactor_data.xlsx         # Provided data for regression case study
    ├── WFP_dataset.csv     # Required for classification (must be downloaded)
    └── Syria_instance.xlsx # Required for classification (must be downloaded)
├── notebooks/                 # Tutorials on conformal prediction and Pyomo/OMLT integration
    ├── data/                     # Auxiliary data used by the notebooks
    └── regression/
        ├── 01_conformal_tutorial.ipynb
        └── 02_uncertainty_integration.ipynb
```

## Getting Started

### 1. Install Dependencies

We recommend using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```


### 2. Prepare Datasets

#### Regression Case Study

The file `unscaled_noisy_reactor_data.xlsx` used in the regression case study is already included in the `data/` folder:

```plaintext
data/
└── unscaled_noisy_reactor_data.xlsx
```
### Classification Case Study

To run the classification experiment, you need two additional datasets that are **not included** in this repository.

#### Step 1: Download Required Files

Download the following files from the [OptiCL GitHub repository](https://github.com/hwiberg/OptiCL):

- [`WFP_dataset.csv`](https://github.com/hwiberg/OptiCL)
- [`Syria_instance.xlsx`](https://github.com/hwiberg/OptiCL)

#### Step 2: Place in the Data Folder

After downloading, move the files into the `data/` directory of this repository:

```plaintext
data/
├── WFP_dataset.csv
└── Syria_instance.xlsx
```

### 3. Run Experiments

#### Regression Case Study

```bash
python regression.py
```

#### Classification Case Study

```bash
python classification.py
```

---

## Tutorial Notebooks

The `notebooks/` folder contains detailed, self-contained tutorials that complement the experimental scripts by providing pedagogical and implementation-focused material.

These notebooks cover:

- **Conformal Prediction Fundamentals**
  - Split and inductive conformal prediction
  - Coverage guarantees and calibration
  - Regression setting

- **Integration with Pyomo and OMLT**
  - Constructing uncertainty-aware constraints
  - Translating conformal sets into deterministic optimization constraints
  - Embedding learned models into Pyomo formulations
  - Using OMLT to represent neural network surrogates
  - End-to-end example of conformal constraints in optimization models

---

## Citation

If you find this repository or the associated methodology useful in your work, please consider citing:

```bibtex
@inproceedings{
ovalle2025conformal,
title={Conformal Mixed-Integer Constraint Learning with Feasibility Guarantees},
author={Daniel Ovalle and Lorenz T. Biegler and Ignacio E Grossmann and Carl D Laird and Mateo Dulce Rubio},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=ZvUZvT8tgg}
}


