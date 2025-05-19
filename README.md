# Conformal Mixed-Integer Constraint Learning (C-MICL)

This repository contains the implementation of the experiments presented in the paper:

**Title:** *Conformal Mixed-Integer Constraint Learning with Feasibility Guarantees*  
**Authors:** Anon.

## Abstract

We propose **Conformal Mixed-Integer Constraint Learning (C-MICL)**, a novel framework that provides probabilistic feasibility guarantees for data-driven constraints in optimization problems. Unlike standard Mixed-Integer Constraint Learning methods that may violate true constraints due to model error or data limitations, C-MICL uses conformal prediction to ensure that feasible solutions are *ground-truth feasible* with probability at least \(1{-}\alpha\), under a conditional independence assumption.

This framework supports both regression and classification tasks without requiring access to the true constraint function and avoids the scalability issues of ensemble-based heuristics. Experiments on real-world applications show that C-MICL consistently meets target feasibility rates, maintains competitive objective performance, and significantly reduces computational cost.

Our approach bridges mathematical optimization and machine learning, offering a principled way to incorporate uncertainty-aware constraints into decision-making with rigorous statistical guarantees.

---

## Repository Structure

```plaintext
.
├── classification.py       # Runs the classification case study (WFP/Syria dataset)
├── regression.py           # Runs the regression case study (chemical reactor dataset)
├── requirements.txt        # Python dependencies
└── data/
    ├── unscaled_noisy_reactor_data.xlsx         # Provided data for regression case study
    ├── WFP_dataset.csv     # Required for classification (must be downloaded)
    └── Syria_instance.xlsx # Required for classification (must be downloaded)


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
