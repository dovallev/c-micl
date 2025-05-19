# Standard library
import copy
import csv
import itertools
import os
import sys
import tempfile
import time

# Third-party libraries
import cloudpickle as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_squared_error,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import resample
from tqdm import tqdm
import pyomo.environ as pyo

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, Dataset

# OMLT
from omlt import OmltBlock
from omlt.io import load_onnx_neural_network_with_bounds, write_onnx_model_with_bounds
from omlt.neuralnet import ReluBigMFormulation

# Project-specific (local) paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def to_pickle(object, path):
    # Get the absolute path of the current directory
    current_path = os.path.dirname(os.path.realpath(__file__))

    # Combine the current directory path with the specified path
    object_pickle_path = os.path.join(current_path, path)

    # Write the object to the pickle file
    with open(object_pickle_path, "wb") as f:
        cp.dump(object, f)

    return object

def from_pickle(path):
    # Get the absolute path of the current directory
    current_path = os.path.dirname(os.path.realpath(__file__))

    # Combine the current directory path with the specified path
    object_pickle_path = os.path.join(current_path, path)

    # Load the object from the pickle file
    with open(object_pickle_path, "rb") as f:
        object = cp.load(f)

    return object



# Helper: bootstrap samples
def generate_bootstrap_samples(X, y, size, P):
    return [resample(X, y, replace=True, n_samples=size, random_state=i) for i in range(P)]

# Helper: train ensemble
def train_ensemble(model_class, hyperparams, X, y, P, size):
    ensemble = []
    boot_samples = generate_bootstrap_samples(X, y, size, P)
    for X_b, y_b in boot_samples:
        model = clone(model_class)
        model.set_params(**hyperparams)
        model.fit(X_b, y_b)
        ensemble.append(model)
    return ensemble

# Helper: predict with ensemble
def ensemble_predict(ensemble, X):
    preds = np.array([m.predict(X) for m in ensemble])
    return np.mean(preds, axis=0)
   
# Define a fully-connected neural network with configurable hidden dimension and number of layers
class NeuralNetwork(nn.Module):
    def __init__(self, hidden_dim=8, num_layers=4):
        super(NeuralNetwork, self).__init__()

        layers = []

        # Input layer: 25 input features → hidden_dim output units
        layers.append(nn.Linear(25, hidden_dim))
        layers.append(nn.ReLU())

        # Add (num_layers - 1) hidden layers, each followed by ReLU activation
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer: hidden_dim → 4 output units (e.g., 4 class logits or targets)
        layers.append(nn.Linear(hidden_dim, 4))

        # Chain all layers into a single sequential model
        self.network = nn.Sequential(*layers)

    # Forward pass definition
    def forward(self, x):
        return self.network(x)


# Custom PyTorch Dataset to wrap a pandas DataFrame for training
class CustomDataset(Dataset):
    def __init__(self, dataframe, input_columns, target_column):
        # Extract input features and convert to float32 torch tensor
        self.X = torch.tensor(dataframe[input_columns].values, dtype=torch.float32)

        # Extract target column and convert to long tensor (for classification tasks)
        self.y = torch.tensor(dataframe[target_column].values, dtype=torch.long)

    # Return the total number of samples
    def __len__(self):
        return len(self.X)

    # Return a single sample (input, target) pair
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def train_nn_kfold_and_final(model_class, dataset, num_epochs=500, lr=0.01, k=5, 
                              save_path="final_model.pth", weight_decay=1e-5, 
                              hidden_dim=8, num_layers=4):
    # Initialize k-fold cross-validation
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []  # To store loss and accuracy for each fold
    
    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\nFold {fold+1}/{k} training...")

        # Split dataset into training and validation subsets
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        # Create DataLoaders
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

        # Initialize a fresh model, optimizer, and loss function
        model = model_class(hidden_dim=hidden_dim, num_layers=num_layers)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        # Training loop over epochs
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for data_x, labels in train_loader:
                # Forward pass
                outputs = model(data_x)
                loss = criterion(outputs, labels)

                # Backpropagation and optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Print loss every 10 epochs
            if (epoch + 1) % 10 == 0:
                avg_loss = running_loss / len(train_loader)
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Evaluation on validation set
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        # No gradient needed for validation
        with torch.no_grad():
            for data_x, labels in val_loader:
                outputs = model(data_x)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Get predictions and compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Compute average validation loss and accuracy
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")

        fold_results.append((avg_val_loss, accuracy))

    # Aggregate cross-validation metrics
    avg_loss = sum([result[0] for result in fold_results]) / k
    avg_accuracy = sum([result[1] for result in fold_results]) / k
    print(f"\nK-Fold CV Results: Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.2f}%")

    # Train final model on the entire dataset
    print("\nTraining the final model on the entire dataset...")
    model = model_class(hidden_dim=hidden_dim, num_layers=num_layers)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # DataLoader for the full dataset
    train_loader_full = DataLoader(dataset, batch_size=32, shuffle=True)

    # Training loop on full data
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader_full:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print full training loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_loss = running_loss / len(train_loader_full)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Save the trained model's parameters
    torch.save(model.state_dict(), save_path)
    print(f"Final model saved to {save_path}")
    
    return model

def evaluate_model(model, loader, class_names=[0,1,2,3]):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient calculation
        for X_batch, y_batch in loader:
            outputs = model(X_batch)
            _, preds = torch.max(outputs, axis=1)
            all_preds.append(preds)
            all_labels.append(y_batch)

    # Concatenate all batches
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()

    # Calculate accuracy
    accuracy = np.mean(all_preds == all_labels)
    print(f"Accuracy: {accuracy*100:.2f}%")

    # Compute and plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

def build_base_model(nutr_val, nutr_req, cost_p):
    # Create a concrete Pyomo model instance
    m = pyo.ConcreteModel()
    
    # Define sets:
    # - m.K: food items (indexed by rows of nutr_val)
    # - m.L: nutrient types (indexed by columns of nutr_req)
    # - m.LABELS: arbitrary label set (e.g., for feasibility classes)
    # - m.VALID / m.INVALID: label subsets
    m.K = pyo.Set(initialize=list(nutr_val.index)) 
    m.L = pyo.Set(initialize=list(nutr_req.columns))
    m.LABELS = pyo.Set(initialize=[0, 1, 2, 3])
    m.VALID = pyo.Set(initialize=[2, 3])
    m.INVALID = pyo.Set(initialize=[0, 1])

    # Create an index-to-position mapper 
    m.mapper = {idx: pos for pos, idx in enumerate(nutr_val.index)}

    # Define decision variables:
    # x[k]: amount of food item k to include in the solution (non-negative)
    m.x = pyo.Var(m.K, within=pyo.NonNegativeReals)

    # Objective: minimize total cost of selected food items
    @m.Objective(sense=pyo.minimize)
    def obj(m):
        return sum(cost_p[k].item() * m.x[k] for k in m.K)

    # Constraints: ensure that each nutrient requirement is met
    @m.Constraint(m.L)
    def meet_req(m, l):
        return sum(m.x[k] * nutr_val.loc[k, l] for k in m.K) >= nutr_req[l].item()

    # Hard constraint: fix the quantity of sugar used to 0.2
    @m.Constraint()
    def sugar_req(m):
        return m.x['Sugar'] == 0.2

    # Hard constraint: fix the quantity of salt used to 0.05
    @m.Constraint()
    def salt_req(m):
        return m.x['Salt'] == 0.05

    return m

def pytorch_to_omlt(model, input_dim, input_bounds):
    # Generate dummy input with the same dimension as the model expects
    x = torch.randn(64, input_dim, requires_grad=True)

    # Container for the ONNX file path
    pytorch_model = None

    # Export the PyTorch model to ONNX and add input bounds
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        torch.onnx.export(
            model,                              # PyTorch model to export
            x,                                  # Example input tensor
            f,                                  # File object for ONNX output
            input_names=["input"],              # Name of input tensor
            output_names=["output"],            # Name of output tensor
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # Allow variable batch sizes
        )

        # Add input bounds to the ONNX model for OMLT compatibility
        write_onnx_model_with_bounds(f.name, None, input_bounds)

        print(f"Wrote PyTorch model to {f.name}")
        pytorch_model = f.name

    # Load the ONNX model with bounds into an OMLT-compatible structure
    network_definition = load_onnx_neural_network_with_bounds(pytorch_model)

    return network_definition

def update_non_conformal_model(m, ensemble, input_bounds):
    # Add a set of predictors (one per ensemble model)
    m.P = pyo.Set(initialize=list(range(len(ensemble))))
    
    # y[j, p]: output of neural net predictor p for label j (continuous)
    m.y = pyo.Var(m.LABELS, m.P)

    # z[p]: binary variable indicating whether predictor p is "activated" or contributes
    m.z = pyo.Var(m.P, within=pyo.Binary)

    # w[v, p]: binary indicating if class v ∈ VALID "wins" over others in predictor p
    m.w = pyo.Var(m.VALID, m.P, within=pyo.Binary)

    # Create a surrogate block per ensemble member using OMLT
    m.surrogate = OmltBlock(m.P)

    for p, model in enumerate(ensemble):
        # Convert PyTorch model to OMLT compatible object
        neural_net = pytorch_to_omlt(model, 25, input_bounds)
        
        # Build ReLU Big-M formulation for the network
        formulation = ReluBigMFormulation(neural_net)
        m.surrogate[p].build_formulation(formulation)

    # Input constraint: map model x[k] decision variable into NN input
    @m.Constraint(m.K, m.P)
    def connect_inputs(m, k, p):
        return m.x[k] == m.surrogate[p].inputs[m.mapper[k]]

    # Output constraint: map surrogate output to y[j, p] variable
    @m.Constraint(m.LABELS, m.P)
    def connect_outputs(m, j, p):
        return m.y[j, p] == m.surrogate[p].outputs[j]

    # Big-M constraint: forces w[v, p] to 0 if invalid label u has higher score than valid v
    @m.Constraint(m.VALID, m.INVALID, m.P)
    def bigm(m, v, u, p):
        return m.y[u, p] + 1e-6 - m.y[v, p] <= m.M * (1 - m.w[v, p])

    # Link w[v, p] to z[p]: if any valid class wins, then z[p] can be 1
    @m.Constraint(m.P)
    def link_z(m, p):
        return m.z[p] <= sum(m.w[v, p] for v in m.VALID)

    # Aggregate conformity constraint: a fraction of predictors must agree with VALID classes
    @m.Constraint()
    def z_activation(m):
        return sum(m.z[p] for p in m.P) >= len(m.P) * (1 - m.alpha)

    # At least one VALID label must win in each predictor 
    @m.Constraint(m.P)
    def palatable_diet(m, p):
        return sum(m.w[v, p] for v in m.VALID) >= 1

    return m

def update_conformal_model(m, qhat, f_model, input_bounds):
    # y_f[j]: output from surrogate for label j
    m.y_f = pyo.Var(m.LABELS)

    # Block to hold the surrogate model formulation
    m.f_surrogate = OmltBlock()

    # q̂: conformal threshold
    m.qhat = pyo.Param(initialize=qhat)

    # z[j]: binary variable indicating whether label j is included in the prediction set
    m.z = pyo.Var(m.LABELS, within=pyo.Binary)

    # Convert PyTorch model to OMLT-compatible network
    f_network_definition = pytorch_to_omlt(f_model, 25, input_bounds)

    # Use Big-M formulation for ReLU neural network
    f_formulation = ReluBigMFormulation(f_network_definition)

    # Embed the surrogate model into the Pyomo block
    m.f_surrogate.build_formulation(f_formulation)

    # Connect decision variable inputs (m.x[k]) to surrogate input layer
    @m.Constraint(m.K)
    def f_connect_inputs(m, k):
        return m.x[k] == m.f_surrogate.inputs[m.mapper[k]]

    # Connect surrogate outputs to prediction variables y_f[j]
    @m.Constraint(m.LABELS)
    def f_connect_outputs(m, j):
        return m.y_f[j] == m.f_surrogate.outputs[j]

    # First Big-M constraint
    @m.Constraint(m.LABELS)
    def big_m(m, j):
        return -m.y_f[j] - m.qhat <= m.M * (1 - m.z[j])

    # Second Big-M
    @m.Constraint(m.LABELS)
    def big_m2(m, j):
        return m.y_f[j] + m.qhat + 1e-6 <= m.M * m.z[j]

    # Enforce at least one valid label is selected (e.g., diet is palatable)
    @m.Constraint()
    def palatable_diet(m):
        return sum(m.z[i] for i in m.VALID) >= 1

    # Prevent selecting invalid labels (e.g., those labeled undesirable)
    @m.Constraint(m.INVALID)
    def no_label(m, i):
        return m.z[i] == 0

    return m

if __name__ == "__main__":
    # Parameters
    P_list = [1, 5, 10]
    alphas = [0.1, 0.05]
    iterations = 100
    epochs = 500
    M_factor = 10

    # Import data
    dataset = pd.read_csv('data/WFP_dataset.csv').sample(frac=1)
    nutr_val = pd.read_excel('data/Syria_instance.xlsx', sheet_name='nutr_val', index_col='Food', engine='openpyxl')
    nutr_req = pd.read_excel('data/Syria_instance.xlsx', sheet_name='nutr_req', index_col='Type', engine='openpyxl')
    cost_p = pd.read_excel('data/Syria_instance.xlsx', sheet_name='FoodCost', index_col='Supplier', engine='openpyxl').iloc[0,:]

    # Define bins and labels
    bins = [0, 0.25, 0.5, 0.75, 1]
    labels = [0,1,2,3]

    # Create categorical column
    dataset['class'] = pd.cut(dataset['label'], bins=bins, labels=labels, right=False)

    y = dataset['class']
    X = dataset.drop(['label', 'class'], axis=1, inplace=False)

    # Data split and loader creation
    X_unseen, X_rest, y_unseen, y_rest = train_test_split(X, y, test_size=0.5, random_state=42)
    X_train, X_cal, y_train, y_cal = train_test_split(X_rest, y_rest, test_size=0.08, random_state=42)

    df_train = copy.deepcopy(X_train)
    df_train['target'] = y_train
    train_dataset = CustomDataset(df_train, input_columns=X_train.columns, target_column='target')   
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    df_calibration = copy.deepcopy(X_cal)
    df_calibration['target'] = y_cal
    cal_dataset = CustomDataset(df_calibration, input_columns=X_train.columns, target_column='target')   
    cal_loader = DataLoader(cal_dataset, batch_size=64, shuffle=False)

    df_unseen = copy.deepcopy(X_unseen)
    df_unseen['target'] = y_unseen
    unseen_dataset = CustomDataset(df_unseen, input_columns=X_train.columns, target_column='target')   
    unseen_loader = DataLoader(unseen_dataset, batch_size=64, shuffle=False)

    df_rest = copy.deepcopy(X_rest)
    df_rest['target'] = y_rest
    rest_dataset = CustomDataset(df_rest, input_columns=X_train.columns, target_column='target')   
    rest_loader = DataLoader(rest_dataset, batch_size=64, shuffle=False)

    # Train MICL
    train_nn_kfold_and_final(NeuralNetwork, 
                                    rest_dataset, 
                                    num_epochs=epochs, 
                                    lr=0.001, 
                                    k=5, 
                                    save_path=f"ensemble_{0}_in_P_{1}.pth", 
                                    weight_decay=0.01, 
                                    hidden_dim=64, 
                                    num_layers=3)
    # Train W-MICL
    for P in P_list[1:]:
        for i in range(P):

            df_bootstrap = copy.deepcopy(X_rest)
            df_bootstrap['target'] = y_rest

            bootstrap_sample = resample(df_bootstrap, 
                            replace=True, 
                            n_samples=len(X_rest)//2, 
                            random_state=P*i)
                     
            bootstrap_dataset = CustomDataset(bootstrap_sample, input_columns=X_train.columns, target_column='target')   

            train_nn_kfold_and_final(NeuralNetwork, 
                                    bootstrap_dataset, 
                                    num_epochs=epochs, 
                                    lr=0.001, 
                                    k=5, 
                                    save_path=f"ensemble_{i}_in_P_{P}.pth", 
                                    weight_decay=0.01, 
                                    hidden_dim=64, 
                                    num_layers=3)

    # Train C-MICL
    train_nn_kfold_and_final(NeuralNetwork, 
                                    train_dataset, 
                                    num_epochs=epochs, 
                                    lr=0.001, 
                                    k=5, 
                                    save_path=f"model_epoch{epochs}_weight{int(10000*0.01)}_hid{64}_numlay{3}.pth", 
                                    weight_decay=0.01, 
                                    hidden_dim=64, 
                                    num_layers=3)


    # Initialize and load the trained model
    model = NeuralNetwork(hidden_dim=64, num_layers=3)
    model.load_state_dict(torch.load(f"model_epoch{epochs}_weight100_hid64_numlay3.pth"))
    model.eval()

    # Evaluate performance on a holdout set (optional diagnostic)
    evaluate_model(model, unseen_loader)

    # Store all labels and logits for conformal calibration
    all_labels = []
    all_logits = []

    # Disable gradients during calibration
    with torch.no_grad():
        for data, labels in cal_loader:
            logits = model(data)  # Raw logits (pre-softmax)
            all_labels.append(labels)
            all_logits.append(logits)

    # Concatenate results into tensors of shape [N, num_classes] and [N]
    all_labels = torch.cat(all_labels, dim=0)
    all_logits = torch.cat(all_logits, dim=0)

    # Compute nonconformity scores: negative logit of the true class
    # s_i = -logit_{true_label}(x_i)
    s = -all_logits[torch.arange(all_labels.size(0)), all_labels]

    # Dictionary to store q̂_α for different α levels
    qhats = {}

    # Conformal quantile computation using empirical quantiles
    for alpha in alphas:
        quantile_level = np.ceil((len(X_cal) + 1) * (1 - alpha)) / len(X_cal)
        qhats[alpha] = np.quantile(s, quantile_level, interpolation="higher")

    # Store maximum absolute logit value (used in MILP big-M constants)
    max_s = float(torch.max(torch.abs(all_logits)))

    csv_columns = ['alpha', 'surrogate_type', 'min_req', 'iteration', 'oracle_prediction', 
                'true_feasibility',  'predicted_feasibility',   
                'objective', 'time', 'continuous_vars', 'binary_vars', 'constraints', 'termination', 'gap', 'qhat',
                'y_list',  'x_list', 'z_list']
    
    P_list.append('Conformal')
    for P in P_list:
         # Construct path to output CSV file
        dir_path = os.path.dirname(os.path.abspath(__file__))
        csv_file = os.path.join(dir_path, f"{str(P)}_classification_optimization_results.csv")
        dict_data = []

        # Iterate over confidence levels (alphas)
        for alpha in alphas:
            # === Load surrogate model(s) and input bounds ===
            if P != 'Conformal':
                # Load ensemble models
                ensemble = []
                for i in range(P):
                    model = NeuralNetwork(hidden_dim=64, num_layers=3)
                    model.load_state_dict(torch.load(f"ensemble_{i}_in_P_{P}.pth"))
                    model.eval()
                    ensemble.append(model)

                # Use bounds from rest set
                lb = np.min(X_rest, axis=0)
                ub = np.max(X_rest, axis=0)
                input_bounds = list(zip(lb, ub))
                qhat = None

            else:
                # Load single conformal model
                f_model = NeuralNetwork(hidden_dim=64, num_layers=3)
                f_model.load_state_dict(torch.load(f"model_epoch{epochs}_weight100_hid64_numlay3.pth"))
                f_model.eval()

                # Use bounds from training set
                lb = np.min(X_train, axis=0)
                ub = np.max(X_train, axis=0)
                input_bounds = list(zip(lb, ub))
                qhat = qhats[alpha]

            # === Load feasibility oracle ===
            oracle = from_pickle('classification_oracle')['best_model']

            # === Optimization loop for multiple cost vectors ===
            indices = cost_p.index
            np.random.seed(0)  # Ensures reproducibility

            for i in range(iterations):
                # Generate a random cost vector
                values = np.random.uniform(0.1, 4, size=len(indices))
                cost_p_ran = pd.Series(values, index=indices).round(2)

                # Build base diet model
                m = build_base_model(nutr_val, nutr_req, cost_p_ran)
                m.alpha = pyo.Param(initialize=alpha)
                m.min_req = pyo.Param(initialize=0.5)
                m.M = pyo.Param(initialize=M_factor * max_s)

                # Attach surrogate logic
                if P != 'Conformal':
                    m = update_non_conformal_model(m, ensemble, input_bounds)
                else:
                    m = update_conformal_model(m, qhat, f_model, input_bounds)

                # === Solve MILP model ===
                solver = pyo.SolverFactory('gurobi_direct')
                solver.options['MIPGap'] = 0.01
                solver.options['Threads'] = 8
                results = solver.solve(m, tee=False)

                # === Extract solution and evaluate ===
                x_val = m.x.extract_values()
                x_df = pd.DataFrame([x_val])

                if not x_df.isnull().values.any():
                    # Bounds and gap
                    lb = results.problem.lower_bound
                    ub = results.problem.upper_bound
                    gap = abs(ub - lb) / (max(lb, ub) + 1e-10)

                    # Oracle prediction
                    y_pred = oracle.predict(x_df)
                    x_list = [x_val[k] for k in nutr_val.index]

                    # Extract surrogate outputs
                    if P != 'Conformal':
                        y_val = m.y.extract_values()
                        y_list = [[m.y[j, p]() for j in m.LABELS] for p in m.P]
                        z_val = m.z.extract_values()
                        z_list = list(z_val.values())
                        pred_feas = float(np.mean(z_list))
                    else:
                        y_val = m.y_f.extract_values()
                        z_val = m.z.extract_values()
                        y_list = list(y_val.values())
                        z_list = list(z_val.values())
                        pred_feas = z_list

                    new_result = {
                        'alpha': alpha, 'surrogate_type': None, 'min_req': None, 'iteration': i,
                        'oracle_prediction': float(y_pred[0]),
                        'true_feasibility': int(float(y_pred[0]) >= 0.5),
                        'predicted_feasibility': pred_feas,
                        'objective': round(m.obj(), 8),
                        'time': round(results.solver.wallclock_time, 4),
                        'continuous_vars': results.problem.number_of_continuous_variables,
                        'binary_vars': results.problem.number_of_binary_variables,
                        'constraints': results.problem.number_of_constraints,
                        'termination': str(results.solver.termination_condition),
                        'gap': round(100 * gap, 4),
                        'qhat': qhat,
                        'y_list': y_list,
                        'x_list': x_list,
                        'z_list': z_list
                    }

                else:
                    new_result = {
                        'alpha': alpha, 'surrogate_type': None, 'min_req': None, 'iteration': i,
                        'oracle_prediction': None,
                        'true_feasibility': None,
                        'predicted_feasibility': None,
                        'objective': None,
                        'time': round(results.solver.wallclock_time, 4),
                        'continuous_vars': results.problem.number_of_continuous_variables,
                        'binary_vars': results.problem.number_of_binary_variables,
                        'constraints': results.problem.number_of_constraints,
                        'termination': str(results.solver.termination_condition),
                        'gap': None,
                        'qhat': qhat,
                        'y_list': None,
                        'x_list': None,
                        'z_list': None
                    }

                # Save result and write to CSV
                dict_data.append(new_result)

                try:
                    with open(csv_file, 'w') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                        writer.writeheader()
                        for data in dict_data:
                            writer.writerow(data)
                except IOError:
                    print("I/O error")


