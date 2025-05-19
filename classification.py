import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.utils import resample
from sklearn.base import clone
import itertools
from tqdm import tqdm
import pyomo.environ as pyo
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.nn.functional import softmax
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tempfile

from omlt.io import load_onnx_neural_network_with_bounds, write_onnx_model_with_bounds
import copy

import os
import sys

import cloudpickle as cp

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

    
def train_ensembles(X_train, y_train, models_and_params, P_list, cv=False, name_file='model'):
    
    bootstrap_size = int(0.5 * len(X_train))  # you can customize this
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for P in P_list:
        print(f'\n Train ensemble of {P} predictors')
        print()
        ensemble_results = {}
        
        # Loop through models
        for name, (base_model, param_grid) in models_and_params.items():
            print(f"\nTuning {name} Ensemble...")
            best_score = float("inf")
            best_params = None
            best_ensemble = None
            t_start = time.time()

            # Manual grid search over hyperparameter combinations
            keys, values = zip(*param_grid.items())
            param_list = itertools.product(*values)
            for param_values in tqdm(param_list):
                params = dict(zip(keys, param_values))
                cv_scores = []

                if cv:
                    for train_idx, val_idx in kf.split(X_train):
                        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
                        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

                        ensemble = train_ensemble(base_model, params, X_tr, y_tr, P, bootstrap_size)
                        y_pred = ensemble_predict(ensemble, X_val)
                        cv_scores.append(mean_squared_error(y_val, y_pred))

                    avg_cv_score = np.mean(cv_scores)
                    if avg_cv_score < best_score:
                        best_params = params
                        best_score = avg_cv_score

                else:
                    best_params = params
                    best_score = 9999

            best_ensemble = train_ensemble(base_model, best_params, X_train, y_train, P, bootstrap_size)

            ensemble_results[name] = {
                "best_score": best_score,
                "best_params": best_params,
                "best_ensemble": best_ensemble,
                "val_time": time.time()- t_start
            }
            print(f"Best score for {name}: {best_score:.6f}")
            print(f"    Params: {best_params}")

        print()
        print(ensemble_results)
        print()

        tail = ''
        if cv:
            tail = '_cv'

        to_pickle(ensemble_results, f'{P}_{name_file}{tail}')

def train_model(X_train, y_train, models_and_params, cv=False, name_file='model'):
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    ensemble_results = {}
    
    # Loop through models
    for name, (base_model, param_grid) in models_and_params.items():
        print(f"\nTuning {name} Ensemble...")
        best_score = float("inf")
        best_params = None
        best_ensemble = None
        t_start = time.time()

        # Manual grid search over hyperparameter combinations
        keys, values = zip(*param_grid.items())
        param_list = itertools.product(*values)
        for param_values in tqdm(param_list):
            params = dict(zip(keys, param_values))
            cv_scores = []

            if cv:
                for train_idx, val_idx in kf.split(X_train):
                    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
                    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

                    model = clone(base_model)
                    model.set_params(**params)
                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_val)
                    cv_scores.append(mean_squared_error(y_val, y_pred))

                avg_cv_score = np.mean(cv_scores)
                if avg_cv_score < best_score:
                    best_params = params
                    best_score = avg_cv_score

            else:
                best_params = params
                best_score = float('inf')

        best_ensemble = clone(base_model)
        best_ensemble.set_params(**best_params)
        best_ensemble.fit(X_train, y_train)

        ensemble_results[name] = {
            "best_score": best_score,
            "best_params": best_params,
            "best_ensemble": best_ensemble,
            "val_time": time.time()- t_start
        }
        print(f"Best score for {name}: {best_score:.6f}")
        print(f"    Params: {best_params}")

    print()
    print(ensemble_results)
    print()

    tail = ''
    if cv:
        tail = '_cv'

    to_pickle(ensemble_results, f'{name_file}{tail}')



class NeuralNetwork(nn.Module):
    def __init__(self, hidden_dim=8, num_layers=4):
        super(NeuralNetwork, self).__init__()
        
        layers = []

        # Input layer
        layers.append(nn.Linear(25, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_dim, 4))

        # Wrap in Sequential
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class CustomDataset(Dataset):
    def __init__(self, dataframe, input_columns, target_column):
        # Convert the input DataFrame into torch tensors
        self.X = torch.tensor(dataframe[input_columns].values, dtype=torch.float32)
        self.y = torch.tensor(dataframe[target_column].values, dtype=torch.long)  # Assuming classification

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def train_nn_kfold_and_final(model_class, dataset, num_epochs=500, lr=0.01, k=5, save_path="final_model.pth", weight_decay=1e-5, hidden_dim=8, num_layers=4):
    # KFold split
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []  # Store the results (e.g., accuracy, loss) for each fold
    
    # Loop over each fold
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\nFold {fold+1}/{k} training...")

        # Split the data into train and validation sets
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        # Create DataLoaders for training and validation data
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

        # Initialize model, optimizer, and loss function for each fold
        model = model_class(hidden_dim=hidden_dim, num_layers=num_layers)  # Initialize model with dropout
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # L2 Regularization (weight_decay)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for images, labels in train_loader:
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = running_loss / len(train_loader)
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Evaluate model on validation set
        model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():  # Disable gradient calculation during validation
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate validation accuracy and loss
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")

        fold_results.append((avg_val_loss, accuracy))

    # Calculate average results across all folds
    avg_loss = sum([result[0] for result in fold_results]) / k
    avg_accuracy = sum([result[1] for result in fold_results]) / k
    print(f"\nK-Fold CV Results: Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.2f}%")

    # Final training on the entire dataset
    print("\nTraining the final model on the entire dataset...")
    model = model_class(hidden_dim=hidden_dim, num_layers=num_layers)   
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # L2 Regularization (weight_decay)

    # Using the entire dataset for final training
    train_loader_full = DataLoader(dataset, batch_size=32, shuffle=True)

    # Training loop on full data
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader_full:
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = running_loss / len(train_loader_full)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Save the final trained model
    torch.save(model.state_dict(), save_path)
    print(f"Final model saved to {save_path}")
    
    return model

def evaluate_model(model, loader, class_names=[0,1,2,3]):
    """
    Evaluates a PyTorch model on a given DataLoader.
    Calculates accuracy and plots the confusion matrix.
    
    Args:
        model (torch.nn.Module): The trained model.
        loader (torch.utils.data.DataLoader): DataLoader with (X, y) batches.
        class_names (list, optional): Names of classes for labeling the confusion matrix.
    """
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
    m = pyo.ConcreteModel()
    m.K = pyo.Set(initialize=list(nutr_val.index)) 
    m.L = pyo.Set(initialize=list(nutr_req.columns))
    m.LABELS = pyo.Set(initialize=[0,1,2,3])

    m.mapper = {idx: pos for pos, idx in enumerate(nutr_val.index)}

    m.x = pyo.Var(m.K, within=pyo.NonNegativeReals)  

    @m.Objective(sense=pyo.minimize)
    def obj(m):
        return sum(cost_p[k].item()*m.x[k] for k in m.K)

    @m.Constraint(m.L)
    def meet_req(m, l):
        return sum(m.x[k] * nutr_val.loc[k, l] for k in m.K) >= nutr_req[l].item()

    @m.Constraint()
    def sugar_req(m):
        return m.x['Sugar'] == 0.2

    @m.Constraint()
    def salt_req(m):
        return m.x['Salt'] == 0.05
    
    return m

def pytorch_to_omlt(model, input_dim, input_bounds):
    x = torch.randn(64, input_dim, requires_grad=True)

    pytorch_model = None
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        torch.onnx.export(
            model,
            x,
            f,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        write_onnx_model_with_bounds(f.name, None, input_bounds)
        print(f"Wrote PyTorch model to {f.name}")
        pytorch_model = f.name

    network_definition = load_onnx_neural_network_with_bounds(pytorch_model)

    return network_definition



if __name__ == "__main__":
    # Parameters
    P_list = [5, 10]
    alphas = [0.1, 0.05]
    iterations = 100
    epochs = 500

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
    for P in P_list:
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


    model = NeuralNetwork(hidden_dim=64, num_layers=3)
    model.load_state_dict(torch.load("model_epoch500_weight100_hid64_numlay3.pth"))
    model.eval()
    evaluate_model(model, unseen_loader)

    all_labels = []
    all_logits = []

    with torch.no_grad():  # 
        for images, labels in cal_loader:
        # for images, labels in unseen_loader:

            # Get raw logits from the model
            logits = model(images)

            # Apply softmax to the logits for the entire batch
            softmax_probs = softmax(logits, dim=1)

            # Save softmax probabilities (for analysis)
            all_labels.append(labels)
            all_logits.append(logits)

    all_labels = torch.cat(all_labels, dim=0)
    all_logits = torch.cat(all_logits, dim=0)

    qhats = {}

    s = - all_logits[torch.arange(all_labels.size(0)), all_labels]

    for alpha in alphas:
        qhats[alpha] = np.quantile(s, np.ceil((len(X_cal)+1)*(1 - alpha))/len(X_cal), interpolation='higher')

    max_s = np.max(np.abs(s))

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
            # Load ensemble or conformal predictors depending on the case
            if not P == 'Conformal':
                ensemble = []
                for i in range(P):
                    model = NeuralNetwork(hidden_dim=64, num_layers=3)
                    model.load_state_dict(torch.load(f"ensemble_{i}_in_P_{P}.pth"))
                    model.eval()
                    ensemble.append(model)

                lb = np.min(X_rest, axis=0)
                ub = np.max(X_rest, axis=0)
                input_bounds = list(zip(lb, ub))
                qhat = None
            else:
                f_model = NeuralNetwork(hidden_dim=64, num_layers=3)
                f_model.load_state_dict(torch.load("model_epoch500_weight100_hid64_numlay3.pth"))
                f_model.eval()
                lb = np.min(X_train, axis=0)
                ub = np.max(X_train, axis=0)
                input_bounds = list(zip(lb, ub))
                qhat = qhats[alpha]

            oracle = from_pickle('wfp/oracle')['best_model']
            new_result = {}
            indices = cost_p.index
            res = []
            np.random.seed(0) # Ensure reproducibility of cost vector



    # lb = np.min(X_rest, axis=0)
    # ub = np.max(X_rest, axis=0)
    # input_bounds = list(zip(lb, ub))
    # to_pickle(input_bounds, 'input_bounds_p')

    # lb = np.min(X_train, axis=0)
    # ub = np.max(X_train, axis=0)
    # input_bounds = list(zip(lb, ub))
    # to_pickle(input_bounds, 'input_bounds_conf')








    
