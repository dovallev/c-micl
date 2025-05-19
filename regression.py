# Standard library
import csv
import itertools
import os
import sys
import time

# Third-party libraries
import cloudpickle as cp
import jax.numpy as npj
from jax.experimental.ode import odeint
from jax.numpy import pi as pi
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import resample
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from tqdm import tqdm
import pyomo.environ as pyo
from lineartree import LinearTreeRegressor
from omlt import OmltBlock
from omlt.gbt import GBTBigMFormulation, GradientBoostedTreeModel
from omlt.io.keras import keras_reader
from omlt.linear_tree import LinearTreeDefinition, LinearTreeGDPFormulation
from omlt.neuralnet import ReluBigMFormulation
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from scikeras.wrappers import KerasRegressor

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

# Build a dummy NN model to wrap latter
def build_mlp_model(input_dim, hidden_layer_sizes=(64,), learning_rate=0.001, alpha=0.0):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))  #first layer is Input

    for units in hidden_layer_sizes:
        model.add(Dense(units, activation='relu', kernel_regularizer=l2(alpha)))

    model.add(Dense(1))  # output layer
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model
    
# Train and cv an ensemble
def train_ensembles(X_train, y_train, models_and_params, P_list, cv=False, name_file='model'):
    # Set the size of each bootstrap sample (e.g., 50% of the training set)
    bootstrap_size = int(0.5 * len(X_train))
    
    # Define 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Loop over different ensemble sizes (number of predictors)
    for P in P_list:
        print(f'\n Train ensemble of {P} predictors')
        print()
        
        # Dictionary to store results for each model
        ensemble_results = {}

        # Loop through each model and its corresponding hyperparameter grid
        for name, (base_model, param_grid) in models_and_params.items():
            print(f"\nTuning {name} Ensemble...")

            # Initialize best tracking variables
            best_score = float("inf")
            best_params = None
            best_ensemble = None
            t_start = time.time()

            # Manual grid search over all parameter combinations
            keys, values = zip(*param_grid.items())
            param_list = itertools.product(*values)

            for param_values in tqdm(param_list):
                params = dict(zip(keys, param_values))
                cv_scores = []

                # If using cross-validation, evaluate this param setting
                if cv:
                    for train_idx, val_idx in kf.split(X_train):
                        # Split into training and validation sets
                        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
                        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

                        # Train an ensemble of P models with bootstrapping
                        ensemble = train_ensemble(base_model, params, X_tr, y_tr, P, bootstrap_size)

                        # Predict using average of ensemble outputs
                        y_pred = ensemble_predict(ensemble, X_val)

                        # Compute error
                        cv_scores.append(mean_squared_error(y_val, y_pred))

                    # Average score over all folds
                    avg_cv_score = np.mean(cv_scores)

                    # Update best score and parameters if current setting is better
                    if avg_cv_score < best_score:
                        best_params = params
                        best_score = avg_cv_score

                else:
                    # If not using CV, just use this param setting (best is not meaningful)
                    best_params = params
                    best_score = float('inf')

            # Train final ensemble on full training set with best parameters
            best_ensemble = train_ensemble(base_model, best_params, X_train, y_train, P, bootstrap_size)

            # Save results for the current model
            ensemble_results[name] = {
                "best_score": best_score,
                "best_params": best_params,
                "best_ensemble": best_ensemble,
                "val_time": time.time() - t_start
            }

            # Print summary for this model
            print(f"Best score for {name}: {best_score:.6f}")
            print(f"    Params: {best_params}")

        # Print summary of all model results for this value of P
        print()
        print(ensemble_results)
        print()

        # Adjust filename to reflect whether CV was used
        tail = ''
        if cv:
            tail = '_cv'

        # Save the results dictionary using pickle
        to_pickle(ensemble_results, f'{P}_{name_file}{tail}')

def train_model(X_train, y_train, models_and_params, cv=False, name_file='model'):

    # Define 5-fold cross-validation splitter
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Dictionary to store results for each model
    ensemble_results = {}

    # Loop through each model and its associated hyperparameter grid
    for name, (base_model, param_grid) in models_and_params.items():
        print(f"\nTuning {name} Ensemble...")

        # Initialize best values
        best_score = float("inf")
        best_params = None
        best_ensemble = None
        t_start = time.time()

        # Get all combinations of hyperparameters
        keys, values = zip(*param_grid.items())
        param_list = itertools.product(*values)

        # Loop through all combinations of parameters
        for param_values in tqdm(param_list):
            params = dict(zip(keys, param_values))
            cv_scores = []

            # If cross-validation is enabled, evaluate using KFold
            if cv:
                for train_idx, val_idx in kf.split(X_train):
                    # Split data into training and validation
                    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
                    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

                    # Clone and train model
                    model = clone(base_model)
                    model.set_params(**params)
                    model.fit(X_tr, y_tr)

                    # Predict and store validation error
                    y_pred = model.predict(X_val)
                    cv_scores.append(mean_squared_error(y_val, y_pred))

                # Average validation error across folds
                avg_cv_score = np.mean(cv_scores)

                # Update best score and parameters if better
                if avg_cv_score < best_score:
                    best_params = params
                    best_score = avg_cv_score

            else:
                # If no CV, just save the current params (actual training occurs below)
                best_params = params
                best_score = float('inf')  # placeholder; not used

        # Train final model on full data using best parameters
        best_ensemble = clone(base_model)
        best_ensemble.set_params(**best_params)
        best_ensemble.fit(X_train, y_train)

        # Store results for current model
        ensemble_results[name] = {
            "best_score": best_score,
            "best_params": best_params,
            "best_ensemble": best_ensemble,
            "val_time": time.time() - t_start
        }

        # Display best result
        print(f"Best score for {name}: {best_score:.6f}")
        print(f"    Params: {best_params}")

    # Print all results
    print()
    print(ensemble_results)
    print()

    # Choose filename suffix based on whether CV was used
    tail = ''
    if cv:
        tail = '_cv'

    # Save results to pickle file
    to_pickle(ensemble_results, f'{name_file}{tail}')


def dma_mr_model(F, z, dt, v_He, v0, F_He, Ft0):
    # Define the ODE system for a differential membrane reactor model
    
    # Constants
    R = 8.314e6  # Ideal gas constant [cm^3·Pa/mol·K]
    k1 = 0.04  # Rate constant for reaction 1 [mol/(cm^3·h)]
    k1_Inv = 6.40e6  # Reverse rate constant for reaction 1
    k2 = 4.20  # Rate constant for reaction 2 [mol/(cm^3·h)]
    k2_Inv = 56.38  # Reverse rate constant for reaction 2
    MM_B = 78.00  # Molecular weight of benzene [g/mol]
    Q = 3600 * 0.01e-4  # Permeation coefficient scaled for time and area
    selec = 1500  # Selectivity of the membrane
    Pt = 101325.0  # Total pressure on the tube side [Pa]
    Ps = 101325.0  # Total pressure on the shell side [Pa]

    # Tube cross-sectional area
    At = 0.25 * np.pi * (dt ** 2)

    # Replace near-zero flow values with a small number to avoid division by zero
    F = npj.where(F <= 1e-9, 1e-9, F)

    # Total molar flows on tube and shell sides
    Ft = F[0:4].sum()  # Tube side (first 4 components)
    Fs = F[4:].sum() + F_He  # Shell side (next 4 + helium)

    # Adjust volume based on changing flow
    v = v0 * (Ft / Ft0)

    # Concentrations in the tube [mol/cm^3]
    C = F[:4] / v

    # Partial pressures on tube side (normalized by reference pressure)
    P0t, P1t, P2t, P3t = (Pt / 101325) * (F[0:4] / Ft)

    # Partial pressures on shell side
    P0s, P1s, P2s, P3s = (Ps / 101325) * (F[4:] / Fs)

    # Reaction rate for reaction 1, with equilibrium correction
    r0 = 3600 * k1 * C[0] * (1 - ((k1_Inv * C[1] * C[2] ** 2) / (k1 * C[0]**2)))
    r0 = npj.where(C[0] <= 1e-9, 0, r0)  # Avoid computation when reactant is near zero

    # Reaction rate for reaction 2, with equilibrium correction
    r1 = 3600 * k2 * C[1] * (1 - ((k2_Inv * C[3] * C[2] ** 3) / (k2 * C[1]**3)))
    r1 = npj.where(C[1] <= 1e-9, 0, r1)  # Avoid computation when reactant is near zero

    # Catalyst effectiveness and void fraction
    eff = 0.9
    vb = 0.5
    Cat = (1 - vb) * eff  # Effective catalyst fraction

    # Derivative of molar flow along reactor length (dF/dz) for each component
    dFdz = npj.array([
        -Cat * r0 * At - (Q / selec) * ((P0t ** 0.25) - (P0s ** 0.25)) * pi * dt,  # Component A
        0.5 * Cat * r0 * At - Cat * r1 * At - (Q / selec) * ((P1t ** 0.25) - (P1s ** 0.25)) * pi * dt,  # Component B
        Cat * r0 * At + Cat * r1 * At - Q * ((P2t ** 0.25) - (P2s ** 0.25)) * pi * dt,  # Component C
        (1 / 3) * Cat * r1 * At - (Q / selec) * ((P3t ** 0.25) - (P3s ** 0.25)) * pi * dt,  # Component D
        (Q / selec) * ((P0t ** 0.25) - (P0s ** 0.25)) * pi * dt,  # Shell component A
        (Q / selec) * ((P1t ** 0.25) - (P1s ** 0.25)) * pi * dt,  # Shell component B
        Q * ((P2t ** 0.25) - (P2s ** 0.25)) * pi * dt,  # Shell component C
        (Q / selec) * ((P3t ** 0.25) - (P3s ** 0.25)) * pi * dt  # Shell component D
    ])
    return dFdz  # Return the spatial derivative of molar flows


def evaluate_ode_system(v0, v_He, T, dt, L):
    # Function to evaluate the ODE system and compute benzene production
    
    # Constants
    R = 8.314e6  # Ideal gas constant [cm^3·Pa/mol·K]
    MM_B = 78.00  # Molecular weight of benzene [g/mol]
    Pt = 101325.0  # Tube pressure [Pa]
    Ps = 101325.0  # Shell pressure [Pa]

    # Inlet molar flow rates computed via ideal gas law [mol/s]
    Ft0 = Pt * v0 / (R * T)  # Total tube side flow at z=0
    F_He = Ps * v_He / (R * T)  # Helium flow on the shell side

    # Initial condition: all flow is species 0 (reactant), others zero
    y0 = npj.hstack((Ft0, npj.zeros(7)))  # 1 tube component + 7 other species

    # Spatial domain for integration
    z = npj.linspace(0, L, 2000)

    # Integrate ODEs over reactor length
    F = odeint(dma_mr_model, y0, z, dt, v_He, v0, F_He, Ft0, rtol=1e-10, atol=1e-10)

    # Extract final value of component 3 (benzene) and convert to mass flow in mg/s
    F_C6H6 = float((F[-1, 3] * 1000) * MM_B)  # mmol/s * g/mol = mg/s
    return F_C6H6  # Return final benzene mass flow rate


def build_base_model(cost):
    # Create a Pyomo concrete model for optimizing reactor design variables
    m = pyo.ConcreteModel('reactor')  # Initialize model with a name

    # Define the index set of design variables
    m.I = pyo.Set(initialize=['v0', 'v_He', 'T', 'dt', 'L'])  # Variables: feed rate, He rate, temperature, diameter, length

    # Create a mapping from variable name to position in the cost vector
    m.mapper = {idx: pos for pos, idx in enumerate(['v0', 'v_He', 'T', 'dt', 'L'])}

    # Define design variables as non-negative real numbers
    m.x = pyo.Var(m.I, within=pyo.NonNegativeReals)

    # Objective function: minimize total cost, using the cost coefficients
    @m.Objective(sense=pyo.minimize)
    def obj(m):
        return sum(cost[m.mapper[i]] * m.x[i] for i in m.I)

    # Constraint: reactor length cannot exceed 1.5 times the diameter
    @m.Constraint()
    def geometry_ub(m):
        return m.x['L'] <= 1.5 * m.x['dt']

    # Constraint: reactor length must be at least 0.1 times the diameter
    @m.Constraint()
    def geometry_lb(m):
        return m.x['L'] >= 0.1 * m.x['dt']

    # Constraint: upper bound on feed gas rate relative to He flow
    @m.Constraint()
    def gasrate_ub(m):
        return m.x['v0'] <= 3 * m.x['v_He']

    # Constraint: lower bound on feed gas rate relative to He flow
    @m.Constraint()
    def gasrate_lb(m):
        return m.x['v0'] >= 0.75 * m.x['v_He']

    # Constraint: upper bound on residence time (feed rate ≤ 120 * length)
    @m.Constraint()
    def residence_ub(m):
        return m.x['v0'] <= 120 * m.x['L']

    # Constraint: lower bound on residence time (feed rate ≥ 20 * length)
    @m.Constraint()
    def residence_lb(m):
        return m.x['v0'] >= 20 * m.x['L']

    # Constraint: upper bound on flow rate based on temperature
    @m.Constraint()
    def flow_bound_high_t(m):
        return m.x['v0'] <= 1.1 * m.x['T']

    return m  # Return the model object

def update_non_conformal_model(m, ensemble, input_bounds):
    # Add a set P to index over the surrogate ensemble members
    m.P = pyo.Set(initialize=list(range(len(ensemble))))

    # Decision variables y[p]: prediction of ensemble member p
    m.y = pyo.Var(m.P, within=pyo.NonNegativeReals)

    # Binary variable z[p]: indicator if model p satisfies the spec
    m.z = pyo.Var(m.P, within=pyo.Binary)

    # Create a surrogate block for each ensemble member
    m.surrogate = OmltBlock(m.P)

    # Loop over ensemble members to build corresponding surrogate formulations
    for p, model in enumerate(ensemble):

        # Linear Decision Tree surrogate
        if surrogate_type in ['LinearDT']:
            ltmodel = LinearTreeDefinition(model, None, input_bounds)
            formulation = LinearTreeGDPFormulation(ltmodel, transformation="bigm")

        # MLP surrogate model
        elif surrogate_type in ['MLP']:
            keras_model = model.model_
            neural_net = keras_reader.load_keras_sequential(keras_model, None, input_bounds)
            formulation = ReluBigMFormulation(neural_net)

        # Default to tree-based surrogate using ONNX
        else:
            initial_types = [("float_input", FloatTensorType([None, model.n_features_in_]))]
            onnx_model = convert_sklearn(model, initial_types=initial_types)
            network_definition = GradientBoostedTreeModel(onnx_model, None, input_bounds)
            formulation = GBTBigMFormulation(network_definition)

        # Build the surrogate model inside its block
        m.surrogate[p].build_formulation(formulation)

    # Constraint: connect model input variables to surrogate inputs
    @m.Constraint(m.I, m.P)
    def connect_inputs(m, i, p):
        return m.x[i] == m.surrogate[p].inputs[m.mapper[i]]

    # Constraint: connect surrogate outputs to ensemble prediction variables
    @m.Constraint(m.P)
    def connect_outputs(m, p):
        return m.y[p] == m.surrogate[p].outputs[0]

    # Constraint: require at least (1 - alpha) fraction of surrogates to meet spec
    @m.Constraint()
    def z_activation(m):
        return sum(m.z[p] for p in m.P) >= len(m.P) * (1 - m.alpha)

    # Constraint: enforce performance threshold if z[p] is activated
    @m.Constraint(m.P)
    def meet_spec(m, p):
        return - m.y[p] + m.min_req <= (m.min_req + 1e-2) * (1 - m.z[p])

    return m

def update_conformal_model(m, qhat, input_bounds):
    # Define output variables for conformal mean and uncertainty predictions
    m.y_f = pyo.Var(within=pyo.NonNegativeReals)
    m.y_u = pyo.Var(within=pyo.NonNegativeReals)

    # Surrogate model blocks for f_model and u_model
    m.f_surrogate = OmltBlock()
    m.u_surrogate = OmltBlock()

    # Define quantile threshold as a Pyomo parameter
    m.qhat = pyo.Param(initialize=qhat)

    # Build surrogate for f_model (mean prediction)
    if surrogate_type in ['LinearDT']:
        f_ltmodel = LinearTreeDefinition(f_model, None, input_bounds)
        f_formulation = LinearTreeGDPFormulation(f_ltmodel, transformation="bigm")

    elif surrogate_type in ['MLP']:
        f_keras_model = f_model.model_
        f_neural_net = keras_reader.load_keras_sequential(f_keras_model, None, input_bounds)
        f_formulation = ReluBigMFormulation(f_neural_net)

    else:
        f_initial_types = [("float_input", FloatTensorType([None, f_model.n_features_in_]))]
        f_onnx_model = convert_sklearn(f_model, initial_types=f_initial_types)
        f_network_definition = GradientBoostedTreeModel(f_onnx_model, None, input_bounds)
        f_formulation = GBTBigMFormulation(f_network_definition)

    m.f_surrogate.build_formulation(f_formulation)

    # Build surrogate for u_model (uncertainty prediction)
    u_keras_model = u_model.model_
    u_neural_net = keras_reader.load_keras_sequential(u_keras_model, None, input_bounds)
    u_formulation = ReluBigMFormulation(u_neural_net)
    m.u_surrogate.build_formulation(u_formulation)

    # Constraint: connect model input to f_surrogate input
    @m.Constraint(m.I)
    def f_connect_inputs(m, i):
        return m.x[i] == m.f_surrogate.inputs[m.mapper[i]]

    # Constraint: assign f_model output to y_f
    @m.Constraint()
    def f_connect_outputs(m):
        return m.y_f == m.f_surrogate.outputs[0]

    # Constraint: connect model input to u_surrogate input
    @m.Constraint(m.I)
    def u_connect_inputs(m, i):
        return m.x[i] == m.u_surrogate.inputs[m.mapper[i]]

    # Constraint: assign u_model output to y_u
    @m.Constraint()
    def u_connect_outputs(m):
        return m.y_u == m.u_surrogate.outputs[0]

    # Constraint: enforce conformal prediction requirement y_f - qhat * y_u ≥ spec
    @m.Constraint()
    def meet_spec(m):
        return m.y_f - m.qhat * m.y_u >= m.min_req

    return m

if __name__ == "__main__":
    # Parameters
    P_list = [1,5,10,25,50]
    alphas = [0.1, 0.05]
    surrogate_list = ['LinearDT', 'GradientBoosting', 'RandomForest', 'MLP']
    iterations = 100
    epochs = 2000

    # Import data
    df = pd.read_excel('data/unscaled_noisy_reactor_data.xlsx')

    # Minor scaling
    num_independent_vars = 5

    independent_vars  = df.columns[:num_independent_vars]
    dependent_vars = df.columns[num_independent_vars:]

    df[independent_vars] = df[independent_vars] / 100
    df[dependent_vars] = df[dependent_vars] / 10
    df['dt'] = df['dt'] * 100

    # Split data
    X = df[independent_vars]
    y = df[dependent_vars]

    X_unseen, X_rest, y_unseen, y_rest = train_test_split(X, y, test_size=0.5, random_state=42)
    X_train, X_cal, y_train, y_cal = train_test_split(X_rest, y_rest, test_size=0.2, random_state=42)

    # Create models
    keras_model = KerasRegressor(
        model=build_mlp_model,
        input_dim=X_train.shape[1], 
        hidden_layer_sizes=(64,),
        learning_rate=0.001,
        alpha=0.0,
        epochs=50,
        batch_size=32,
        verbose=0
    )

    models_and_params = {
        "LinearDT": (
            LinearTreeRegressor( LinearRegression(), criterion="mse"),
            {
                "max_depth": [5],
                "min_samples_split": [10],
                'max_bins': [40], 
            }
        ),
        "RandomForest": (
            RandomForestRegressor(),
            {
                "n_estimators":  [15],
                "max_depth": [5],
                "min_samples_split": [3], 
                'max_features': [0.6] 
            }
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(),
            {
                "n_estimators": [15], 
                "learning_rate": [0.2],  
                "max_depth": [5], 
                "min_samples_split": [5],  
                'max_features': [0.6] 
            }
        ),
        "MLP": (
            keras_model,
            {
                "hidden_layer_sizes": [(32, 32)], 
                "alpha": [0.01],
                "epochs": [epochs] 
            }
        ),
    }

    # Train MICL
    train_model(X_rest, y_rest, models_and_params, cv=True, name_file='1_ensemble_results')

    # Train W-MICL
    train_ensembles(X_rest, y_rest, models_and_params, P_list=P_list[1:], cv=True, name_file='ensemble_results')

    # Train underlying conformal predictor
    train_model(X_train, y_train, models_and_params, cv=True, name_file='conformal')

    qhats = {}

    # Train uncertainty model
    for surr in surrogate_list:

        # Load trained surrogate model (f_model) from file
        conformal_results = from_pickle('conformal_cv')
        f_model = conformal_results[surr]['best_ensemble']

        # Compute absolute residuals on training set
        y_pred = f_model.predict(X_train)
        error_train = np.abs(y_pred.reshape(-1,1) - y_train)

        # Define model and hyperparameters for uncertainty prediction (u_model)
        models_and_params = {"MLP": (
                keras_model,
                {
                    "hidden_layer_sizes": [(32, 32)], 
                    "alpha": [0.01],
                    "epochs": [epochs] 
                }
            )}

        # Train u_model to predict absolute residuals, using cross-validation
        train_model(X_train, error_train, models_and_params, cv=True, name_file=f'{surr}_uncertainty') 

        # Load trained uncertainty model from file
        u_results = from_pickle(f'{surr}_uncertainty_cv')
        u_model = u_results['MLP']['best_ensemble']

        # Compute normalized residuals on calibration set
        # This is: |f(x) - y| / u(x)
        s = np.abs(f_model.predict(X_cal).reshape(-1,1) - y_cal) / u_model.predict(X_cal)

        # Compute conformal quantiles for each alpha to ensure coverage
        for alpha in alphas:
            qhats[(surr, alpha)] = np.quantile(
                s, 
                np.ceil((len(X_cal)+1)*(1 - alpha))/len(X_cal),  # conservative split conformal correction
                interpolation='higher'
            )


    # Optimization loop
    csv_columns = ['alpha', 'surrogate_type', 'min_req', 'iteration', 'oracle_prediction', 
                'true_feasibility',  'predicted_feasibility',   
                'objective', 'time', 'continuous_vars', 'binary_vars', 'constraints', 'termination', 'gap', 'qhat',
                'y_list',  'x_list', 'z_list']
    
    # Iterate over ensemble sizes and surrogate model types
    P_list.append('Conformal')
    for P in P_list:
        for surrogate_type in surrogate_list:
            # Construct path to output CSV file
            dir_path = os.path.dirname(os.path.abspath(__file__))
            csv_file = os.path.join(dir_path, f"{str(P)}_{surrogate_type}_optimization_results.csv")
            dict_data = []

            # Iterate over confidence levels (alphas)
            for alpha in alphas:
                # Load ensemble or conformal predictors depending on the case
                if not P == 'Conformal':
                    ensemble = from_pickle(f'{P}_ensemble_results_cv')[surrogate_type]['best_ensemble']
                    if type(ensemble) != list:
                        ensemble = [ensemble]
                    lb = np.min(X_rest, axis=0)
                    ub = np.max(X_rest, axis=0)
                    input_bounds = list(zip(lb, ub))
                    qhat = None
                else:
                    f_model = from_pickle(f'conformal_cv')[surrogate_type]['best_ensemble']
                    u_model = from_pickle(f'{surrogate_type}_uncertainty_cv')['MLP']['best_ensemble']
                    lb = np.min(X_train, axis=0)
                    ub = np.max(X_train, axis=0)
                    input_bounds = list(zip(lb, ub))
                    qhat = qhats[(surrogate_type, alpha)]

                new_result = {}

                res = []
                np.random.seed(0)  # Ensure reproducibility of cost vector

                # Perform optimization for specified number of iterations
                for i in range(iterations):
                    # Randomly sample cost vector
                    cost = np.random.uniform(-4, 4, size=5)
                    cost[cost < 0] /= 10  # Shrink negative weights

                    # Build base Pyomo model
                    m =  build_base_model(cost)
                    m.alpha = pyo.Param(initialize=alpha)
                    m.min_req = pyo.Param(initialize=5.0)

                    # Update model based on type of surrogate
                    if not P == 'Conformal':
                        m = update_non_conformal_model(m, ensemble, input_bounds)
                    else:
                        m = update_conformal_model(m, qhat, input_bounds)

                    # Solve the model using Gurobi
                    solver = pyo.SolverFactory('gurobi_direct')
                    solver.options['MIPGap'] = 0.01
                    solver.options['Threads'] = 8
                    results = solver.solve(m, tee=False)

                    # Extract input values as DataFrame
                    x_val = m.x.extract_values()
                    x_df = pd.DataFrame([x_val])

                    # If solution is valid (not NaN), evaluate performance
                    if not x_df.isnull().values.any():
                        y_pred = evaluate_ode_system(m.x['v0']()*100, m.x['v_He']()*100, m.x['T']()*100, m.x['dt'](), m.x['L']()*100)

                        lb = results.problem.lower_bound
                        ub = results.problem.upper_bound
                        gap = abs(ub-lb)/(max(lb,ub) + 1e-10)  # Compute MIP gap

                        x_list = [x_val[k] for k in m.I]

                        # Collect output and feasibility metrics
                        if not P == 'Conformal':
                            y_val = m.y.extract_values()
                            z_val = m.z.extract_values()
                            y_list = list(y_val.values())
                            z_list = list(z_val.values())
                            pred_feas = float(np.mean(z_list))
                        else:
                            y_list = [m.y_f(), m.y_u()]
                            z_list = None
                            pred_feas = m.y_f() - m.qhat * m.y_u()

                        # Store results of successful solve
                        new_result = {
                            'alpha': alpha, 'surrogate_type': surrogate_type, 'min_req': 50, 'iteration': i,
                            'oracle_prediction': float(y_pred), 'true_feasibility': int(float(y_pred) >= 50),
                            'predicted_feasibility': pred_feas, 'objective': round(m.obj(), 8),
                            'time': round(results.solver.wallclock_time, 4),
                            'continuous_vars': results.problem.number_of_continuous_variables,
                            'binary_vars': results.problem.number_of_binary_variables,
                            'constraints': results.problem.number_of_constraints,
                            'termination': str(results.solver.termination_condition), 'gap': round(100 * gap, 4),
                            'qhat': qhat, 'y_list': y_list, 'x_list': x_list, 'z_list': z_list
                        }

                    else:
                        # Handle infeasible or invalid solutions
                        new_result = {
                            'alpha': alpha, 'surrogate_type': surrogate_type, 'min_req': 50, 'iteration': i,
                            'oracle_prediction': None, 'true_feasibility': None, 'predicted_feasibility': None,
                            'objective': None, 'time': round(results.solver.wallclock_time, 4),
                            'continuous_vars': results.problem.number_of_continuous_variables,
                            'binary_vars': results.problem.number_of_binary_variables,
                            'constraints': results.problem.number_of_constraints,
                            'termination': str(results.solver.termination_condition), 'gap': None,
                            'qhat': qhat, 'y_list': None, 'x_list': None, 'z_list': None
                        }

                    # Append current iteration result
                    dict_data.append(new_result)

                    # Write all results up to now into CSV file
                    try:
                        with open(csv_file, 'w') as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                            writer.writeheader()
                            for data in dict_data:
                                writer.writerow(data)
                    except IOError:
                        print("I/O error")