"""
Search Space Definition for AutoML Pipeline Optimization.
Vector format: [scaler_id, feature_id, feature_param, model_id, hp1, hp2, hp3, hp4]
"""
import numpy as np


LOWER_BOUNDS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
UPPER_BOUNDS = [2.99, 2.99, 1.0, 4.99, 1.0, 1.0, 1.0, 1.0]

SCALER_NAMES = {0: 'None', 1: 'Standard', 2: 'MinMax'}
FEATURE_NAMES = {0: 'None', 1: 'SelectKBest', 2: 'PCA'}
MODEL_NAMES_CLF = {0: 'LogisticRegression', 1: 'SVM', 2: 'RandomForest', 3: 'XGBoost', 4: 'MLP'}
MODEL_NAMES_REG = {0: 'Ridge', 1: 'SVR', 2: 'RandomForest', 3: 'XGBoost', 4: 'MLP'}


def get_bounds():
    """Return (lower_bounds, upper_bounds) as numpy arrays."""
    return np.array(LOWER_BOUNDS), np.array(UPPER_BOUNDS)


def decode_vector(vec, task='classification', n_features=10, n_classes=2):
    """
    Decode a solution vector into a pipeline configuration dict.

    Args:
        vec: array-like of length 8
        task: 'classification' or 'regression'
        n_features: number of features in dataset
        n_classes: number of classes (for multiclass support)

    Returns:
        dict with keys: scaler, feature, feature_params, model, model_params
    """
    vec = np.array(vec)

    scaler_id = int(vec[0])
    feature_id = int(vec[1])
    feature_param = float(np.clip(vec[2], 0.0, 1.0))
    model_id = int(vec[3])
    hp1 = float(np.clip(vec[4], 0.0, 1.0))
    hp2 = float(np.clip(vec[5], 0.0, 1.0))
    hp3 = float(np.clip(vec[6], 0.0, 1.0))
    hp4 = float(np.clip(vec[7], 0.0, 1.0))

    # Scaler
    scaler_map = {0: None, 1: 'standard', 2: 'minmax'}
    scaler = scaler_map[scaler_id]

    # Feature selection
    feature_map = {0: None, 1: 'selectkbest', 2: 'pca'}
    feature = feature_map[feature_id]
    k = max(1, int(feature_param * n_features))
    feature_params = {'k': k, 'n_components': k}

    # Model and hyperparameters
    if task == 'classification':
        model_name_map = MODEL_NAMES_CLF
    else:
        model_name_map = MODEL_NAMES_REG

    model_name = model_name_map[model_id]

    model_params = {}

    if model_id == 0:
        # LR (classification) / Ridge (regression)
        reg_strength = 10 ** (hp1 * 4 - 2)  # [0.01, 100]
        max_iter = int(hp2 * 900) + 100       # [100, 1000]
        if task == 'classification':
            model_params = {
                'C': reg_strength,
                'max_iter': max_iter,
                'solver': 'lbfgs' if n_classes <= 2 else 'lbfgs',
                'multi_class': 'multinomial' if n_classes > 2 else 'auto'
            }
        else:
            model_params = {'alpha': reg_strength}

    elif model_id == 1:
        # SVM / SVR
        C = 10 ** (hp1 * 4 - 2)          # [0.01, 100]
        gamma = 10 ** (hp2 * 4 - 4)      # [0.0001, 1.0]
        model_params = {'C': C, 'gamma': gamma, 'kernel': 'rbf'}
        if task == 'classification' and n_classes > 2:
            model_params['decision_function_shape'] = 'ovr'

    elif model_id == 2:
        # Random Forest
        n_estimators = int(hp1 * 490) + 10    # [10, 500]
        max_depth_val = int(hp2 * 19) + 1     # [1, 20]
        max_depth = None if max_depth_val >= 19 else max_depth_val
        min_samples_split = int(hp3 * 8) + 2  # [2, 10]
        model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split
        }

    elif model_id == 3:
        # XGBoost
        learning_rate = hp1 * 0.29 + 0.01  # [0.01, 0.3]
        n_estimators = int(hp2 * 490) + 10   # [10, 500]
        max_depth = int(hp3 * 9) + 1         # [1, 10]
        subsample = hp4 * 0.5 + 0.5          # [0.5, 1.0]
        model_params = {
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'subsample': subsample
        }

    elif model_id == 4:
        # MLP
        hidden_units = int(hp1 * 224) + 32       # [32, 256]
        lr_init = 10 ** (hp2 * 3 - 4)            # [0.0001, 0.1]
        alpha = 10 ** (hp3 * 4 - 4)              # [0.0001, 1.0]
        model_params = {
            'hidden_layer_sizes': (hidden_units, hidden_units // 2),
            'learning_rate_init': lr_init,
            'alpha': alpha,
            'max_iter': 500
        }

    return {
        'scaler': scaler,
        'scaler_id': scaler_id,
        'feature': feature,
        'feature_id': feature_id,
        'feature_params': feature_params,
        'model': model_name,
        'model_id': model_id,
        'model_params': model_params,
        'n_classes': n_classes
    }


def get_pipeline_description(vec, task='classification', n_features=10, n_classes=2):
    """Get human-readable pipeline description string."""
    decoded = decode_vector(vec, task, n_features, n_classes)

    scaler_str = SCALER_NAMES.get(decoded['scaler_id'], 'None')
    feature_str = FEATURE_NAMES.get(decoded['feature_id'], 'None')
    if task == 'classification':
        model_str = MODEL_NAMES_CLF.get(decoded['model_id'], 'Unknown')
    else:
        model_str = MODEL_NAMES_REG.get(decoded['model_id'], 'Unknown')

    parts = [s for s in [scaler_str, feature_str, model_str] if s != 'None']
    return '→'.join(parts) if parts else 'NoPreproc→' + model_str


def get_search_space_info():
    """Print description of the search space."""
    print("\n" + "="*60)
    print("SEARCH SPACE INFORMATION")
    print("="*60)
    print(f"Vector length: 8")
    print(f"Index 0: scaler_id       [0-2]   None/Standard/MinMax")
    print(f"Index 1: feature_id      [0-2]   None/SelectKBest/PCA")
    print(f"Index 2: feature_param   [0,1]   k or n_components ratio")
    print(f"Index 3: model_id        [0-4]   LR/SVM/RF/XGB/MLP")
    print(f"Index 4: hyperparam_1    [0,1]")
    print(f"Index 5: hyperparam_2    [0,1]")
    print(f"Index 6: hyperparam_3    [0,1]")
    print(f"Index 7: hyperparam_4    [0,1]")
    print(f"\nBounds:")
    print(f"  Lower: {LOWER_BOUNDS}")
    print(f"  Upper: {UPPER_BOUNDS}")
    print(f"\nModels (Classification): LR, SVM, RandomForest, XGBoost, MLP")
    print(f"Models (Regression):     Ridge, SVR, RandomForest, XGBoost, MLP")
    print("="*60 + "\n")
