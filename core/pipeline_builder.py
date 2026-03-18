"""
Pipeline Builder: constructs sklearn Pipeline from decoded configuration dict.
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def build_pipeline(decoded_dict, task='classification'):
    """
    Build sklearn Pipeline from decoded configuration dict.

    Args:
        decoded_dict: dict from decode_vector()
        task: 'classification' or 'regression'

    Returns:
        sklearn.Pipeline
    """
    steps = []

    # --- Scaler ---
    scaler_name = decoded_dict.get('scaler')
    if scaler_name == 'standard':
        steps.append(('scaler', StandardScaler()))
    elif scaler_name == 'minmax':
        steps.append(('scaler', MinMaxScaler()))
    # None: no scaler step added

    # --- Feature Selection ---
    feature_name = decoded_dict.get('feature')
    feature_params = decoded_dict.get('feature_params', {})

    if feature_name == 'selectkbest':
        k = feature_params.get('k', 5)
        score_func = f_classif if task == 'classification' else f_regression
        steps.append(('feature', SelectKBest(score_func=score_func, k=k)))
    elif feature_name == 'pca':
        n_components = feature_params.get('n_components', 5)
        steps.append(('feature', PCA(n_components=n_components, random_state=42)))
    # None: no feature step added

    # --- Model ---
    model_id = decoded_dict.get('model_id', 0)
    model_params = decoded_dict.get('model_params', {})
    n_classes = decoded_dict.get('n_classes', 2)

    if task == 'classification':
        model = _build_classifier(model_id, model_params, n_classes)
    else:
        model = _build_regressor(model_id, model_params)

    steps.append(('model', model))

    return Pipeline(steps)


def _build_classifier(model_id, params, n_classes=2):
    """Build classification model."""
    if model_id == 0:
        # Logistic Regression
        kwargs = {
            'C': params.get('C', 1.0),
            'max_iter': params.get('max_iter', 500),
            'random_state': 42,
            'solver': 'lbfgs'
        }
        if n_classes > 2:
            kwargs['multi_class'] = 'multinomial'
        else:
            kwargs['multi_class'] = 'auto'
        return LogisticRegression(**kwargs)

    elif model_id == 1:
        # SVM
        kwargs = {
            'C': params.get('C', 1.0),
            'gamma': params.get('gamma', 0.1),
            'kernel': 'rbf',
            'random_state': 42
        }
        if n_classes > 2:
            kwargs['decision_function_shape'] = 'ovr'
        return SVC(**kwargs)

    elif model_id == 2:
        # Random Forest
        return RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', None),
            min_samples_split=params.get('min_samples_split', 2),
            random_state=42,
            n_jobs=-1
        )

    elif model_id == 3:
        # XGBoost
        if XGBOOST_AVAILABLE:
            kwargs = {
                'learning_rate': params.get('learning_rate', 0.1),
                'n_estimators': params.get('n_estimators', 100),
                'max_depth': params.get('max_depth', 6),
                'subsample': params.get('subsample', 0.8),
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
            return XGBClassifier(**kwargs)
        else:
            return RandomForestClassifier(random_state=42, n_jobs=-1)

    elif model_id == 4:
        # MLP
        return MLPClassifier(
            hidden_layer_sizes=params.get('hidden_layer_sizes', (128, 64)),
            learning_rate_init=params.get('learning_rate_init', 0.001),
            alpha=params.get('alpha', 0.0001),
            max_iter=params.get('max_iter', 500),
            random_state=42
        )

    return LogisticRegression(random_state=42)


def _build_regressor(model_id, params):
    """Build regression model."""
    if model_id == 0:
        # Ridge
        return Ridge(alpha=params.get('alpha', 1.0))

    elif model_id == 1:
        # SVR
        return SVR(
            C=params.get('C', 1.0),
            gamma=params.get('gamma', 0.1),
            kernel='rbf'
        )

    elif model_id == 2:
        # Random Forest
        return RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', None),
            min_samples_split=params.get('min_samples_split', 2),
            random_state=42,
            n_jobs=-1
        )

    elif model_id == 3:
        # XGBoost
        if XGBOOST_AVAILABLE:
            return XGBRegressor(
                learning_rate=params.get('learning_rate', 0.1),
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 6),
                subsample=params.get('subsample', 0.8),
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        else:
            return RandomForestRegressor(random_state=42, n_jobs=-1)

    elif model_id == 4:
        # MLP
        return MLPRegressor(
            hidden_layer_sizes=params.get('hidden_layer_sizes', (128, 64)),
            learning_rate_init=params.get('learning_rate_init', 0.001),
            alpha=params.get('alpha', 0.0001),
            max_iter=params.get('max_iter', 500),
            random_state=42
        )

    return Ridge()
