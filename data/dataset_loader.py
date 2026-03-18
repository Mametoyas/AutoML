"""
Dataset loader for AutoML experiments.
Supports: heart (classification), student (classification),
          housing (regression), diabetes (regression).
"""
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, load_diabetes
from sklearn.preprocessing import LabelEncoder


def load_dataset(name, raw_dir='./data/raw/', use_synthetic=False):
    """
    Load dataset by name.

    Args:
        name: 'heart', 'student', 'housing', or 'diabetes'
        raw_dir: directory containing raw CSV files
        use_synthetic: if True, skip Kaggle files and use fallbacks

    Returns:
        tuple: (X, y, task_type, dataset_info)
    """
    loaders = {
        'heart': _load_heart,
        'student': _load_student,
        'housing': _load_housing,
        'diabetes': _load_diabetes
    }

    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(loaders.keys())}")

    return loaders[name](raw_dir, use_synthetic)


def _load_heart(raw_dir, use_synthetic=False):
    """Load Heart Disease dataset (binary classification)."""
    csv_path = os.path.join(raw_dir, 'heart.csv')

    if not use_synthetic and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Try 'target' column
            target_col = None
            for col in ['target', 'Target', 'label', 'Label', 'class', 'Class']:
                if col in df.columns:
                    target_col = col
                    break

            if target_col is None:
                target_col = df.columns[-1]

            y = df[target_col].values.astype(int)
            X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).values.astype(np.float64)

            info = {
                'name': 'Heart Disease',
                'source': 'Kaggle: sintariosatya/heart-disease-dataset',
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'task': 'classification',
                'n_classes': len(np.unique(y))
            }
            return X, y, 'classification', info

        except Exception as e:
            print(f"  Warning: Failed to load heart.csv ({e}), using fallback.")

    # Fallback
    X, y = make_classification(n_samples=303, n_features=13, n_informative=10,
                                n_redundant=2, random_state=42)
    X = X.astype(np.float64)
    info = {
        'name': 'Heart Disease (synthetic)',
        'source': 'sklearn make_classification (fallback)',
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'task': 'classification',
        'n_classes': 2
    }
    return X, y, 'classification', info


def _load_student(raw_dir, use_synthetic=False):
    """Load Student Performance dataset (classification)."""
    csv_path = os.path.join(raw_dir, 'student_performance.csv')

    if not use_synthetic and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)

            # Auto-detect target column
            target_col = None
            target_candidates = ['pass', 'Pass', 'result', 'Result', 'grade', 'Grade',
                                  'performance', 'Performance', 'label', 'Label',
                                  'GradeClass', 'grade_class']
            for col in target_candidates:
                if col in df.columns:
                    target_col = col
                    break

            if target_col is None:
                target_col = df.columns[-1]

            # Process target
            y_raw = df[target_col].copy()

            if y_raw.dtype == object or y_raw.dtype.name == 'category':
                # Categorical target: label encode
                le = LabelEncoder()
                y = le.fit_transform(y_raw.astype(str))
            else:
                y = y_raw.values.astype(float)
                # If more than 5 unique numeric values: binarize at median
                if len(np.unique(y)) > 5:
                    median_val = np.median(y)
                    y = (y >= median_val).astype(int)
                else:
                    y = y.astype(int)

            # Process features
            df_feat = df.drop(columns=[target_col])

            # Label encode remaining categoricals
            for col in df_feat.select_dtypes(include=['object', 'category']).columns:
                le = LabelEncoder()
                df_feat[col] = le.fit_transform(df_feat[col].astype(str))

            df_feat = df_feat.dropna()
            X = df_feat.values.astype(np.float64)
            y = y[:len(X)]

            n_classes = len(np.unique(y))
            info = {
                'name': 'Student Performance',
                'source': 'Kaggle: nabeelqureshitiii/student-performance-dataset',
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'task': 'classification',
                'n_classes': n_classes
            }
            return X, y, 'classification', info

        except Exception as e:
            print(f"  Warning: Failed to load student_performance.csv ({e}), using fallback.")

    # Fallback
    X, y = make_classification(n_samples=500, n_features=15, n_informative=10,
                                n_redundant=3, n_classes=3, n_clusters_per_class=1,
                                random_state=42)
    X = X.astype(np.float64)
    info = {
        'name': 'Student Performance (synthetic)',
        'source': 'sklearn make_classification (fallback)',
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'task': 'classification',
        'n_classes': 3
    }
    return X, y, 'classification', info


def _load_housing(raw_dir, use_synthetic=False):
    """Load Housing Prices dataset (regression)."""
    csv_path = os.path.join(raw_dir, 'housing.csv')

    if not use_synthetic and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)

            # Auto-detect target column
            target_col = None
            target_candidates = ['price', 'Price', 'SalePrice', 'median_house_value',
                                  'PRICE', 'sale_price']
            for col in target_candidates:
                if col in df.columns:
                    target_col = col
                    break

            if target_col is None:
                target_col = df.columns[-1]

            y_raw = df[target_col].copy()
            df_feat = df.drop(columns=[target_col])

            # Map yes/no to 1/0
            for col in df_feat.columns:
                if df_feat[col].dtype == object:
                    unique_vals = df_feat[col].str.lower().unique()
                    if set(unique_vals).issubset({'yes', 'no', 'nan', 'none', ''}):
                        df_feat[col] = df_feat[col].str.lower().map({'yes': 1, 'no': 0}).fillna(0)
                    else:
                        le = LabelEncoder()
                        df_feat[col] = le.fit_transform(df_feat[col].astype(str))

            # Drop NaN
            df_combined = pd.concat([df_feat, y_raw.rename('__target__')], axis=1).dropna()
            y = df_combined['__target__'].values.astype(np.float64)
            X = df_combined.drop(columns=['__target__']).values.astype(np.float64)

            # Log-transform target
            y = np.log1p(y)

            info = {
                'name': 'Housing Prices',
                'source': 'Kaggle: yasserh/housing-prices-dataset',
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'task': 'regression',
                'log_transformed': True
            }
            return X, y, 'regression', info

        except Exception as e:
            print(f"  Warning: Failed to load housing.csv ({e}), using fallback.")

    # Fallback: California Housing
    try:
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        X = data.data.astype(np.float64)
        y = np.log1p(data.target)
        info = {
            'name': 'Housing Prices (California fallback)',
            'source': 'sklearn fetch_california_housing',
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'task': 'regression',
            'log_transformed': True
        }
        return X, y, 'regression', info
    except Exception:
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=500, n_features=8, noise=0.1, random_state=42)
        y = np.log1p(np.abs(y))
        info = {
            'name': 'Housing Prices (synthetic fallback)',
            'source': 'sklearn make_regression',
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'task': 'regression',
            'log_transformed': True
        }
        return X, y.astype(np.float64), 'regression', info


def _load_diabetes(raw_dir, use_synthetic=False):
    """Load Diabetes dataset (regression, sklearn built-in)."""
    data = load_diabetes()
    X = data.data.astype(np.float64)
    y = data.target.astype(np.float64)
    info = {
        'name': 'Diabetes',
        'source': 'sklearn load_diabetes (built-in)',
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'task': 'regression',
        'log_transformed': False
    }
    return X, y, 'regression', info
