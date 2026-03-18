"""
Fitness Evaluator for AutoML Pipeline Optimization.
Supports classification (accuracy, f1_weighted) and regression (RMSE, MAE, R²).
"""
import warnings
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.exceptions import ConvergenceWarning

from .search_space import decode_vector
from .pipeline_builder import build_pipeline


class FitnessEvaluator:
    """
    Evaluates pipeline configurations using cross-validation.
    Caches results for efficiency.
    """

    def __init__(self, X, y, task='classification', cv=5, metric='auto', n_classes=2):
        """
        Initialize fitness evaluator.

        Args:
            X: feature matrix (np.ndarray)
            y: target vector (np.ndarray)
            task: 'classification' or 'regression'
            cv: number of folds
            metric: 'accuracy', 'f1_weighted', 'rmse', 'mae', 'r2', or 'auto'
            n_classes: number of unique classes (for classification)
        """
        self.X = X
        self.y = y
        self.task = task
        self.cv = cv
        self.metric = metric
        self.n_features = X.shape[1]
        self.n_classes = n_classes

        # CV strategy
        if task == 'classification':
            self.cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            self.cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=42)

        # Cache
        self._cache = {}
        self.eval_count = 0
        self.cache_hits = 0
        self.errors = []

        # Last computed metrics (for regression)
        self.last_metrics = {}

        # Suppress convergence warnings
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

    def _cache_key(self, vector):
        return tuple(round(float(v), 4) for v in vector)

    def evaluate(self, vector):
        """
        Evaluate pipeline configuration, return primary fitness (higher = better).

        Args:
            vector: solution vector of length 8

        Returns:
            float: fitness value in [0, 1]
        """
        key = self._cache_key(vector)
        if key in self._cache:
            self.cache_hits += 1
            self.last_metrics = self._cache[key].get('metrics', {})
            return self._cache[key]['fitness']

        self.eval_count += 1

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                decoded = decode_vector(vector, self.task, self.n_features, self.n_classes)
                pipeline = build_pipeline(decoded, self.task)

                if self.task == 'classification':
                    fitness = self._eval_classification(pipeline)
                    self._cache[key] = {'fitness': fitness, 'metrics': {}}
                else:
                    fitness, metrics = self._eval_regression(pipeline)
                    self._cache[key] = {'fitness': fitness, 'metrics': metrics}
                    self.last_metrics = metrics

                return fitness

        except Exception as e:
            self.errors.append(str(e))
            self._cache[key] = {'fitness': 0.0, 'metrics': {}}
            return 0.0

    def _eval_classification(self, pipeline):
        """Evaluate classification pipeline."""
        if self.metric == 'f1_weighted':
            scores = cross_val_score(pipeline, self.X, self.y,
                                     cv=self.cv_strategy,
                                     scoring='f1_weighted',
                                     error_score=0.0)
        else:  # accuracy or auto
            scores = cross_val_score(pipeline, self.X, self.y,
                                     cv=self.cv_strategy,
                                     scoring='accuracy',
                                     error_score=0.0)
        return float(np.mean(scores))

    def _eval_regression(self, pipeline):
        """Evaluate regression pipeline, return (fitness, metrics_dict)."""
        # RMSE
        mse_scores = cross_val_score(pipeline, self.X, self.y,
                                     cv=self.cv_strategy,
                                     scoring='neg_mean_squared_error',
                                     error_score=0.0)
        rmse_cv = float(np.sqrt(np.mean(-mse_scores)))

        # MAE
        mae_scores = cross_val_score(pipeline, self.X, self.y,
                                     cv=self.cv_strategy,
                                     scoring='neg_mean_absolute_error',
                                     error_score=0.0)
        mae_cv = float(np.mean(-mae_scores))

        # R²
        r2_scores = cross_val_score(pipeline, self.X, self.y,
                                    cv=self.cv_strategy,
                                    scoring='r2',
                                    error_score=0.0)
        r2_cv = float(np.mean(r2_scores))

        metrics = {'rmse': rmse_cv, 'mae': mae_cv, 'r2': r2_cv}

        # Primary fitness
        if self.metric == 'mae':
            fitness = 1.0 / (1.0 + mae_cv)
        elif self.metric == 'r2':
            fitness = max(0.0, r2_cv)
        else:  # rmse or auto
            fitness = 1.0 / (1.0 + rmse_cv)

        return fitness, metrics

    def evaluate_full(self, vector):
        """
        Evaluate and return full metrics dict.

        Returns:
            dict with all metrics
        """
        fitness = self.evaluate(vector)
        key = self._cache_key(vector)
        cached = self._cache.get(key, {})
        metrics = cached.get('metrics', {})

        result = {'fitness': fitness}
        result.update(metrics)
        return result

    def get_stats(self):
        """Return evaluation statistics."""
        return {
            'eval_count': self.eval_count,
            'cache_hits': self.cache_hits,
            'cache_size': len(self._cache),
            'error_count': len(self.errors)
        }
