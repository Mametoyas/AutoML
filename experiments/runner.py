"""
Experiment Runner: orchestrates AutoML experiments across datasets and optimizers.
"""
import os
import sys
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                              mean_squared_error, mean_absolute_error, r2_score)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.search_space import get_bounds, decode_vector, get_pipeline_description
from core.fitness import FitnessEvaluator
from core.pipeline_builder import build_pipeline
from data.dataset_loader import load_dataset
from optimizers import OPTIMIZER_REGISTRY, REQUIRED_OPTIMIZERS, BONUS_OPTIMIZERS


DEFAULT_CONFIG = {
    'datasets': ['heart', 'student', 'housing', 'diabetes'],
    'optimizers': REQUIRED_OPTIMIZERS,
    'pop_size': 20,
    'max_iter': 50,
    'cv_folds': 5,
    'n_runs': 3,
    'seed': 42,
    'output_dir': './results'
}

DATASET_CONFIG = {
    'heart':    {'task': 'classification', 'metric': 'accuracy'},
    'student':  {'task': 'classification', 'metric': 'f1_weighted'},
    'housing':  {'task': 'regression',     'metric': 'rmse'},
    'diabetes': {'task': 'regression',     'metric': 'rmse'},
}


class ExperimentRunner:
    """Runs AutoML experiments across multiple datasets and optimizers."""

    def __init__(self, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.results = []
        self.convergence_data = {}
        os.makedirs(self.config['output_dir'], exist_ok=True)

    def run_all(self, use_synthetic=False, verbose=True):
        """Run all experiments."""
        all_results = {}

        for dataset_name in self.config['datasets']:
            dataset_results = self.run_dataset(dataset_name, use_synthetic, verbose)
            all_results[dataset_name] = dataset_results

        return all_results

    def run_dataset(self, dataset_name, use_synthetic=False, verbose=True):
        """Run experiments for a single dataset."""
        # Load data
        try:
            X, y, task_type, dataset_info = load_dataset(
                dataset_name,
                raw_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw'),
                use_synthetic=use_synthetic
            )
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return []

        n_classes = dataset_info.get('n_classes', 2)
        ds_config = DATASET_CONFIG.get(dataset_name, {
            'task': task_type, 'metric': 'auto'
        })
        task = ds_config['task']
        metric = ds_config['metric']

        if verbose:
            self._print_dataset_header(dataset_info)

        bounds = get_bounds()
        dataset_results = []

        for opt_name in self.config['optimizers']:
            if opt_name not in OPTIMIZER_REGISTRY:
                print(f"  Warning: Unknown optimizer {opt_name}, skipping.")
                continue

            opt_class = OPTIMIZER_REGISTRY[opt_name]
            run_fitnesses = []
            run_times = []
            run_evals = []
            run_test_metrics = []
            run_convergences = []
            best_run_vector = None
            best_run_fitness = -np.inf

            for run_idx in range(self.config['n_runs']):
                seed = self.config['seed'] + run_idx * 100

                # Train/test split (stratified for classification)
                if task == 'classification':
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=seed, stratify=y
                    )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=seed
                    )

                # Create fitness evaluator
                evaluator = FitnessEvaluator(
                    X_train, y_train,
                    task=task,
                    cv=self.config['cv_folds'],
                    metric=metric,
                    n_classes=n_classes
                )

                # Create optimizer
                optimizer = opt_class(
                    fitness_fn=evaluator.evaluate,
                    bounds=bounds,
                    pop_size=self.config['pop_size'],
                    max_iter=self.config['max_iter'],
                    seed=seed,
                    task=task,
                    n_features=X_train.shape[1],
                    n_classes=n_classes,
                    verbose=verbose
                )

                # Run optimization
                best_vector, best_fitness = optimizer.optimize()
                results = optimizer.get_results()

                run_fitnesses.append(best_fitness)
                run_times.append(results['total_time'])
                run_evals.append(results['total_evals'])
                run_convergences.append(results['convergence_curve'])

                if best_fitness > best_run_fitness:
                    best_run_fitness = best_fitness
                    best_run_vector = best_vector.copy()

                # Evaluate on test set
                test_metrics = self._eval_test(
                    best_vector, X_train, X_test, y_train, y_test,
                    task, n_classes
                )
                run_test_metrics.append(test_metrics)

                if verbose:
                    self._print_completion(opt_name, best_fitness, test_metrics,
                                           results['total_time'], task)

            # Aggregate results
            mean_fitness = np.mean(run_fitnesses)
            std_fitness = np.std(run_fitnesses)
            mean_time = np.mean(run_times)
            mean_evals = np.mean(run_evals)

            pipeline_desc = get_pipeline_description(
                best_run_vector, task, X.shape[1], n_classes
            )

            # Store per-run results
            for run_idx, (fit, tm, ev, tm_) in enumerate(
                    zip(run_fitnesses, run_times, run_evals, run_test_metrics)):
                result_row = {
                    'dataset': dataset_name,
                    'optimizer': opt_name,
                    'run': run_idx,
                    'cv_fitness': fit,
                    'total_time': tm,
                    'n_evals': ev,
                    'best_pipeline': pipeline_desc,
                    'task': task
                }
                result_row.update(tm_)
                self.results.append(result_row)

            # Store convergence
            key = f"{dataset_name}_{opt_name}"
            self.convergence_data[key] = {
                'dataset': dataset_name,
                'optimizer': opt_name,
                'curves': run_convergences
            }

            # Summary for this optimizer
            summary = {
                'optimizer': opt_name,
                'mean_fitness': mean_fitness,
                'std_fitness': std_fitness,
                'mean_time': mean_time,
                'mean_evals': mean_evals,
                'pipeline': pipeline_desc,
                'test_metrics': run_test_metrics
            }

            # Average test metrics
            if run_test_metrics:
                for key_m in run_test_metrics[0]:
                    vals = [m[key_m] for m in run_test_metrics if key_m in m]
                    if vals:
                        summary[f'mean_{key_m}'] = np.mean(vals)

            dataset_results.append(summary)

        if verbose and dataset_results:
            self._print_results_table(dataset_name, dataset_results, task)

        return dataset_results

    def _eval_test(self, best_vector, X_train, X_test, y_train, y_test,
                   task, n_classes):
        """Evaluate best pipeline on test set."""
        try:
            decoded = decode_vector(best_vector, task, X_train.shape[1], n_classes)
            pipeline = build_pipeline(decoded, task)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            if task == 'classification':
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                return {'test_accuracy': acc, 'test_f1': f1}
            else:
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                return {'test_rmse': rmse, 'test_mae': mae, 'test_r2': r2}

        except Exception as e:
            if task == 'classification':
                return {'test_accuracy': 0.0, 'test_f1': 0.0}
            else:
                return {'test_rmse': 999.0, 'test_mae': 999.0, 'test_r2': 0.0}

    def _print_dataset_header(self, info):
        """Print dataset header box."""
        name = info.get('name', 'Unknown')
        task = info.get('task', 'unknown')
        n = info.get('n_samples', 0)
        f = info.get('n_features', 0)
        source = info.get('source', 'unknown')

        print(f"\n+{'='*54}+")
        print(f"|  Dataset: {name:<43}|")
        print(f"|  Task: {task:<10} | Samples: {n:<6} | Features: {f:<8}|")
        print(f"|  Source: {source:<45}|")
        print(f"+{'='*54}+")

    def _print_completion(self, opt_name, cv_fitness, test_metrics, total_time, task):
        """Print completion line."""
        if task == 'classification':
            acc = test_metrics.get('test_accuracy', 0.0)
            print(f"  [OK] {opt_name} Done | CV: {cv_fitness:.4f} | Test Acc: {acc:.4f} | Time: {total_time:.1f}s")
        else:
            rmse = test_metrics.get('test_rmse', 0.0)
            r2 = test_metrics.get('test_r2', 0.0)
            print(f"  [OK] {opt_name} Done | CV: {cv_fitness:.4f} | Test RMSE: {rmse:.4f} | R²: {r2:.4f} | Time: {total_time:.1f}s")

    def _print_results_table(self, dataset_name, results, task):
        """Print formatted results table."""
        ds_display = dataset_name.replace('_', ' ').title()
        task_display = 'Classification' if task == 'classification' else 'Regression'

        print(f"\n{'='*71}")
        print(f"RESULTS: {ds_display} — {task_display}")
        print(f"{'='*71}")

        if task == 'classification':
            print(f"{'Optimizer':<10}| {'CV Fitness':<15}| {'Test Acc':<12}| {'Test F1':<12}| {'Time':<8}| {'Evals'}")
            print(f"{'-'*10}+{'-'*15}+{'-'*12}+{'-'*12}+{'-'*8}+{'-'*6}")

            for r in results:
                opt = r['optimizer']
                cv = f"{r['mean_fitness']:.3f}±{r['std_fitness']:.3f}"
                acc = f"{r.get('mean_test_accuracy', 0.0):.4f}"
                f1 = f"{r.get('mean_test_f1', 0.0):.4f}"
                t = f"{r['mean_time']:.1f}s"
                ev = f"{int(r['mean_evals'])}"
                print(f"{opt:<10}| {cv:<15}| {acc:<12}| {f1:<12}| {t:<8}| {ev}")

        else:
            print(f"{'Optimizer':<10}| {'CV Fitness':<15}| {'Test RMSE':<12}| {'Test MAE':<12}| {'Test R²':<10}| {'Time'}")
            print(f"{'-'*10}+{'-'*15}+{'-'*12}+{'-'*12}+{'-'*10}+{'-'*8}")

            for r in results:
                opt = r['optimizer']
                cv = f"{r['mean_fitness']:.3f}±{r['std_fitness']:.3f}"
                rmse = f"{r.get('mean_test_rmse', 0.0):.4f}"
                mae = f"{r.get('mean_test_mae', 0.0):.4f}"
                r2 = f"{r.get('mean_test_r2', 0.0):.4f}"
                t = f"{r['mean_time']:.1f}s"
                print(f"{opt:<10}| {cv:<15}| {rmse:<12}| {mae:<12}| {r2:<10}| {t}")

        print(f"{'='*71}")
