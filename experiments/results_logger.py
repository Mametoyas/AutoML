"""
Results Logger: saves experiment results to CSV and JSON.
"""
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime


class ResultsLogger:
    """Logs and saves experiment results."""

    def __init__(self, output_dir='./results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def save_results(self, results_list, convergence_data):
        """
        Save results to CSV and convergence data to JSON.

        Args:
            results_list: list of result dicts from runner
            convergence_data: dict of convergence curves
        """
        if results_list:
            df = pd.DataFrame(results_list)
            csv_path = os.path.join(self.output_dir, 'results.csv')
            df.to_csv(csv_path, index=False)
            print(f"\n[chart] Results saved to: {csv_path}")

        if convergence_data:
            # Convert numpy arrays to lists for JSON
            json_data = {}
            for key, data in convergence_data.items():
                json_data[key] = {
                    'dataset': data['dataset'],
                    'optimizer': data['optimizer'],
                    'curves': [
                        [float(v) for v in curve]
                        for curve in data.get('curves', [])
                    ]
                }

            json_path = os.path.join(self.output_dir, 'convergence.json')
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"[chart] Convergence data saved to: {json_path}")

    def print_summary_table(self, results_list):
        """Print summary table grouped by task type."""
        if not results_list:
            print("No results to display.")
            return

        df = pd.DataFrame(results_list)

        # Classification datasets
        clf_data = df[df['task'] == 'classification']
        if not clf_data.empty:
            print(f"\n{'='*80}")
            print("SUMMARY — CLASSIFICATION DATASETS")
            print(f"{'='*80}")
            print(f"{'Dataset':<12}{'Optimizer':<10}{'CV Fitness':<15}{'Test Acc':<12}{'Test F1':<12}{'Time':<10}{'Evals'}")
            print(f"{'-'*80}")

            for (ds, opt), grp in clf_data.groupby(['dataset', 'optimizer']):
                cv = f"{grp['cv_fitness'].mean():.3f}±{grp['cv_fitness'].std():.3f}"
                acc = f"{grp['test_accuracy'].mean():.4f}" if 'test_accuracy' in grp else 'N/A'
                f1 = f"{grp['test_f1'].mean():.4f}" if 'test_f1' in grp else 'N/A'
                t = f"{grp['total_time'].mean():.1f}s"
                ev = f"{int(grp['n_evals'].mean())}"
                print(f"{ds:<12}{opt:<10}{cv:<15}{acc:<12}{f1:<12}{t:<10}{ev}")
            print(f"{'='*80}")

        # Regression datasets
        reg_data = df[df['task'] == 'regression']
        if not reg_data.empty:
            print(f"\n{'='*90}")
            print("SUMMARY — REGRESSION DATASETS")
            print(f"{'='*90}")
            print(f"{'Dataset':<12}{'Optimizer':<10}{'CV Fitness':<15}{'Test RMSE':<12}{'Test MAE':<12}{'Test R²':<12}{'Time'}")
            print(f"{'-'*90}")

            for (ds, opt), grp in reg_data.groupby(['dataset', 'optimizer']):
                cv = f"{grp['cv_fitness'].mean():.3f}±{grp['cv_fitness'].std():.3f}"
                rmse = f"{grp['test_rmse'].mean():.4f}" if 'test_rmse' in grp else 'N/A'
                mae = f"{grp['test_mae'].mean():.4f}" if 'test_mae' in grp else 'N/A'
                r2 = f"{grp['test_r2'].mean():.4f}" if 'test_r2' in grp else 'N/A'
                t = f"{grp['total_time'].mean():.1f}s"
                print(f"{ds:<12}{opt:<10}{cv:<15}{rmse:<12}{mae:<12}{r2:<12}{t}")
            print(f"{'='*90}")

    def print_best_pipelines(self, results_list):
        """Print best pipeline per optimizer per dataset."""
        if not results_list:
            return

        df = pd.DataFrame(results_list)

        print(f"\n{'='*70}")
        print("BEST PIPELINES PER OPTIMIZER PER DATASET")
        print(f"{'='*70}")

        for (ds, opt), grp in df.groupby(['dataset', 'optimizer']):
            best_idx = grp['cv_fitness'].idxmax()
            best_row = grp.loc[best_idx]
            pipeline = best_row.get('best_pipeline', 'N/A')
            fitness = best_row.get('cv_fitness', 0.0)
            print(f"  {ds}/{opt}: {pipeline} (CV={fitness:.4f})")

        print(f"{'='*70}")
