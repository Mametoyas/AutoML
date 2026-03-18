"""
AutoML System with Metaheuristic Optimization
Main entry point.

Usage:
  python main.py                         # full run with 7 required optimizers
  python main.py --quick                 # fast run (pop=10, iter=15, n_runs=1)
  python main.py --all                   # include 3 bonus optimizers (10 total)
  python main.py --use-synthetic         # skip Kaggle, use built-in datasets
  python main.py --datasets heart diabetes
  python main.py --optimizers GA DE GWO
"""
import os
import sys
import io
import time
import argparse

# Ensure project root is in path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

import numpy as np


BANNER = """
+==================================================================+
|      AutoML System with Metaheuristic Optimization               |
|      7 Required Algorithms + 3 Bonus Algorithms                  |
|      GA | GP | DE | PSO | ACO | ABC | GWO                        |
|      + WOA | HHO | CS (bonus, use --all)                         |
+==================================================================+
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description='AutoML System with Metaheuristic Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--datasets', nargs='+',
                        default=['heart', 'student', 'housing', 'diabetes'],
                        choices=['heart', 'student', 'housing', 'diabetes'],
                        help='Datasets to run (default: all 4)')
    parser.add_argument('--optimizers', nargs='+',
                        default=None,
                        help='Optimizers to run (default: 7 required)')
    parser.add_argument('--pop-size', type=int, default=20,
                        help='Population size (default: 20)')
    parser.add_argument('--max-iter', type=int, default=50,
                        help='Max iterations (default: 50)')
    parser.add_argument('--n-runs', type=int, default=3,
                        help='Number of runs per optimizer (default: 3)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: pop=10, iter=15, n_runs=1')
    parser.add_argument('--all', action='store_true',
                        help='Include bonus optimizers WOA, HHO, CS (10 total)')
    parser.add_argument('--use-synthetic', action='store_true',
                        help='Use synthetic/built-in datasets (skip Kaggle)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip visualization (faster)')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Output directory for results (default: ./results)')
    return parser.parse_args()


def check_dataset_availability(raw_dir, use_synthetic):
    """Check and print dataset availability status."""
    print("\n📋 Dataset Availability Check:")

    datasets_info = [
        ('heart',    'heart.csv',               'Kaggle: sintariosatya/heart-disease-dataset'),
        ('student',  'student_performance.csv',  'Kaggle: nabeelqureshitiii/student-performance-dataset'),
        ('housing',  'housing.csv',              'Kaggle: yasserh/housing-prices-dataset'),
        ('diabetes', None,                        'sklearn built-in (always available)'),
    ]

    for name, filename, source in datasets_info:
        if filename is None:
            print(f"  [OK] {name:<12} — {source}")
        elif use_synthetic:
            print(f"  [!]️  {name:<12} — Using synthetic fallback (--use-synthetic)")
        else:
            path = os.path.join(raw_dir, filename)
            if os.path.exists(path):
                print(f"  [OK] {name:<12} — {source}")
            else:
                print(f"  [!]️  {name:<12} — Not found, using fallback (run setup_kaggle.py --download)")


def main():
    args = parse_args()

    print(BANNER)

    # Apply --quick flag
    if args.quick:
        pop_size = 10
        max_iter = 15
        n_runs = 1
        print("⚡ Quick mode: pop=10, iter=15, n_runs=1")
    else:
        pop_size = args.pop_size
        max_iter = args.max_iter
        n_runs = args.n_runs

    # Determine optimizers
    from optimizers import OPTIMIZER_REGISTRY, REQUIRED_OPTIMIZERS, BONUS_OPTIMIZERS, DEFAULT_RUN

    if args.optimizers:
        optimizers = args.optimizers
    elif getattr(args, 'all', False):
        optimizers = REQUIRED_OPTIMIZERS + BONUS_OPTIMIZERS
    else:
        optimizers = DEFAULT_RUN  # 7 required

    # Validate optimizers
    valid_opts = [o for o in optimizers if o in OPTIMIZER_REGISTRY]
    invalid_opts = [o for o in optimizers if o not in OPTIMIZER_REGISTRY]
    if invalid_opts:
        print(f"  [!]️  Unknown optimizers: {invalid_opts}")
    optimizers = valid_opts

    print(f"\n🔧 Configuration:")
    print(f"   Datasets:   {args.datasets}")
    print(f"   Optimizers: {optimizers}")
    print(f"   Pop size:   {pop_size}")
    print(f"   Max iter:   {max_iter}")
    print(f"   N runs:     {n_runs}")
    print(f"   Output dir: {args.output_dir}")

    # Check datasets
    raw_dir = os.path.join(_ROOT, 'data', 'raw')
    check_dataset_availability(raw_dir, args.use_synthetic)

    # Print search space summary
    from core.search_space import get_search_space_info
    get_search_space_info()

    # Run experiments
    from experiments.runner import ExperimentRunner
    from experiments.results_logger import ResultsLogger

    config = {
        'datasets': args.datasets,
        'optimizers': optimizers,
        'pop_size': pop_size,
        'max_iter': max_iter,
        'cv_folds': 5,
        'n_runs': n_runs,
        'seed': 42,
        'output_dir': args.output_dir
    }

    print(f"\n{'='*60}")
    print("Starting AutoML Experiments...")
    print(f"{'='*60}")

    wall_clock_start = time.time()

    runner = ExperimentRunner(config)
    all_results = runner.run_all(use_synthetic=args.use_synthetic, verbose=True)

    # Save results
    logger = ResultsLogger(output_dir=args.output_dir)
    logger.save_results(runner.results, runner.convergence_data)
    logger.print_summary_table(runner.results)
    logger.print_best_pipelines(runner.results)

    # Generate plots
    if not args.no_plots:
        print(f"\n[art] Generating plots...")
        try:
            from visualization.plot_results import generate_all_plots
            generate_all_plots(all_results, runner.convergence_data, args.output_dir)
        except Exception as e:
            print(f"  [!]️  Plot generation failed: {e}")

    wall_clock_end = time.time()
    total_time = wall_clock_end - wall_clock_start

    print(f"\n{'='*60}")
    print(f"[OK] All experiments complete!")
    print(f"   Total wall-clock time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"   Results saved to: {args.output_dir}/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
