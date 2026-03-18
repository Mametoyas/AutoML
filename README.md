# AutoML System with Metaheuristic Optimization

A complete, production-ready AutoML system that uses 10 metaheuristic optimization algorithms to automatically search for the best machine learning pipeline for any tabular dataset.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AutoML Metaheuristic System                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ┌──────────┐    ┌──────────────┐    ┌─────────────────────┐   │
│   │  Dataset │    │  Search Space│    │    Fitness           │   │
│   │  Loader  │───▶│  Encoding    │───▶│    Evaluator         │   │
│   │          │    │  (8-dim vec) │    │  (CV: acc/rmse/r2)   │   │
│   └──────────┘    └──────────────┘    └────────┬────────────┘   │
│                                                  │               │
│   ┌──────────────────────────────────────────────▼───────────┐  │
│   │              Metaheuristic Optimizers                     │  │
│   │  ┌────┐ ┌────┐ ┌────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐  │  │
│   │  │ GA │ │ GP │ │ DE │ │ PSO │ │ ACO │ │ ABC │ │ GWO │  │  │
│   │  └────┘ └────┘ └────┘ └─────┘ └─────┘ └─────┘ └─────┘  │  │
│   │  ┌─────┐ ┌─────┐ ┌────┐  (bonus algorithms)              │  │
│   │  │ WOA │ │ HHO │ │ CS │                                   │  │
│   │  └─────┘ └─────┘ └────┘                                   │  │
│   └───────────────────────────────────────────────────────────┘  │
│                         │                                         │
│   ┌─────────────────────▼───────────────────────────────────┐   │
│   │              Pipeline Builder                             │   │
│   │   [Scaler] → [Feature Selection] → [Model]               │   │
│   └─────────────────────────────────────────────────────────┘   │
│                         │                                         │
│   ┌─────────────────────▼───────────────────────────────────┐   │
│   │         Results Logger + Visualization                    │   │
│   │   results.csv  convergence.json  plots/*.png              │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Solution Encoding (8-Dimensional Vector)

| Index | Name | Type | Range | Description |
|-------|------|------|-------|-------------|
| 0 | scaler_id | int | 0–2 | None / StandardScaler / MinMaxScaler |
| 1 | feature_id | int | 0–2 | None / SelectKBest / PCA |
| 2 | feature_param | float | [0,1] | k or n_components ratio |
| 3 | model_id | int | 0–4 | LR/Ridge · SVM/SVR · RF · XGB · MLP |
| 4 | hyperparam_1 | float | [0,1] | Model-specific (C, n_estimators, lr, ...) |
| 5 | hyperparam_2 | float | [0,1] | Model-specific (gamma, max_depth, ...) |
| 6 | hyperparam_3 | float | [0,1] | Model-specific (min_samples_split, ...) |
| 7 | hyperparam_4 | float | [0,1] | Model-specific (subsample, ...) |

## Pipeline Example

| Vector | Decoded Pipeline |
|--------|-----------------|
| [1, 1, 0.6, 3, 0.3, 0.5, 0.7, 0.8] | Standard → SelectKBest(k=8) → XGBoost(lr=0.097, n=255, depth=7) |
| [2, 2, 0.4, 4, 0.5, 0.2, 0.3, 0.5] | MinMax → PCA(n=5) → MLP(128, 64) |
| [0, 0, 0.5, 2, 0.8, 0.6, 0.4, 0.5] | None → None → RandomForest(n=402, depth=12) |

## Fitness Functions

### Classification
```
fitness = (1/K) × Σ accuracy_k    (for metric='accuracy')
fitness = (1/K) × Σ f1_weighted_k  (for metric='f1_weighted')
Range: [0, 1], higher is better
```

### Regression
```
RMSE_cv = sqrt((1/K) × Σ MSE_k)
MAE_cv  = (1/K) × Σ MAE_k
R²_cv   = (1/K) × Σ R²_k

fitness_rmse = 1 / (1 + RMSE_cv)    → range (0, 1]
fitness_mae  = 1 / (1 + MAE_cv)     → range (0, 1]
fitness_r2   = max(0, R²_cv)        → range [0, 1]
```

All three regression metrics (RMSE, MAE, R²) are computed and reported.

## Algorithm Reference

| # | Optimizer | Year | Category | AutoML Use Case | Required |
|---|-----------|------|----------|-----------------|----------|
| 1 | GA | 1992 | Evolutionary | Discrete pipeline search | YES |
| 2 | GP | 1992 | Evolutionary | Tree-structured pipelines | YES |
| 3 | DE | 1997 | Evolutionary | Continuous hyperparam tuning | YES |
| 4 | PSO | 1995 | Swarm | Fast float-space convergence | YES |
| 5 | ACO | 1996 | Swarm | Categorical component select | YES |
| 6 | ABC | 2005 | Swarm | Local neighborhood exploit | YES |
| 7 | GWO | 2014 | Swarm | α/β/δ hierarchy guiding | YES |
| 8 | WOA | 2016 | Swarm | Spiral hyperparam exploit | bonus |
| 9 | HHO | 2019 | Swarm | Multi-phase escape | bonus |
| 10 | CS | 2009 | Evolutionary | Lévy flight diversity | bonus |

## Datasets

| Name | Kaggle Slug | Task | Fallback |
|------|-------------|------|----------|
| Heart Disease | sintariosatya/heart-disease-dataset | Classification | make_classification |
| Student Perf. | nabeelqureshitiii/student-performance-dataset | Classification | make_classification |
| Housing Prices | yasserh/housing-prices-dataset | Regression | california_housing |
| Diabetes (sklearn) | — built-in — | Regression | always available |

## Quick Start

### Option A — With Kaggle Datasets
```bash
pip install -r requirements.txt
python setup_kaggle.py --download
python main.py --quick     # quick test
python main.py             # full run
```

### Option B — Without Kaggle (Synthetic/Built-in Data)
```bash
pip install -r requirements.txt
python main.py --use-synthetic --quick
python main.py --use-synthetic
```

### Option C — Full Run with All 10 Algorithms
```bash
python main.py --all
python main.py --all --use-synthetic
```

### Other Options
```bash
# Select specific datasets and optimizers
python main.py --datasets heart diabetes --optimizers GA DE GWO

# Tune hyperparameters
python main.py --pop-size 30 --max-iter 100 --n-runs 5

# Skip plots (faster)
python main.py --no-plots
```

## Project Structure
```
automl_metaheuristic/
├── README.md
├── requirements.txt
├── setup_kaggle.py
├── main.py
├── core/
│   ├── search_space.py      # 8-dim vector encoding & bounds
│   ├── pipeline_builder.py  # sklearn Pipeline factory
│   └── fitness.py           # CV evaluator (RMSE + MAE + R²)
├── optimizers/
│   ├── base_optimizer.py
│   ├── genetic_algorithm.py     # GA
│   ├── genetic_programming.py   # GP
│   ├── differential_evolution.py # DE
│   ├── particle_swarm.py        # PSO
│   ├── ant_colony.py            # ACO
│   ├── artificial_bee_colony.py # ABC
│   ├── grey_wolf_optimizer.py   # GWO ← REQUIRED
│   ├── whale_optimization.py    # WOA (bonus)
│   ├── harris_hawks.py          # HHO (bonus)
│   └── cuckoo_search.py         # CS  (bonus)
├── data/
│   ├── dataset_loader.py
│   ├── kaggle_downloader.py
│   └── raw/
├── experiments/
│   ├── runner.py
│   └── results_logger.py
└── visualization/
    └── plot_results.py
```

## References

- **GA**: Holland, J.H. (1992). *Adaptation in Natural and Artificial Systems*. MIT Press.
- **GP**: Koza, J.R. (1992). *Genetic Programming*. MIT Press.
- **DE**: Storn, R. & Price, K. (1997). Differential Evolution – A Simple and Efficient Heuristic for Global Optimization. *Journal of Global Optimization*, 11(4), 341-359.
- **PSO**: Kennedy, J. & Eberhart, R. (1995). Particle Swarm Optimization. *Proceedings of ICNN*, 1942-1948.
- **ACO**: Dorigo, M., Maniezzo, V., & Colorni, A. (1996). Ant System: Optimization by a Colony of Cooperating Agents. *IEEE Transactions on Systems, Man, and Cybernetics*, 26(1), 29-41.
- **ABC**: Karaboga, D. (2005). An Idea Based on Honey Bee Swarm for Numerical Optimization. Technical Report TR-06, Erciyes University.
- **GWO**: Mirjalili, S., Mirjalili, S.M., & Lewis, A. (2014). Grey Wolf Optimizer. *Advances in Engineering Software*, 69, 46-61.
- **WOA**: Mirjalili, S. & Lewis, A. (2016). The Whale Optimization Algorithm. *Advances in Engineering Software*, 95, 51-67.
- **HHO**: Heidari, A.A. et al. (2019). Harris Hawks Optimization. *Future Generation Computer Systems*, 97, 849-872.
- **CS**: Yang, X.S. & Deb, S. (2009). Cuckoo Search via Lévy Flights. *Proceedings of NaBIC 2009*, 210-214.
