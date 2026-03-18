"""
Artificial Bee Colony (ABC) optimizer.
"""
import numpy as np
from .base_optimizer import BaseOptimizer
from core.search_space import get_pipeline_description


class ArtificialBeeColony(BaseOptimizer):
    """
    ABC with employed, onlooker, and scout phases.
    """

    optimizer_name = "Artificial Bee Colony (ABC)"

    def __init__(self, fitness_fn, bounds, pop_size=20, max_iter=50, seed=42,
                 limit=10,
                 task='classification', n_features=10, n_classes=2, verbose=True):
        super().__init__(fitness_fn, bounds, pop_size, max_iter, seed)
        self.limit = limit
        self.n_employed = max(2, pop_size // 2)
        self.n_onlooker = max(2, pop_size // 2)
        self.task = task
        self.n_features = n_features
        self.n_classes = n_classes
        self.verbose = verbose

    def _neighborhood_search(self, sources, fitness_vals, i):
        """Perform neighborhood search around source i."""
        candidates = [j for j in range(len(sources)) if j != i]
        k = self.rng.choice(candidates)
        j = self.rng.randint(0, self.dim)
        phi = self.rng.uniform(-1, 1)

        new_source = sources[i].copy()
        new_source[j] = sources[i][j] + phi * (sources[i][j] - sources[k][j])
        new_source = self.clip_vector(new_source)
        return new_source

    def optimize(self):
        """Run ABC optimization."""
        self._start_timer()

        # Initialize food sources
        sources = np.array([self.random_vector() for _ in range(self.n_employed)])
        fitness_vals = np.array([self._evaluate(s) for s in sources])
        trials = np.zeros(self.n_employed, dtype=int)

        best_idx = np.argmax(fitness_vals)
        self.best_vector = sources[best_idx].copy()
        self.best_fitness = fitness_vals[best_idx]

        for t in range(self.max_iter):
            # --- Employed phase ---
            for i in range(self.n_employed):
                new_source = self._neighborhood_search(sources, fitness_vals, i)
                new_fit = self._evaluate(new_source)
                if new_fit >= fitness_vals[i]:
                    sources[i] = new_source
                    fitness_vals[i] = new_fit
                    trials[i] = 0
                else:
                    trials[i] += 1

            # --- Onlooker phase ---
            total_fit = np.sum(fitness_vals)
            if total_fit > 0:
                probs = fitness_vals / total_fit
            else:
                probs = np.ones(self.n_employed) / self.n_employed

            for _ in range(self.n_onlooker):
                i = self.rng.choice(self.n_employed, p=probs)
                new_source = self._neighborhood_search(sources, fitness_vals, i)
                new_fit = self._evaluate(new_source)
                if new_fit >= fitness_vals[i]:
                    sources[i] = new_source
                    fitness_vals[i] = new_fit
                    trials[i] = 0
                else:
                    trials[i] += 1

            # --- Scout phase ---
            for i in range(self.n_employed):
                if trials[i] > self.limit:
                    sources[i] = self.random_vector()
                    fitness_vals[i] = self._evaluate(sources[i])
                    trials[i] = 0

            # Update best
            best_idx = np.argmax(fitness_vals)
            if fitness_vals[best_idx] > self.best_fitness:
                self.best_fitness = fitness_vals[best_idx]
                self.best_vector = sources[best_idx].copy()

            self.convergence_curve.append(self.best_fitness)

            if self.verbose:
                pipeline_desc = get_pipeline_description(self.best_vector, self.task, self.n_features, self.n_classes)
                print(f"\r[ABC] Iter {t+1:2d}/{self.max_iter} | Best: {self.best_fitness:.4f} | Pipeline: {pipeline_desc}", end='', flush=True)

        if self.verbose:
            print()

        self._stop_timer()
        return self.best_vector, self.best_fitness
