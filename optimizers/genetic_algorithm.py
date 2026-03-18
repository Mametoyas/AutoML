"""
Genetic Algorithm (GA) optimizer.
"""
import numpy as np
from tqdm import tqdm
from .base_optimizer import BaseOptimizer
from core.search_space import get_pipeline_description


class GeneticAlgorithm(BaseOptimizer):
    """
    Genetic Algorithm with tournament selection, single-point crossover,
    Gaussian mutation, and elitism.
    """

    optimizer_name = "Genetic Algorithm (GA)"

    def __init__(self, fitness_fn, bounds, pop_size=20, max_iter=50, seed=42,
                 crossover_rate=0.8, mutation_rate=0.1, tournament_k=3,
                 task='classification', n_features=10, n_classes=2, verbose=True):
        super().__init__(fitness_fn, bounds, pop_size, max_iter, seed)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.task = task
        self.n_features = n_features
        self.n_classes = n_classes
        self.verbose = verbose

    def _tournament_select(self, population, fitness_values):
        """Tournament selection."""
        indices = self.rng.choice(len(population), self.tournament_k, replace=False)
        best_idx = indices[np.argmax(fitness_values[indices])]
        return population[best_idx].copy()

    def _crossover(self, p1, p2):
        """Single-point crossover."""
        if self.rng.random() < self.crossover_rate:
            cp = self.rng.randint(1, self.dim)
            c1 = np.concatenate([p1[:cp], p2[cp:]])
            c2 = np.concatenate([p2[:cp], p1[cp:]])
            return c1, c2
        return p1.copy(), p2.copy()

    def _mutate(self, vec):
        """Gaussian mutation."""
        vec = vec.copy()
        for j in range(self.dim):
            if self.rng.random() < self.mutation_rate:
                vec[j] += self.rng.normal(0, 0.1 * (self.ub[j] - self.lb[j]))
        return self.clip_vector(vec)

    def optimize(self):
        """Run GA optimization."""
        self._start_timer()

        # Initialize population
        population = np.array([self.random_vector() for _ in range(self.pop_size)])
        fitness_values = np.array([self._evaluate(p) for p in population])

        best_idx = np.argmax(fitness_values)
        self.best_vector = population[best_idx].copy()
        self.best_fitness = fitness_values[best_idx]

        pbar = tqdm(range(self.max_iter), desc=f'[GA]', leave=False) if self.verbose else range(self.max_iter)

        for t in pbar:
            new_population = []

            # Elitism: keep best 2
            elite_indices = np.argsort(fitness_values)[-2:]
            elites = [population[i].copy() for i in elite_indices]

            # Generate offspring
            while len(new_population) < self.pop_size - 2:
                p1 = self._tournament_select(population, fitness_values)
                p2 = self._tournament_select(population, fitness_values)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                new_population.append(c1)
                if len(new_population) < self.pop_size - 2:
                    new_population.append(c2)

            # Add elites
            new_population.extend(elites)
            population = np.array(new_population)
            fitness_values = np.array([self._evaluate(p) for p in population])

            best_idx = np.argmax(fitness_values)
            if fitness_values[best_idx] > self.best_fitness:
                self.best_fitness = fitness_values[best_idx]
                self.best_vector = population[best_idx].copy()

            self.convergence_curve.append(self.best_fitness)

            if self.verbose:
                pipeline_desc = get_pipeline_description(self.best_vector, self.task, self.n_features, self.n_classes)
                print(f"\r[GA]  Iter {t+1:2d}/{self.max_iter} | Best: {self.best_fitness:.4f} | Pipeline: {pipeline_desc}", end='', flush=True)

        if self.verbose:
            print()

        self._stop_timer()
        return self.best_vector, self.best_fitness
