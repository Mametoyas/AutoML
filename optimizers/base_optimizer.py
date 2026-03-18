"""
Abstract Base Optimizer class for all metaheuristic optimizers.
"""
import time
import numpy as np
from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    """
    Abstract base class for all metaheuristic optimizers.
    All optimizers maximize fitness (higher = better).
    """

    optimizer_name = "BaseOptimizer"

    def __init__(self, fitness_fn, bounds, pop_size=20, max_iter=50, seed=42):
        """
        Initialize optimizer.

        Args:
            fitness_fn: callable, takes vector, returns float (higher = better)
            bounds: tuple (lower_bounds, upper_bounds) as np.arrays
            pop_size: population size
            max_iter: maximum iterations
            seed: random seed for reproducibility
        """
        self.fitness_fn = fitness_fn
        self.lb = np.array(bounds[0], dtype=float)
        self.ub = np.array(bounds[1], dtype=float)
        self.dim = len(self.lb)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Results
        self.best_vector = None
        self.best_fitness = -np.inf
        self.convergence_curve = []
        self.total_time = 0.0
        self.total_evals = 0
        self._start_time = None

    @abstractmethod
    def optimize(self):
        """
        Run optimization.

        Returns:
            tuple: (best_vector, best_fitness)
        """
        pass

    def clip_vector(self, vec):
        """Clip vector to bounds."""
        return np.clip(vec, self.lb, self.ub)

    def random_vector(self):
        """Generate random vector within bounds."""
        return self.lb + self.rng.random(self.dim) * (self.ub - self.lb)

    def _evaluate(self, vec):
        """Evaluate fitness with bounds clipping."""
        vec = self.clip_vector(vec)
        fitness = self.fitness_fn(vec)
        self.total_evals += 1
        return fitness

    def get_results(self):
        """Return optimization results as dict."""
        return {
            'best_vector': self.best_vector,
            'best_fitness': self.best_fitness,
            'convergence_curve': self.convergence_curve,
            'total_time': self.total_time,
            'total_evals': self.total_evals,
            'optimizer_name': self.optimizer_name
        }

    def _start_timer(self):
        self._start_time = time.time()

    def _stop_timer(self):
        self.total_time = time.time() - self._start_time
