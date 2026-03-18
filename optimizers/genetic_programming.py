"""
Genetic Programming (GP) optimizer with tree-structured representation.
"""
import numpy as np
from tqdm import tqdm
from .base_optimizer import BaseOptimizer
from core.search_space import get_pipeline_description


class GeneticProgramming(BaseOptimizer):
    """
    Genetic Programming with tree-structured pipeline representation,
    subtree crossover, and point mutation.
    """

    optimizer_name = "Genetic Programming (GP)"

    def __init__(self, fitness_fn, bounds, pop_size=20, max_iter=50, seed=42,
                 crossover_rate=0.8, mutation_rate=0.15, tournament_k=3,
                 task='classification', n_features=10, n_classes=2, verbose=True):
        super().__init__(fitness_fn, bounds, pop_size, max_iter, seed)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.task = task
        self.n_features = n_features
        self.n_classes = n_classes
        self.verbose = verbose

    def _random_tree(self):
        """Create a random pipeline tree."""
        return {
            'scaler': self.rng.randint(0, 3),
            'feature': {
                'method': self.rng.randint(0, 3),
                'param': self.rng.random()
            },
            'model': {
                'id': self.rng.randint(0, 5),
                'params': list(self.rng.random(4))
            }
        }

    def _tree_to_vector(self, tree):
        """Convert tree dict to vector[8]."""
        vec = np.zeros(8)
        vec[0] = min(tree['scaler'], 2) + 0.0
        vec[1] = min(tree['feature']['method'], 2) + 0.0
        vec[2] = np.clip(tree['feature']['param'], 0.0, 1.0)
        vec[3] = min(tree['model']['id'], 4) + 0.0
        params = tree['model']['params']
        for i, p in enumerate(params[:4]):
            vec[4 + i] = np.clip(p, 0.0, 1.0)
        return self.clip_vector(vec)

    def _vector_to_tree(self, vec):
        """Convert vector to tree dict."""
        return {
            'scaler': int(np.clip(vec[0], 0, 2.99)),
            'feature': {
                'method': int(np.clip(vec[1], 0, 2.99)),
                'param': float(np.clip(vec[2], 0.0, 1.0))
            },
            'model': {
                'id': int(np.clip(vec[3], 0, 4.99)),
                'params': [float(np.clip(vec[i], 0.0, 1.0)) for i in range(4, 8)]
            }
        }

    def _subtree_crossover(self, t1, t2):
        """Swap subtree at random node."""
        t1 = {
            'scaler': t1['scaler'],
            'feature': dict(t1['feature']),
            'model': {'id': t1['model']['id'], 'params': list(t1['model']['params'])}
        }
        t2 = {
            'scaler': t2['scaler'],
            'feature': dict(t2['feature']),
            'model': {'id': t2['model']['id'], 'params': list(t2['model']['params'])}
        }

        choice = self.rng.choice(['scaler', 'feature', 'model', 'model.params'])

        if choice == 'scaler':
            t1['scaler'], t2['scaler'] = t2['scaler'], t1['scaler']
        elif choice == 'feature':
            t1['feature'], t2['feature'] = t2['feature'], t1['feature']
        elif choice == 'model':
            t1['model']['id'], t2['model']['id'] = t2['model']['id'], t1['model']['id']
        elif choice == 'model.params':
            t1['model']['params'], t2['model']['params'] = t2['model']['params'], t1['model']['params']

        return t1, t2

    def _point_mutation(self, tree, rate=0.15):
        """Mutate each node independently."""
        tree = {
            'scaler': tree['scaler'],
            'feature': dict(tree['feature']),
            'model': {'id': tree['model']['id'], 'params': list(tree['model']['params'])}
        }

        if self.rng.random() < rate:
            tree['scaler'] = self.rng.randint(0, 3)
        if self.rng.random() < rate:
            tree['feature']['method'] = self.rng.randint(0, 3)
        if self.rng.random() < rate:
            tree['feature']['param'] = self.rng.random()
        if self.rng.random() < rate:
            tree['model']['id'] = self.rng.randint(0, 5)
        for i in range(4):
            if self.rng.random() < rate:
                tree['model']['params'][i] = self.rng.random()

        return tree

    def _tournament_select(self, trees, fitness_values):
        """Tournament selection."""
        indices = self.rng.choice(len(trees), self.tournament_k, replace=False)
        best_idx = indices[np.argmax([fitness_values[i] for i in indices])]
        return trees[best_idx]

    def optimize(self):
        """Run GP optimization."""
        self._start_timer()

        # Initialize population of trees
        trees = [self._random_tree() for _ in range(self.pop_size)]
        vectors = [self._tree_to_vector(t) for t in trees]
        fitness_values = [self._evaluate(v) for v in vectors]

        best_idx = np.argmax(fitness_values)
        self.best_vector = vectors[best_idx].copy()
        self.best_fitness = fitness_values[best_idx]
        best_tree = trees[best_idx]

        for t in range(self.max_iter):
            new_trees = []

            # Elitism
            elite_indices = np.argsort(fitness_values)[-2:]
            elites = [trees[i] for i in elite_indices]

            while len(new_trees) < self.pop_size - 2:
                p1 = self._tournament_select(trees, fitness_values)
                p2 = self._tournament_select(trees, fitness_values)

                if self.rng.random() < self.crossover_rate:
                    c1, c2 = self._subtree_crossover(p1, p2)
                else:
                    c1, c2 = p1, p2

                c1 = self._point_mutation(c1, self.mutation_rate)
                c2 = self._point_mutation(c2, self.mutation_rate)
                new_trees.append(c1)
                if len(new_trees) < self.pop_size - 2:
                    new_trees.append(c2)

            new_trees.extend(elites)
            trees = new_trees
            vectors = [self._tree_to_vector(tree) for tree in trees]
            fitness_values = [self._evaluate(v) for v in vectors]

            best_idx = np.argmax(fitness_values)
            if fitness_values[best_idx] > self.best_fitness:
                self.best_fitness = fitness_values[best_idx]
                self.best_vector = vectors[best_idx].copy()
                best_tree = trees[best_idx]

            self.convergence_curve.append(self.best_fitness)

            if self.verbose:
                pipeline_desc = get_pipeline_description(self.best_vector, self.task, self.n_features, self.n_classes)
                print(f"\r[GP]  Iter {t+1:2d}/{self.max_iter} | Best: {self.best_fitness:.4f} | Pipeline: {pipeline_desc}", end='', flush=True)

        if self.verbose:
            print()

        self._stop_timer()
        return self.best_vector, self.best_fitness
