"""
Optimizer Registry for AutoML Metaheuristic System.
"""
from .genetic_algorithm import GeneticAlgorithm
from .genetic_programming import GeneticProgramming
from .differential_evolution import DifferentialEvolution
from .particle_swarm import ParticleSwarmOptimization
from .ant_colony import AntColonyOptimization
from .artificial_bee_colony import ArtificialBeeColony
from .grey_wolf_optimizer import GreyWolfOptimizer
from .whale_optimization import WhaleOptimizationAlgorithm
from .harris_hawks import HarrisHawksOptimization
from .cuckoo_search import CuckooSearch

OPTIMIZER_REGISTRY = {
    'GA':  GeneticAlgorithm,
    'GP':  GeneticProgramming,
    'DE':  DifferentialEvolution,
    'PSO': ParticleSwarmOptimization,
    'ACO': AntColonyOptimization,
    'ABC': ArtificialBeeColony,
    'GWO': GreyWolfOptimizer,        # REQUIRED by spec
    'WOA': WhaleOptimizationAlgorithm,
    'HHO': HarrisHawksOptimization,
    'CS':  CuckooSearch,
}

REQUIRED_OPTIMIZERS = ['GA', 'GP', 'DE', 'PSO', 'ACO', 'ABC', 'GWO']
BONUS_OPTIMIZERS = ['WOA', 'HHO', 'CS']
DEFAULT_RUN = REQUIRED_OPTIMIZERS  # run all 7 by default

__all__ = [
    'GeneticAlgorithm', 'GeneticProgramming', 'DifferentialEvolution',
    'ParticleSwarmOptimization', 'AntColonyOptimization', 'ArtificialBeeColony',
    'GreyWolfOptimizer', 'WhaleOptimizationAlgorithm', 'HarrisHawksOptimization',
    'CuckooSearch', 'OPTIMIZER_REGISTRY', 'REQUIRED_OPTIMIZERS',
    'BONUS_OPTIMIZERS', 'DEFAULT_RUN'
]
