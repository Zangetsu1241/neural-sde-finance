"""Research-grade toy neural SDE benchmark, without live trading claims."""
from .models import NeuralSDE, GBM, HestonFilter
from .data import simulate_heston, load_prices, prepare
from .research import Config, experiment
__all__ = ["NeuralSDE", "GBM", "HestonFilter", "simulate_heston", "load_prices", "prepare", "Config", "experiment"]