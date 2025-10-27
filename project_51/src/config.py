"""
Configuration settings for the Inventory Transfer Optimization System.

This module contains all configuration parameters used throughout the project.
"""

from pathlib import Path
from typing import Optional

# ============================================================================
# CORE SETTINGS
# ============================================================================

# Global random seed for reproducibility
RANDOM_SEED = 2025

# Project directories
DATA_DIR = "data"
RESULTS_DIR = "results"
VISUALIZATIONS_DIR = "visualizations"
LOGS_DIR = "logs"

# Required data files
REQUIRED_DATA_FILES = [
    "sales_data.csv",
    "inventory_data.csv",
    "stores.csv",
    "products.csv",
    "distance_matrix.csv",
    "transport_cost_matrix.csv",
]

# ============================================================================
# DATA GENERATION
# ============================================================================

# Product and store settings
NUM_PRODUCTS = 30
SALES_DAYS = 90

# Inventory thresholds (in days)
MIN_INVENTORY_DAYS = 7  # Below this is shortage
MAX_INVENTORY_DAYS = 21  # Above this is excess

# Inventory distribution percentages
EXCESS_PERCENT = 20  # Percentage of items with excess
SHORTAGE_PERCENT = 20  # Percentage of items with shortage

# Store cities configuration
STORE_CITIES = {
    "Hanoi": {"lat_range": (20.9, 21.1), "lon_range": (105.7, 105.9), "count": 7},
    "Da Nang": {"lat_range": (16.0, 16.1), "lon_range": (108.2, 108.3), "count": 5},
    "Ho Chi Minh City": {
        "lat_range": (10.7, 10.9),
        "lon_range": (106.6, 106.8),
        "count": 8,
    },
}

# Product categories
PRODUCT_CATEGORIES = ["Electronics", "Clothing", "Home Goods", "Food", "Beauty"]

# ============================================================================
# OPTIMIZATION ALGORITHMS
# ============================================================================

# Genetic Algorithm settings
GA_POPULATION_SIZE = 50
GA_GENERATIONS = 50
GA_CROSSOVER_PROB = 0.6
GA_MUTATION_PROB = 0.3
GA_TOURNAMENT_SIZE = 3

# Rule-Based optimization settings
DISTANCE_WEIGHT = 0.4
EXCESS_WEIGHT = 0.3
NEEDED_WEIGHT = 0.3
MAX_TRANSFER_DISTANCE_KM = 500
BASE_TRANSPORT_COST_PER_KM = 100


# ============================================================================
# ENVIRONMENT SETTINGS
# ============================================================================

# Environment configurations
ENVIRONMENTS = {
    "development": {
        "debug": True,
        "ga_population": 30,
        "ga_generations": 20,
        "num_products": 20,
        "sales_days": 90,
    },
    "production": {
        "debug": False,
        "ga_population": GA_POPULATION_SIZE,
        "ga_generations": GA_GENERATIONS,
        "num_products": NUM_PRODUCTS,
        "sales_days": SALES_DAYS,
    },
    "testing": {
        "debug": True,
        "ga_population": 10,
        "ga_generations": 5,
        "num_products": 10,
        "sales_days": 30,
    },
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_data_file_path(filename: str, data_dir: str = DATA_DIR) -> Path:
    """Get the full path for a data file."""
    return Path(data_dir) / filename


def get_results_file_path(filename: str, results_dir: str = RESULTS_DIR) -> Path:
    """Get the full path for a results file."""
    return Path(results_dir) / filename


def create_directories(base_path: Optional[Path] = None) -> dict:
    """Create all necessary project directories."""
    if base_path is None:
        base_path = Path.cwd()

    directories = {
        "data": base_path / DATA_DIR,
        "results": base_path / RESULTS_DIR,
        "visualizations": base_path / VISUALIZATIONS_DIR,
        "logs": base_path / LOGS_DIR,
    }

    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    return directories


def get_environment_config(env: str = "development") -> dict:
    """Get configuration for specific environment."""
    return ENVIRONMENTS.get(env.lower(), ENVIRONMENTS["development"])


def get_ga_config() -> dict:
    """Get genetic algorithm configuration as dictionary."""
    return {
        "population_size": GA_POPULATION_SIZE,
        "num_generations": GA_GENERATIONS,
        "crossover_prob": GA_CROSSOVER_PROB,
        "mutation_prob": GA_MUTATION_PROB,
        "tournament_size": GA_TOURNAMENT_SIZE,
    }
