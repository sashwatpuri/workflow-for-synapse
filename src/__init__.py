"""
Smart Farming Prediction System
Source Package

This package contains all core modules for the smart farming system.
"""

__version__ = "1.0.0"
__author__ = "Smart Farming Team"

# Import main classes for easy access
from .oop_architecture import (
    Farm, Crop, Sensor, WeatherData, IrrigationController,
    GeoLocation, CropType, IrrigationType, DiseaseStatus, Region
)

from .probability_models import (
    MarkovChainMoisture, PoissonRainfallModel, GaussianSensorNoise,
    MonteCarloSimulator, BayesianIrrigationPredictor
)

from .scheduling_algorithms import (
    GreedyPriorityScheduler, HeapBasedScheduler,
    DynamicProgrammingWaterAllocator, ZoneBasedScheduler
)

from .data_preprocessing import DataPreprocessor
from .model_training import ModelTrainer
from .simulation import FarmSimulator, create_sample_farms

__all__ = [
    # OOP Architecture
    'Farm', 'Crop', 'Sensor', 'WeatherData', 'IrrigationController',
    'GeoLocation', 'CropType', 'IrrigationType', 'DiseaseStatus', 'Region',
    
    # Probability Models
    'MarkovChainMoisture', 'PoissonRainfallModel', 'GaussianSensorNoise',
    'MonteCarloSimulator', 'BayesianIrrigationPredictor',
    
    # Scheduling Algorithms
    'GreedyPriorityScheduler', 'HeapBasedScheduler',
    'DynamicProgrammingWaterAllocator', 'ZoneBasedScheduler',
    
    # Data & Training
    'DataPreprocessor', 'ModelTrainer',
    
    # Simulation
    'FarmSimulator', 'create_sample_farms'
]
