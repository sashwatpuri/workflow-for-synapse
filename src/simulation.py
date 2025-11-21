"""
Smart Farming Simulation Module

This module simulates:
1. Soil moisture dynamics
2. Plant water uptake
3. Rainfall events (Poisson process)
4. Irrigation effects
5. Complete farm ecosystem over time

Integrates all components:
- OOP architecture (Farm, Crop, Sensor, IrrigationController)
- Probability models (Markov, Poisson, Monte Carlo)
- Scheduling algorithms (Greedy, Heap-based)
- ML predictions
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Optional import for progress bar
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        return iterable

# Import our modules
from .oop_architecture import (
    Farm, Crop, Sensor, WeatherData, IrrigationController,
    GeoLocation, CropType, IrrigationType, DiseaseStatus, Region,
    SensorReading
)
from .probability_models import (
    MarkovChainMoisture, PoissonRainfallModel, GaussianSensorNoise,
    MonteCarloSimulator
)
from .scheduling_algorithms import GreedyPriorityScheduler, HeapBasedScheduler


class FarmSimulator:
    """
    Main simulation class for smart farming system
    
    Simulates complete farm ecosystem including:
    - Soil moisture dynamics
    - Weather patterns
    - Crop growth
    - Irrigation decisions
    - Sensor readings
    """
    
    def __init__(self, farms: List[Farm], simulation_days: int = 30):
        """
        Initialize simulator
        
        Args:
            farms: List of Farm objects
            simulation_days: Number of days to simulate
        """
        self.farms = farms
        self.simulation_days = simulation_days
        self.current_time = datetime.now()
        
        # Simulation state
        self.moisture_history = {farm.farm_id: [] for farm in farms}
        self.irrigation_events = {farm.farm_id: [] for farm in farms}
        self.rainfall_events = []
        self.sensor_readings = {farm.farm_id: [] for farm in farms}
        
        # Models
        self.markov_model = MarkovChainMoisture(n_states=10)
        self.rainfall_model = PoissonRainfallModel()
        self.noise_model = GaussianSensorNoise(noise_std=0.5)
        
        # Scheduler
        self.scheduler = GreedyPriorityScheduler(
            total_water_budget=50000,  # 50,000 liters per day
            time_window_hours=24
        )
        
        # Statistics
        self.stats = {
            'total_water_used': 0.0,
            'total_irrigations': 0,
            'total_rainfall': 0.0,
            'avg_moisture': 0.0
        }
    
    def initialize_models(self):
        """Initialize probability models with historical data"""
        print("Initializing probability models...")
        
        # Generate synthetic historical data for model training
        # In real scenario, use actual historical data
        
        # Markov model for moisture
        moisture_sequence = [35.0]
        for _ in range(100):
            change = np.random.normal(-0.3, 1.5)
            new_moisture = moisture_sequence[-1] + change
            moisture_sequence.append(np.clip(new_moisture, 0, 100))
        
        self.markov_model.fit(moisture_sequence)
        
        # Poisson model for rainfall
        rainfall_data = []
        for _ in range(365):
            if np.random.random() < 0.25:  # 25% chance of rain
                rainfall_data.append(np.random.exponential(8))
            else:
                rainfall_data.append(0.0)
        
        self.rainfall_model.fit(rainfall_data, time_period_days=365)
        
        print("Models initialized successfully")
    
    def simulate_day(self, day: int):
        """
        Simulate one day of farm operations
        
        Args:
            day: Day number (0-indexed)
        """
        current_date = self.current_time + timedelta(days=day)
        
        # 1. Generate rainfall (Poisson process)
        n_rain_events = np.random.poisson(self.rainfall_model.lambda_rate)
        daily_rainfall = 0.0
        if n_rain_events > 0:
            rainfall_amounts = np.random.exponential(
                self.rainfall_model.mean_rainfall, n_rain_events
            )
            daily_rainfall = rainfall_amounts.sum()
            self.rainfall_events.append({
                'date': current_date,
                'amount_mm': daily_rainfall,
                'n_events': n_rain_events
            })
        
        self.stats['total_rainfall'] += daily_rainfall
        
        # 2. Update each farm
        for farm in self.farms:
            self._update_farm(farm, current_date, daily_rainfall)
        
        # 3. Make irrigation decisions
        self._schedule_irrigation(current_date)
    
    def _update_farm(self, farm: Farm, current_date: datetime, rainfall: float):
        """
        Update farm state for one day
        
        Args:
            farm: Farm object
            current_date: Current simulation date
            rainfall: Daily rainfall amount (mm)
        """
        # Get current moisture
        current_moisture = farm.get_current_moisture()
        if current_moisture == 0:
            current_moisture = 30.0  # Initialize if first day
        
        # Calculate evapotranspiration
        if farm.weather_data:
            latest_weather = farm.weather_data[-1]
            et0 = latest_weather.calculate_evapotranspiration()
            kc = farm.crop.get_crop_coefficient(current_date) if farm.crop else 1.0
            etc = kc * et0
        else:
            etc = 3.0  # Default ET
        
        # Calculate moisture change
        # Δθ = (P + I - ET - D) / soil_depth
        # P = rainfall, I = irrigation, ET = evapotranspiration, D = drainage
        
        # Rainfall contribution (mm to % moisture, assuming 1mm = 1% for top 100mm soil)
        rainfall_contribution = rainfall * 0.1
        
        # ET loss
        et_loss = etc * 0.15  # Convert mm to % moisture
        
        # Drainage (excess water drains)
        drainage = 0.0
        if current_moisture > 60:
            drainage = (current_moisture - 60) * 0.2
        
        # Plant water uptake (additional to ET)
        if farm.crop:
            water_stress = farm.crop.calculate_water_stress(current_moisture)
            uptake = farm.crop.water_requirement * 0.1 * (1 - water_stress)
        else:
            uptake = 0.5
        
        # Update moisture
        moisture_change = rainfall_contribution - et_loss - drainage - uptake
        
        # Add Markov uncertainty
        markov_pred = self.markov_model.predict_next(current_moisture)
        markov_noise = np.random.normal(0, markov_pred.std * 0.1)
        
        new_moisture = current_moisture + moisture_change + markov_noise
        new_moisture = np.clip(new_moisture, 0, 100)
        
        # Add sensor reading with noise
        noisy_moisture = self.noise_model.add_noise(np.array([new_moisture]))[0]
        
        reading = SensorReading(
            sensor_id=farm.sensor.sensor_id,
            timestamp=current_date,
            soil_moisture=noisy_moisture,
            soil_ph=6.5 + np.random.normal(0, 0.2),
            temperature=25 + np.random.normal(0, 3),
            humidity=65 + np.random.normal(0, 10),
            rainfall=rainfall
        )
        
        farm.sensor.add_reading(reading)
        
        # Update weather data
        weather = WeatherData(
            timestamp=current_date,
            temperature_c=reading.temperature,
            humidity_percent=reading.humidity,
            rainfall_mm=rainfall,
            sunlight_hours=8 + np.random.normal(0, 1.5)
        )
        farm.add_weather_data(weather)
        
        # Store history
        self.moisture_history[farm.farm_id].append({
            'date': current_date,
            'moisture': new_moisture,
            'noisy_moisture': noisy_moisture,
            'rainfall': rainfall,
            'et': etc,
            'irrigation': 0.0  # Will be updated if irrigation occurs
        })
    
    def _schedule_irrigation(self, current_date: datetime):
        """
        Schedule and execute irrigation for all farms
        
        Args:
            current_date: Current simulation date
        """
        # Prepare farm data for scheduler
        farms_data = []
        for farm in self.farms:
            if not farm.crop:
                continue
            
            current_moisture = farm.get_current_moisture()
            
            farm_data = {
                'farm_id': farm.farm_id,
                'current_moisture': current_moisture,
                'optimal_moisture': farm.crop.optimal_moisture_range[1],
                'growth_stage': farm.crop.get_growth_stage(current_date),
                'disease_status': farm.disease_status.value,
                'hours_since_irrigation': 24,  # Simplified
                'water_needed': 1500,  # liters
                'duration_minutes': 30
            }
            farms_data.append(farm_data)
        
        # Schedule irrigation
        schedule = self.scheduler.schedule_greedy(farms_data)
        
        # Execute irrigation
        for entry in schedule:
            farm = next(f for f in self.farms if f.farm_id == entry.farm_id)
            
            # Activate irrigation
            water_delivered = farm.irrigation_controller.activate(
                duration_minutes=(entry.end_time - entry.start_time).total_seconds() / 60
            )
            
            # Update moisture
            moisture_increase = (entry.water_allocated / 10000) * \
                              farm.irrigation_controller.efficiency
            
            # Update last moisture record
            if self.moisture_history[farm.farm_id]:
                self.moisture_history[farm.farm_id][-1]['irrigation'] = moisture_increase
                
                # Update sensor reading
                latest_reading = farm.sensor.get_latest_reading()
                if latest_reading:
                    latest_reading.soil_moisture += moisture_increase
                    latest_reading.soil_moisture = np.clip(latest_reading.soil_moisture, 0, 100)
            
            # Record irrigation event
            self.irrigation_events[farm.farm_id].append({
                'date': current_date,
                'water_used': entry.water_allocated,
                'duration_minutes': (entry.end_time - entry.start_time).total_seconds() / 60,
                'priority': entry.priority_score
            })
            
            self.stats['total_water_used'] += entry.water_allocated
            self.stats['total_irrigations'] += 1
    
    def run_simulation(self):
        """Run complete simulation"""
        print("\n" + "="*70)
        print("SMART FARMING SIMULATION")
        print("="*70)
        print(f"Farms: {len(self.farms)}")
        print(f"Simulation period: {self.simulation_days} days")
        print(f"Start date: {self.current_time.strftime('%Y-%m-%d')}")
        
        # Initialize models
        self.initialize_models()
        
        # Run daily simulation
        print("\nRunning simulation...")
        for day in tqdm(range(self.simulation_days), desc="Simulating days"):
            self.simulate_day(day)
        
        # Calculate final statistics
        self._calculate_statistics()
        
        print("\n" + "="*70)
        print("SIMULATION COMPLETE")
        print("="*70)
        self.print_statistics()
    
    def _calculate_statistics(self):
        """Calculate final simulation statistics"""
        all_moisture = []
        for farm_id, history in self.moisture_history.items():
            moisture_values = [h['moisture'] for h in history]
            all_moisture.extend(moisture_values)
        
        self.stats['avg_moisture'] = np.mean(all_moisture) if all_moisture else 0
        self.stats['min_moisture'] = np.min(all_moisture) if all_moisture else 0
        self.stats['max_moisture'] = np.max(all_moisture) if all_moisture else 0
        
        # Water efficiency
        self.stats['water_per_farm_per_day'] = self.stats['total_water_used'] / \
                                                (len(self.farms) * self.simulation_days)
    
    def print_statistics(self):
        """Print simulation statistics"""
        print(f"\nSimulation Statistics:")
        print(f"  Total water used: {self.stats['total_water_used']:.2f} liters")
        print(f"  Total irrigations: {self.stats['total_irrigations']}")
        print(f"  Total rainfall: {self.stats['total_rainfall']:.2f} mm")
        print(f"  Average moisture: {self.stats['avg_moisture']:.2f}%")
        print(f"  Moisture range: [{self.stats['min_moisture']:.2f}%, {self.stats['max_moisture']:.2f}%]")
        print(f"  Water per farm per day: {self.stats['water_per_farm_per_day']:.2f} liters")
    
    def plot_results(self, farm_ids: List[str] = None, output_dir: str = 'results'):
        """
        Plot simulation results
        
        Args:
            farm_ids: List of farm IDs to plot (None = plot all)
            output_dir: Output directory for plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if farm_ids is None:
            farm_ids = list(self.moisture_history.keys())[:5]  # Plot first 5
        
        # Plot 1: Moisture over time
        fig, axes = plt.subplots(len(farm_ids), 1, figsize=(14, 4*len(farm_ids)))
        if len(farm_ids) == 1:
            axes = [axes]
        
        for idx, farm_id in enumerate(farm_ids):
            history = self.moisture_history[farm_id]
            dates = [h['date'] for h in history]
            moisture = [h['moisture'] for h in history]
            noisy_moisture = [h['noisy_moisture'] for h in history]
            rainfall = [h['rainfall'] for h in history]
            irrigation = [h['irrigation'] for h in history]
            
            ax = axes[idx]
            
            # Plot moisture
            ax.plot(dates, moisture, label='True Moisture', linewidth=2, color='blue')
            ax.plot(dates, noisy_moisture, label='Sensor Reading', linewidth=1, 
                   alpha=0.6, color='lightblue', linestyle='--')
            
            # Plot rainfall events
            rain_dates = [dates[i] for i, r in enumerate(rainfall) if r > 0]
            rain_amounts = [r for r in rainfall if r > 0]
            if rain_dates:
                ax.scatter(rain_dates, [80]*len(rain_dates), s=[r*10 for r in rain_amounts],
                          alpha=0.5, color='cyan', label='Rainfall', marker='v')
            
            # Plot irrigation events
            irr_dates = [dates[i] for i, irr in enumerate(irrigation) if irr > 0]
            if irr_dates:
                ax.scatter(irr_dates, [85]*len(irr_dates), s=100,
                          alpha=0.7, color='green', label='Irrigation', marker='^')
            
            # Optimal range
            farm = next(f for f in self.farms if f.farm_id == farm_id)
            if farm.crop:
                min_m, max_m = farm.crop.optimal_moisture_range
                ax.axhspan(min_m, max_m, alpha=0.2, color='green', label='Optimal Range')
            
            ax.set_ylabel('Soil Moisture (%)', fontsize=11)
            ax.set_title(f'Farm: {farm_id}', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(alpha=0.3)
        
        axes[-1].set_xlabel('Date', fontsize=11)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/moisture_simulation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nPlots saved to {output_dir}/")


def create_sample_farms(n_farms: int = 10) -> List[Farm]:
    """Create sample farms for simulation"""
    farms = []
    
    crop_types = [CropType.WHEAT, CropType.RICE, CropType.MAIZE, CropType.COTTON, CropType.SOYBEAN]
    irrigation_types = [IrrigationType.DRIP, IrrigationType.SPRINKLER, IrrigationType.MANUAL]
    regions = [Region.NORTH_INDIA, Region.SOUTH_INDIA, Region.CENTRAL_USA]
    
    for i in range(n_farms):
        # Create farm
        location = GeoLocation(
            latitude=20 + np.random.uniform(0, 15),
            longitude=70 + np.random.uniform(0, 20)
        )
        farm = Farm(f'FARM{i:03d}', np.random.choice(regions), location)
        
        # Add crop
        sowing_date = datetime.now() - timedelta(days=np.random.randint(30, 90))
        harvest_date = sowing_date + timedelta(days=np.random.randint(90, 150))
        crop = Crop(np.random.choice(crop_types), sowing_date, harvest_date)
        farm.set_crop(crop)
        
        # Add sensor
        sensor = Sensor(f'SENS{i:03d}', location)
        # Initialize with a reading
        initial_reading = SensorReading(
            sensor_id=f'SENS{i:03d}',
            timestamp=datetime.now(),
            soil_moisture=np.random.uniform(25, 40),
            soil_ph=np.random.uniform(6.0, 7.5),
            temperature=np.random.uniform(20, 30),
            humidity=np.random.uniform(50, 80),
            rainfall=0.0
        )
        sensor.add_reading(initial_reading)
        farm.set_sensor(sensor)
        
        # Add irrigation controller
        controller = IrrigationController(np.random.choice(irrigation_types))
        farm.set_irrigation(controller)
        
        # Set disease status
        farm.disease_status = np.random.choice(list(DiseaseStatus))
        farm.ndvi_index = np.random.uniform(0.4, 0.9)
        
        farms.append(farm)
    
    return farms


if __name__ == "__main__":
    # Create sample farms
    print("Creating sample farms...")
    farms = create_sample_farms(n_farms=10)
    
    # Run simulation
    simulator = FarmSimulator(farms, simulation_days=30)
    simulator.run_simulation()
    
    # Plot results
    simulator.plot_results()
    
    print("\n=== Simulation Demo Complete ===")
