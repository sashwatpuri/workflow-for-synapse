"""
Object-Oriented Programming (OOPS) Architecture
Smart Farming Prediction System

This module implements the complete OOP architecture with:
- Encapsulation: Data hiding and access control
- Inheritance: Base classes and derived classes
- Polymorphism: Method overriding and interfaces
- Abstraction: Abstract base classes and interfaces

Classes:
- Farm: Represents a farm with all attributes
- Crop: Represents crop-specific data
- Sensor: IoT sensor data management
- WeatherData: Environmental conditions
- IrrigationController: Irrigation system control
- Scheduler: Irrigation scheduling logic
- ModelPredictor: ML model predictions
- DataProcessor: Data preprocessing and feature engineering
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum


# ==================== ENUMERATIONS ====================

class CropType(Enum):
    """Enumeration for crop types"""
    WHEAT = "Wheat"
    RICE = "Rice"
    MAIZE = "Maize"
    COTTON = "Cotton"
    SOYBEAN = "Soybean"


class IrrigationType(Enum):
    """Enumeration for irrigation types"""
    DRIP = "Drip"
    SPRINKLER = "Sprinkler"
    MANUAL = "Manual"
    NONE = "None"


class FertilizerType(Enum):
    """Enumeration for fertilizer types"""
    ORGANIC = "Organic"
    INORGANIC = "Inorganic"
    MIXED = "Mixed"


class DiseaseStatus(Enum):
    """Enumeration for crop disease status"""
    NONE = "None"
    MILD = "Mild"
    MODERATE = "Moderate"
    SEVERE = "Severe"


class Region(Enum):
    """Enumeration for regions"""
    NORTH_INDIA = "North India"
    SOUTH_INDIA = "South India"
    CENTRAL_USA = "Central USA"
    SOUTH_USA = "South USA"
    EAST_AFRICA = "East Africa"


# ==================== DATA CLASSES ====================

@dataclass
class SensorReading:
    """Represents a single sensor reading"""
    sensor_id: str
    timestamp: datetime
    soil_moisture: float
    soil_ph: float
    temperature: float
    humidity: float
    rainfall: float = 0.0
    noise_level: float = 0.0  # Gaussian noise
    
    def add_noise(self, std_dev: float = 0.5):
        """Add Gaussian noise to simulate sensor uncertainty"""
        self.noise_level = np.random.normal(0, std_dev)
        self.soil_moisture += self.noise_level
        self.soil_moisture = max(0, min(100, self.soil_moisture))  # Clamp to [0, 100]


@dataclass
class WeatherData:
    """Represents weather conditions"""
    timestamp: datetime
    temperature_c: float
    humidity_percent: float
    rainfall_mm: float
    sunlight_hours: float
    
    def calculate_evapotranspiration(self) -> float:
        """
        Calculate reference evapotranspiration (ET0) using simplified Penman-Monteith
        ET0 ≈ 0.0023 * (T_mean + 17.8) * sqrt(T_max - T_min) * Ra
        Simplified proxy: ET0 ≈ 0.1 * temp * (1 - humidity/100) * sunlight
        """
        et0 = 0.1 * self.temperature_c * (1 - self.humidity_percent / 100) * self.sunlight_hours
        return max(0, et0)
    
    def predict_rain_probability(self) -> float:
        """
        Predict probability of rain based on humidity and temperature
        P(rain) = sigmoid(0.05 * humidity - 0.02 * temp - 2)
        """
        z = 0.05 * self.humidity_percent - 0.02 * self.temperature_c - 2
        return 1 / (1 + np.exp(-z))


@dataclass
class GeoLocation:
    """Represents geographical location"""
    latitude: float
    longitude: float
    
    def distance_to(self, other: 'GeoLocation') -> float:
        """Calculate Haversine distance in kilometers"""
        R = 6371  # Earth radius in km
        lat1, lon1 = np.radians(self.latitude), np.radians(self.longitude)
        lat2, lon2 = np.radians(other.latitude), np.radians(other.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c


# ==================== ABSTRACT BASE CLASSES ====================

class Predictable(ABC):
    """Abstract interface for predictable entities"""
    
    @abstractmethod
    def predict_next_state(self, hours_ahead: int) -> Dict:
        """Predict future state"""
        pass


class Schedulable(ABC):
    """Abstract interface for schedulable entities"""
    
    @abstractmethod
    def calculate_priority(self) -> float:
        """Calculate scheduling priority"""
        pass


# ==================== CORE CLASSES ====================

class Sensor:
    """
    Represents an IoT sensor with data collection and noise modeling
    
    Attributes:
        sensor_id: Unique sensor identifier
        location: Geographical location
        readings: Historical sensor readings
        calibration_offset: Sensor calibration offset
    """
    
    def __init__(self, sensor_id: str, location: GeoLocation):
        self.sensor_id = sensor_id
        self.location = location
        self.readings: List[SensorReading] = []
        self.calibration_offset = 0.0
        self._noise_std = 0.5  # Standard deviation for Gaussian noise
    
    def add_reading(self, reading: SensorReading):
        """Add a new sensor reading with noise simulation"""
        reading.add_noise(self._noise_std)
        self.readings.append(reading)
    
    def get_latest_reading(self) -> Optional[SensorReading]:
        """Get the most recent sensor reading"""
        return self.readings[-1] if self.readings else None
    
    def get_average_moisture(self, hours: int = 24) -> float:
        """Calculate average soil moisture over last N hours"""
        if not self.readings:
            return 0.0
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_readings = [r for r in self.readings if r.timestamp >= cutoff_time]
        
        if not recent_readings:
            return self.readings[-1].soil_moisture
        
        return np.mean([r.soil_moisture for r in recent_readings])
    
    def calibrate(self, reference_value: float, measured_value: float):
        """Calibrate sensor based on reference measurement"""
        self.calibration_offset = reference_value - measured_value
    
    def __repr__(self):
        return f"Sensor(id={self.sensor_id}, readings={len(self.readings)})"


class Crop:
    """
    Represents a crop with growth characteristics and requirements
    
    Attributes:
        crop_type: Type of crop
        sowing_date: Date when crop was sown
        harvest_date: Expected harvest date
        water_requirement: Daily water requirement (mm)
        optimal_moisture_range: Optimal soil moisture range (min, max)
    """
    
    # Crop-specific water requirements (mm/day)
    WATER_REQUIREMENTS = {
        CropType.WHEAT: 4.5,
        CropType.RICE: 7.5,
        CropType.MAIZE: 5.0,
        CropType.COTTON: 5.5,
        CropType.SOYBEAN: 5.0
    }
    
    # Optimal soil moisture ranges (%)
    MOISTURE_RANGES = {
        CropType.WHEAT: (25, 40),
        CropType.RICE: (40, 60),
        CropType.MAIZE: (30, 45),
        CropType.COTTON: (25, 40),
        CropType.SOYBEAN: (30, 45)
    }
    
    def __init__(self, crop_type: CropType, sowing_date: datetime, harvest_date: datetime):
        self.crop_type = crop_type
        self.sowing_date = sowing_date
        self.harvest_date = harvest_date
        self.water_requirement = self.WATER_REQUIREMENTS[crop_type]
        self.optimal_moisture_range = self.MOISTURE_RANGES[crop_type]
    
    def get_growth_stage(self, current_date: datetime) -> str:
        """
        Determine current growth stage
        Stages: Initial (0-25%), Development (25-50%), Mid (50-75%), Late (75-100%)
        """
        total_days = (self.harvest_date - self.sowing_date).days
        elapsed_days = (current_date - self.sowing_date).days
        progress = (elapsed_days / total_days) * 100
        
        if progress < 25:
            return "Initial"
        elif progress < 50:
            return "Development"
        elif progress < 75:
            return "Mid"
        else:
            return "Late"
    
    def get_crop_coefficient(self, current_date: datetime) -> float:
        """
        Get crop coefficient (Kc) based on growth stage
        Used for ET calculation: ETc = Kc * ET0
        """
        stage = self.get_growth_stage(current_date)
        kc_values = {
            "Initial": 0.4,
            "Development": 0.7,
            "Mid": 1.15,
            "Late": 0.8
        }
        return kc_values[stage]
    
    def is_moisture_optimal(self, current_moisture: float) -> bool:
        """Check if current moisture is in optimal range"""
        min_moisture, max_moisture = self.optimal_moisture_range
        return min_moisture <= current_moisture <= max_moisture
    
    def calculate_water_stress(self, current_moisture: float) -> float:
        """
        Calculate water stress factor (0 = no stress, 1 = severe stress)
        """
        min_moisture, max_moisture = self.optimal_moisture_range
        
        if current_moisture >= min_moisture:
            return 0.0
        else:
            # Linear stress increase as moisture drops below minimum
            stress = (min_moisture - current_moisture) / min_moisture
            return min(1.0, stress)
    
    def __repr__(self):
        return f"Crop(type={self.crop_type.value}, stage={self.get_growth_stage(datetime.now())})"


class IrrigationController:
    """
    Controls irrigation system with actuation logic
    
    Attributes:
        irrigation_type: Type of irrigation system
        flow_rate: Water flow rate (liters/minute)
        efficiency: System efficiency (0-1)
        is_active: Current activation status
    """
    
    # Flow rates for different irrigation types (L/min)
    FLOW_RATES = {
        IrrigationType.DRIP: 4.0,
        IrrigationType.SPRINKLER: 15.0,
        IrrigationType.MANUAL: 10.0,
        IrrigationType.NONE: 0.0
    }
    
    # System efficiencies
    EFFICIENCIES = {
        IrrigationType.DRIP: 0.90,
        IrrigationType.SPRINKLER: 0.75,
        IrrigationType.MANUAL: 0.60,
        IrrigationType.NONE: 0.0
    }
    
    def __init__(self, irrigation_type: IrrigationType):
        self.irrigation_type = irrigation_type
        self.flow_rate = self.FLOW_RATES[irrigation_type]
        self.efficiency = self.EFFICIENCIES[irrigation_type]
        self.is_active = False
        self.total_water_used = 0.0  # liters
        self.activation_history: List[Tuple[datetime, int]] = []  # (timestamp, duration_minutes)
    
    def activate(self, duration_minutes: int) -> float:
        """
        Activate irrigation for specified duration
        Returns: Effective water delivered (liters)
        """
        if self.irrigation_type == IrrigationType.NONE:
            return 0.0
        
        self.is_active = True
        water_delivered = self.flow_rate * duration_minutes * self.efficiency
        self.total_water_used += water_delivered
        self.activation_history.append((datetime.now(), duration_minutes))
        
        return water_delivered
    
    def deactivate(self):
        """Deactivate irrigation system"""
        self.is_active = False
    
    def calculate_water_savings(self, baseline_type: IrrigationType = IrrigationType.MANUAL) -> float:
        """
        Calculate water savings compared to baseline irrigation type
        Returns: Percentage of water saved
        """
        if not self.activation_history:
            return 0.0
        
        total_minutes = sum(duration for _, duration in self.activation_history)
        baseline_flow = self.FLOW_RATES[baseline_type]
        baseline_efficiency = self.EFFICIENCIES[baseline_type]
        
        baseline_water = baseline_flow * total_minutes * baseline_efficiency
        actual_water = self.total_water_used
        
        if baseline_water == 0:
            return 0.0
        
        savings = ((baseline_water - actual_water) / baseline_water) * 100
        return max(0.0, savings)
    
    def __repr__(self):
        status = "ACTIVE" if self.is_active else "INACTIVE"
        return f"IrrigationController(type={self.irrigation_type.value}, status={status})"


class Farm(Predictable, Schedulable):
    """
    Main Farm class representing a complete farm entity
    
    Attributes:
        farm_id: Unique farm identifier
        region: Geographical region
        location: GPS coordinates
        crop: Crop being grown
        sensor: IoT sensor
        irrigation_controller: Irrigation system
        ndvi_index: Normalized Difference Vegetation Index
        disease_status: Current crop disease status
    """
    
    def __init__(self, farm_id: str, region: Region, location: GeoLocation):
        self.farm_id = farm_id
        self.region = region
        self.location = location
        self.crop: Optional[Crop] = None
        self.sensor: Optional[Sensor] = None
        self.irrigation_controller: Optional[IrrigationController] = None
        self.weather_data: List[WeatherData] = []
        self.ndvi_index = 0.0
        self.disease_status = DiseaseStatus.NONE
        self.fertilizer_type: Optional[FertilizerType] = None
        self.pesticide_usage_ml = 0.0
        self.yield_kg_per_hectare = 0.0
        
        # Markov chain state for soil moisture
        self.moisture_states: List[float] = []
        self.transition_matrix: Optional[np.ndarray] = None
    
    def set_crop(self, crop: Crop):
        """Set the crop for this farm"""
        self.crop = crop
    
    def set_sensor(self, sensor: Sensor):
        """Set the IoT sensor for this farm"""
        self.sensor = sensor
    
    def set_irrigation(self, controller: IrrigationController):
        """Set the irrigation controller"""
        self.irrigation_controller = controller
    
    def add_weather_data(self, weather: WeatherData):
        """Add weather data"""
        self.weather_data.append(weather)
    
    def get_current_moisture(self) -> float:
        """Get current soil moisture from sensor"""
        if self.sensor:
            latest = self.sensor.get_latest_reading()
            return latest.soil_moisture if latest else 0.0
        return 0.0
    
    def calculate_irrigation_need(self) -> float:
        """
        Calculate irrigation need based on multiple factors
        Returns: Probability of irrigation needed (0-1)
        """
        if not self.crop or not self.sensor:
            return 0.0
        
        current_moisture = self.get_current_moisture()
        min_moisture, _ = self.crop.optimal_moisture_range
        
        # Factor 1: Moisture deficit
        moisture_deficit = max(0, min_moisture - current_moisture) / min_moisture
        
        # Factor 2: Weather (ET and rain probability)
        et_factor = 0.0
        rain_prob = 0.0
        if self.weather_data:
            latest_weather = self.weather_data[-1]
            et0 = latest_weather.calculate_evapotranspiration()
            kc = self.crop.get_crop_coefficient(datetime.now())
            etc = kc * et0
            et_factor = min(1.0, etc / 10.0)  # Normalize
            rain_prob = latest_weather.predict_rain_probability()
        
        # Factor 3: Crop stress
        stress_factor = self.crop.calculate_water_stress(current_moisture)
        
        # Combined probability using weighted average
        irrigation_prob = (
            0.4 * moisture_deficit +
            0.3 * et_factor +
            0.2 * stress_factor +
            0.1 * (1 - rain_prob)  # Less need if rain likely
        )
        
        return min(1.0, irrigation_prob)
    
    def predict_next_state(self, hours_ahead: int = 6) -> Dict:
        """
        Predict future soil moisture using Markov process
        
        Markov assumption: P(M_t+1 | M_t, M_t-1, ...) = P(M_t+1 | M_t)
        """
        current_moisture = self.get_current_moisture()
        
        # Simple Markov model: next_moisture = current_moisture - ET + rainfall - drainage
        if self.weather_data and self.crop:
            latest_weather = self.weather_data[-1]
            et0 = latest_weather.calculate_evapotranspiration()
            kc = self.crop.get_crop_coefficient(datetime.now())
            etc = kc * et0
            
            # Moisture loss per hour
            moisture_loss_per_hour = etc / 24  # Convert daily ET to hourly
            
            # Expected rainfall (using Poisson process)
            rain_prob = latest_weather.predict_rain_probability()
            expected_rainfall = rain_prob * latest_weather.rainfall_mm
            
            # Predict moisture
            predicted_moisture = current_moisture - (moisture_loss_per_hour * hours_ahead)
            predicted_moisture += expected_rainfall * 0.1  # Rainfall contribution
            predicted_moisture = max(0, min(100, predicted_moisture))
            
            return {
                'predicted_moisture': predicted_moisture,
                'current_moisture': current_moisture,
                'moisture_change': predicted_moisture - current_moisture,
                'hours_ahead': hours_ahead,
                'irrigation_needed': predicted_moisture < self.crop.optimal_moisture_range[0]
            }
        
        return {
            'predicted_moisture': current_moisture,
            'current_moisture': current_moisture,
            'moisture_change': 0.0,
            'hours_ahead': hours_ahead,
            'irrigation_needed': False
        }
    
    def calculate_priority(self) -> float:
        """
        Calculate scheduling priority (higher = more urgent)
        
        Priority based on:
        1. Water stress level
        2. Crop growth stage importance
        3. Disease severity
        4. Time since last irrigation
        """
        priority = 0.0
        
        if not self.crop:
            return priority
        
        # Factor 1: Water stress (0-40 points)
        current_moisture = self.get_current_moisture()
        stress = self.crop.calculate_water_stress(current_moisture)
        priority += stress * 40
        
        # Factor 2: Growth stage (0-30 points)
        stage = self.crop.get_growth_stage(datetime.now())
        stage_importance = {
            "Initial": 0.6,
            "Development": 0.8,
            "Mid": 1.0,  # Most critical
            "Late": 0.7
        }
        priority += stage_importance[stage] * 30
        
        # Factor 3: Disease severity (0-20 points)
        disease_impact = {
            DiseaseStatus.NONE: 0.0,
            DiseaseStatus.MILD: 0.3,
            DiseaseStatus.MODERATE: 0.6,
            DiseaseStatus.SEVERE: 1.0
        }
        priority += disease_impact[self.disease_status] * 20
        
        # Factor 4: Time since last irrigation (0-10 points)
        if self.irrigation_controller and self.irrigation_controller.activation_history:
            last_irrigation_time, _ = self.irrigation_controller.activation_history[-1]
            hours_since = (datetime.now() - last_irrigation_time).total_seconds() / 3600
            time_factor = min(1.0, hours_since / 48)  # Normalize to 48 hours
            priority += time_factor * 10
        else:
            priority += 10  # Never irrigated = high priority
        
        return priority
    
    def __repr__(self):
        return f"Farm(id={self.farm_id}, region={self.region.value}, crop={self.crop})"


# ==================== UTILITY CLASSES ====================

class DataProcessor:
    """
    Handles data preprocessing and feature engineering
    """
    
    @staticmethod
    def load_farm_data(filepath: str) -> pd.DataFrame:
        """Load farm data from CSV"""
        df = pd.read_csv(filepath)
        return df
    
    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for ML models
        
        Features:
        - Moisture trends
        - Rainfall windows
        - Sunlight exposure
        - ET proxy
        - NDVI-based health score
        - Growth stage
        """
        df = df.copy()
        
        # Convert dates
        df['sowing_date'] = pd.to_datetime(df['sowing_date'], format='%d-%m-%Y')
        df['harvest_date'] = pd.to_datetime(df['harvest_date'], format='%d-%m-%Y')
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y')
        
        # Growth stage progress
        df['days_since_sowing'] = (df['timestamp'] - df['sowing_date']).dt.days
        df['growth_progress'] = df['days_since_sowing'] / df['total_days']
        
        # Growth stage categorical
        df['growth_stage'] = pd.cut(df['growth_progress'], 
                                     bins=[0, 0.25, 0.5, 0.75, 1.0],
                                     labels=['Initial', 'Development', 'Mid', 'Late'])
        
        # ET proxy (simplified)
        df['et_proxy'] = 0.1 * df['temperature_C'] * (1 - df['humidity_%'] / 100) * df['sunlight_hours']
        
        # NDVI health score (normalized)
        df['ndvi_health_score'] = df['NDVI_index'] * 100
        
        # Moisture deficit
        crop_moisture_min = {
            'Wheat': 25, 'Rice': 40, 'Maize': 30, 'Cotton': 25, 'Soybean': 30
        }
        df['moisture_min'] = df['crop_type'].map(crop_moisture_min)
        df['moisture_deficit'] = df['moisture_min'] - df['soil_moisture_%']
        df['moisture_deficit'] = df['moisture_deficit'].clip(lower=0)
        
        # Rainfall windows (7-day, 14-day moving averages would go here)
        # For now, using current rainfall
        df['rainfall_7day'] = df['rainfall_mm']  # Placeholder
        
        # Water stress indicator
        df['water_stress'] = (df['moisture_deficit'] / df['moisture_min']).clip(0, 1)
        
        # Irrigation need (binary target)
        df['irrigation_needed'] = (df['soil_moisture_%'] < df['moisture_min']).astype(int)
        
        return df
    
    @staticmethod
    def create_farm_objects(df: pd.DataFrame) -> List[Farm]:
        """Create Farm objects from DataFrame"""
        farms = []
        
        for _, row in df.iterrows():
            # Create location
            location = GeoLocation(row['latitude'], row['longitude'])
            
            # Create farm
            region_map = {
                'North India': Region.NORTH_INDIA,
                'South India': Region.SOUTH_INDIA,
                'Central USA': Region.CENTRAL_USA,
                'South USA': Region.SOUTH_USA,
                'East Africa': Region.EAST_AFRICA
            }
            region = region_map[row['region']]
            farm = Farm(row['farm_id'], region, location)
            
            # Create and set crop
            crop_type_map = {
                'Wheat': CropType.WHEAT,
                'Rice': CropType.RICE,
                'Maize': CropType.MAIZE,
                'Cotton': CropType.COTTON,
                'Soybean': CropType.SOYBEAN
            }
            crop_type = crop_type_map[row['crop_type']]
            crop = Crop(crop_type, row['sowing_date'], row['harvest_date'])
            farm.set_crop(crop)
            
            # Create and set sensor
            sensor = Sensor(row['sensor_id'], location)
            reading = SensorReading(
                sensor_id=row['sensor_id'],
                timestamp=row['timestamp'],
                soil_moisture=row['soil_moisture_%'],
                soil_ph=row['soil_pH'],
                temperature=row['temperature_C'],
                humidity=row['humidity_%'],
                rainfall=row['rainfall_mm']
            )
            sensor.add_reading(reading)
            farm.set_sensor(sensor)
            
            # Create and set irrigation controller
            irrigation_type_map = {
                'Drip': IrrigationType.DRIP,
                'Sprinkler': IrrigationType.SPRINKLER,
                'Manual': IrrigationType.MANUAL,
                'None': IrrigationType.NONE
            }
            irrigation_type = irrigation_type_map[row['irrigation_type']]
            controller = IrrigationController(irrigation_type)
            farm.set_irrigation(controller)
            
            # Add weather data
            weather = WeatherData(
                timestamp=row['timestamp'],
                temperature_c=row['temperature_C'],
                humidity_percent=row['humidity_%'],
                rainfall_mm=row['rainfall_mm'],
                sunlight_hours=row['sunlight_hours']
            )
            farm.add_weather_data(weather)
            
            # Set other attributes
            farm.ndvi_index = row['NDVI_index']
            disease_map = {
                'None': DiseaseStatus.NONE,
                'Mild': DiseaseStatus.MILD,
                'Moderate': DiseaseStatus.MODERATE,
                'Severe': DiseaseStatus.SEVERE
            }
            farm.disease_status = disease_map[row['crop_disease_status']]
            farm.yield_kg_per_hectare = row['yield_kg_per_hectare']
            
            farms.append(farm)
        
        return farms


if __name__ == "__main__":
    # Example usage
    print("=== Smart Farming OOP Architecture ===\n")
    
    # Create a sample farm
    location = GeoLocation(28.6139, 77.2090)  # Delhi
    farm = Farm("FARM001", Region.NORTH_INDIA, location)
    
    # Create and set crop
    sowing = datetime(2024, 1, 15)
    harvest = datetime(2024, 5, 15)
    crop = Crop(CropType.WHEAT, sowing, harvest)
    farm.set_crop(crop)
    
    # Create and set sensor
    sensor = Sensor("SENS001", location)
    reading = SensorReading(
        sensor_id="SENS001",
        timestamp=datetime.now(),
        soil_moisture=22.5,
        soil_ph=6.5,
        temperature=25.0,
        humidity=65.0,
        rainfall=5.0
    )
    sensor.add_reading(reading)
    farm.set_sensor(sensor)
    
    # Create and set irrigation
    controller = IrrigationController(IrrigationType.DRIP)
    farm.set_irrigation(controller)
    
    # Add weather data
    weather = WeatherData(
        timestamp=datetime.now(),
        temperature_c=25.0,
        humidity_percent=65.0,
        rainfall_mm=5.0,
        sunlight_hours=8.5
    )
    farm.add_weather_data(weather)
    
    # Test methods
    print(f"Farm: {farm}")
    print(f"Current Moisture: {farm.get_current_moisture():.2f}%")
    print(f"Irrigation Need Probability: {farm.calculate_irrigation_need():.2f}")
    print(f"Priority Score: {farm.calculate_priority():.2f}")
    
    # Predict future state
    prediction = farm.predict_next_state(hours_ahead=6)
    print(f"\nPrediction (6 hours ahead):")
    for key, value in prediction.items():
        print(f"  {key}: {value}")
    
    # Test irrigation
    print(f"\nIrrigation Controller: {controller}")
    water_used = controller.activate(duration_minutes=30)
    print(f"Water delivered: {water_used:.2f} liters")
    
    print("\n=== OOP Architecture Demo Complete ===")
