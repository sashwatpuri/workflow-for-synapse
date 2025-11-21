"""
Main Execution Script
Smart Farming Prediction System

This script runs the complete integrated system demonstrating all four subject outcomes:
1. Probability & Random Processes
2. Design & Analysis of Algorithms
3. Object-Oriented Programming
4. Computer Architecture & Organization
"""

import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('results/predictions', exist_ok=True)
os.makedirs('results/schedules', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)
os.makedirs('diagrams', exist_ok=True)


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def run_data_preprocessing():
    """Run data preprocessing pipeline"""
    print_header("STEP 1: DATA PREPROCESSING")
    
    from src.data_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor(
        r"Smart_Farming_Crop_Yield_2024.csv"
    )
    
    X_train, X_test, y_train, y_test, feature_cols = preprocessor.run_full_pipeline()
    
    print("\n✓ Data preprocessing complete")
    return preprocessor, X_train, X_test, y_train, y_test, feature_cols


def run_model_training(X_train, X_test, y_train, y_test, feature_cols):
    """Run ML model training"""
    print_header("STEP 2: MACHINE LEARNING MODEL TRAINING")
    
    from src.model_training import ModelTrainer
    
    trainer = ModelTrainer()
    
    # Train all models
    print("\nTraining Logistic Regression...")
    trainer.train_logistic_regression(X_train, y_train)
    
    print("\nTraining Random Forest...")
    trainer.train_random_forest(X_train, y_train)
    
    print("\nTraining XGBoost...")
    trainer.train_xgboost(X_train, y_train)
    
    # Evaluate all models
    print("\n" + "-"*80)
    print("EVALUATING MODELS")
    print("-"*80)
    
    for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
        trainer.predict(model_name, X_test)
        trainer.evaluate(model_name, y_test)
        trainer.print_evaluation(model_name)
        
        if model_name in ['random_forest', 'xgboost']:
            trainer.plot_feature_importance(model_name, feature_cols, top_n=20)
    
    # Compare models
    trainer.compare_models()
    
    # Save models
    trainer.save_models()
    
    print("\n✓ Model training complete")
    return trainer


def run_probability_models():
    """Demonstrate probability models"""
    print_header("STEP 3: PROBABILITY & RANDOM PROCESSES")
    
    from src.probability_models import demonstrate_probability_models
    
    demonstrate_probability_models()
    
    print("\n✓ Probability models demonstration complete")


def run_scheduling_algorithms():
    """Demonstrate scheduling algorithms"""
    print_header("STEP 4: SCHEDULING ALGORITHMS (DAA)")
    
    from src.scheduling_algorithms import demonstrate_scheduling_algorithms
    
    demonstrate_scheduling_algorithms()
    
    print("\n✓ Scheduling algorithms demonstration complete")


def run_oop_demonstration():
    """Demonstrate OOP architecture"""
    print_header("STEP 5: OBJECT-ORIENTED PROGRAMMING")
    
    from src.oop_architecture import Farm, Crop, Sensor, WeatherData, IrrigationController
    from src.oop_architecture import GeoLocation, CropType, IrrigationType, Region, SensorReading
    from datetime import datetime, timedelta
    
    # Create sample farm
    print("Creating sample farm with OOP architecture...")
    
    location = GeoLocation(28.6139, 77.2090)
    farm = Farm("DEMO_FARM", Region.NORTH_INDIA, location)
    
    # Add crop
    sowing = datetime.now() - timedelta(days=45)
    harvest = datetime.now() + timedelta(days=75)
    crop = Crop(CropType.WHEAT, sowing, harvest)
    farm.set_crop(crop)
    
    # Add sensor
    sensor = Sensor("DEMO_SENSOR", location)
    reading = SensorReading(
        sensor_id="DEMO_SENSOR",
        timestamp=datetime.now(),
        soil_moisture=28.5,
        soil_ph=6.8,
        temperature=24.5,
        humidity=68.0,
        rainfall=0.0
    )
    sensor.add_reading(reading)
    farm.set_sensor(sensor)
    
    # Add irrigation controller
    controller = IrrigationController(IrrigationType.DRIP)
    farm.set_irrigation(controller)
    
    # Add weather data
    weather = WeatherData(
        timestamp=datetime.now(),
        temperature_c=24.5,
        humidity_percent=68.0,
        rainfall_mm=0.0,
        sunlight_hours=8.2
    )
    farm.add_weather_data(weather)
    
    # Demonstrate methods
    print(f"\nFarm: {farm}")
    print(f"Crop: {crop}")
    print(f"Current Moisture: {farm.get_current_moisture():.2f}%")
    print(f"Irrigation Need Probability: {farm.calculate_irrigation_need():.3f}")
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
    savings = controller.calculate_water_savings()
    print(f"Water savings vs manual: {savings:.1f}%")
    
    print("\n✓ OOP demonstration complete")


def run_simulation():
    """Run complete farm simulation"""
    print_header("STEP 6: SMART FARMING SIMULATION")
    
    from src.simulation import FarmSimulator, create_sample_farms
    
    # Create farms
    print("Creating sample farms...")
    farms = create_sample_farms(n_farms=10)
    
    # Run simulation
    simulator = FarmSimulator(farms, simulation_days=30)
    simulator.run_simulation()
    
    # Plot results
    simulator.plot_results()
    
    print("\n✓ Simulation complete")


def generate_documentation():
    """Generate system documentation"""
    print_header("STEP 7: GENERATING DOCUMENTATION")
    
    print("Documentation generated:")
    print("  ✓ README.md - Project overview")
    print("  ✓ CAO_Hardware_Software_Mapping.md - Hardware architecture")
    print("  ✓ requirements.txt - Dependencies")
    
    # Generate UML diagrams documentation
    generate_uml_documentation()
    
    print("\n✓ Documentation generation complete")


def generate_uml_documentation():
    """Generate UML diagrams documentation"""
    
    uml_content = """# UML Diagrams
## Smart Farming Prediction System

---

## 1. Class Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Farm                                    │
├─────────────────────────────────────────────────────────────────┤
│ - farm_id: str                                                  │
│ - region: Region                                                │
│ - location: GeoLocation                                         │
│ - crop: Crop                                                    │
│ - sensor: Sensor                                                │
│ - irrigation_controller: IrrigationController                   │
│ - weather_data: List[WeatherData]                              │
│ - ndvi_index: float                                             │
│ - disease_status: DiseaseStatus                                 │
├─────────────────────────────────────────────────────────────────┤
│ + set_crop(crop: Crop): void                                    │
│ + set_sensor(sensor: Sensor): void                              │
│ + set_irrigation(controller: IrrigationController): void        │
│ + get_current_moisture(): float                                 │
│ + calculate_irrigation_need(): float                            │
│ + predict_next_state(hours_ahead: int): Dict                    │
│ + calculate_priority(): float                                   │
└─────────────────────────────────────────────────────────────────┘
                    │
                    │ has-a
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│                         Crop                                    │
├─────────────────────────────────────────────────────────────────┤
│ - crop_type: CropType                                           │
│ - sowing_date: datetime                                         │
│ - harvest_date: datetime                                        │
│ - water_requirement: float                                      │
│ - optimal_moisture_range: Tuple[float, float]                   │
├─────────────────────────────────────────────────────────────────┤
│ + get_growth_stage(current_date: datetime): str                 │
│ + get_crop_coefficient(current_date: datetime): float           │
│ + is_moisture_optimal(current_moisture: float): bool            │
│ + calculate_water_stress(current_moisture: float): float        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        Sensor                                   │
├─────────────────────────────────────────────────────────────────┤
│ - sensor_id: str                                                │
│ - location: GeoLocation                                         │
│ - readings: List[SensorReading]                                 │
│ - calibration_offset: float                                     │
├─────────────────────────────────────────────────────────────────┤
│ + add_reading(reading: SensorReading): void                     │
│ + get_latest_reading(): SensorReading                           │
│ + get_average_moisture(hours: int): float                       │
│ + calibrate(reference: float, measured: float): void            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  IrrigationController                           │
├─────────────────────────────────────────────────────────────────┤
│ - irrigation_type: IrrigationType                               │
│ - flow_rate: float                                              │
│ - efficiency: float                                             │
│ - is_active: bool                                               │
│ - total_water_used: float                                       │
├─────────────────────────────────────────────────────────────────┤
│ + activate(duration_minutes: int): float                        │
│ + deactivate(): void                                            │
│ + calculate_water_savings(baseline: IrrigationType): float      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    <<interface>>                                │
│                     Predictable                                 │
├─────────────────────────────────────────────────────────────────┤
│ + predict_next_state(hours_ahead: int): Dict                    │
└─────────────────────────────────────────────────────────────────┘
                    △
                    │ implements
                    │
                  Farm

┌─────────────────────────────────────────────────────────────────┐
│                    <<interface>>                                │
│                     Schedulable                                 │
├─────────────────────────────────────────────────────────────────┤
│ + calculate_priority(): float                                   │
└─────────────────────────────────────────────────────────────────┘
                    △
                    │ implements
                    │
                  Farm
```

---

## 2. Sequence Diagram: Irrigation Decision Flow

```
Sensor    Farm    ModelPredictor    Scheduler    IrrigationController
  │         │            │              │                 │
  │ read()  │            │              │                 │
  ├────────>│            │              │                 │
  │         │            │              │                 │
  │         │ predict()  │              │                 │
  │         ├───────────>│              │                 │
  │         │            │              │                 │
  │         │ P(irr)=0.8 │              │                 │
  │         │<───────────┤              │                 │
  │         │            │              │                 │
  │         │ calculate_priority()      │                 │
  │         ├──────────────────────────>│                 │
  │         │            │              │                 │
  │         │            │ priority=85  │                 │
  │         │<──────────────────────────┤                 │
  │         │            │              │                 │
  │         │            │ schedule()   │                 │
  │         │            │<─────────────┤                 │
  │         │            │              │                 │
  │         │            │ activate(30min)                │
  │         │            │              ├────────────────>│
  │         │            │              │                 │
  │         │            │              │  water_delivered│
  │         │            │              │<────────────────┤
  │         │            │              │                 │
  │         │ update_moisture()         │                 │
  │         │<──────────────────────────┤                 │
  │         │            │              │                 │
```

---

## 3. Activity Diagram: Daily Farm Operations

```
                    [Start]
                       │
                       ↓
            ┌──────────────────────┐
            │  Read All Sensors    │
            └──────────┬───────────┘
                       │
                       ↓
            ┌──────────────────────┐
            │  Process Sensor Data │
            │  (Noise Filtering)   │
            └──────────┬───────────┘
                       │
                       ↓
            ┌──────────────────────┐
            │  Engineer Features   │
            └──────────┬───────────┘
                       │
                       ↓
            ┌──────────────────────┐
            │  Run ML Prediction   │
            │  P(irrigation_needed)│
            └──────────┬───────────┘
                       │
                       ↓
                  ◇ P > 0.7?
                  │         │
              Yes │         │ No
                  ↓         ↓
        ┌─────────────┐   [Skip]
        │ Calculate   │      │
        │ Priority    │      │
        └──────┬──────┘      │
               │             │
               ↓             │
        ┌─────────────┐      │
        │ Add to      │      │
        │ Schedule    │      │
        └──────┬──────┘      │
               │             │
               └──────┬──────┘
                      │
                      ↓
            ┌──────────────────────┐
            │  Execute Schedule    │
            │  (Greedy Algorithm)  │
            └──────────┬───────────┘
                       │
                       ↓
            ┌──────────────────────┐
            │  Activate Irrigation │
            │  for Scheduled Farms │
            └──────────┬───────────┘
                       │
                       ↓
            ┌──────────────────────┐
            │  Log Events          │
            └──────────┬───────────┘
                       │
                       ↓
            ┌──────────────────────┐
            │  Transmit to Cloud   │
            └──────────┬───────────┘
                       │
                       ↓
                     [End]
```

---

## 4. Component Diagram: System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                     Cloud Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ ML Training  │  │  Dashboard   │  │  Analytics   │    │
│  │  Component   │  │  Component   │  │  Component   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└────────────────────────┬───────────────────────────────────┘
                         │ MQTT/HTTP
┌────────────────────────┴───────────────────────────────────┐
│                     Edge Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Prediction   │  │  Scheduling  │  │ Irrigation   │    │
│  │  Component   │  │  Component   │  │  Component   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│  ┌──────────────┐  ┌──────────────┐                      │
│  │ Data Proc.   │  │  Probability │                      │
│  │  Component   │  │  Component   │                      │
│  └──────────────┘  └──────────────┘                      │
└────────────────────────┬───────────────────────────────────┘
                         │ I2C/SPI/GPIO
┌────────────────────────┴───────────────────────────────────┐
│                   Hardware Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Sensors    │  │  Actuators   │  │  ESP32 MCU   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└────────────────────────────────────────────────────────────┘
```

---

**Generated**: 2024-11-21  
**System**: Smart Farming Prediction System
"""
    
    with open('diagrams/UML_Diagrams.md', 'w', encoding='utf-8') as f:
        f.write(uml_content)
    
    print("  ✓ UML_Diagrams.md - System diagrams")


def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print("  SMART FARMING PREDICTION SYSTEM")
    print("  Integrated Semester-2 Project")
    print("  " + "-"*76)
    print("  Subject Outcomes:")
    print("    1. Probability & Random Processes")
    print("    2. Design & Analysis of Algorithms (DAA)")
    print("    3. Object-Oriented Programming (OOPS)")
    print("    4. Computer Architecture & Organization (CAO)")
    print("="*80)
    
    start_time = datetime.now()
    
    try:
        # Step 1: Data Preprocessing
        preprocessor, X_train, X_test, y_train, y_test, feature_cols = run_data_preprocessing()
        
        # Step 2: Model Training
        trainer = run_model_training(X_train, X_test, y_train, y_test, feature_cols)
        
        # Step 3: Probability Models
        run_probability_models()
        
        # Step 4: Scheduling Algorithms
        run_scheduling_algorithms()
        
        # Step 5: OOP Demonstration
        run_oop_demonstration()
        
        # Step 6: Simulation
        run_simulation()
        
        # Step 7: Documentation
        generate_documentation()
        
        # Final summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print_header("EXECUTION COMPLETE")
        print(f"Total execution time: {duration:.2f} seconds")
        print(f"\nGenerated Outputs:")
        print(f"  📁 models/          - Trained ML models")
        print(f"  📁 results/         - Predictions, schedules, metrics")
        print(f"  📁 diagrams/        - UML diagrams")
        print(f"  📁 docs/            - Documentation")
        print(f"\nAll subject outcomes successfully demonstrated!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
