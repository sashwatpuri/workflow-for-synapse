# UML Diagrams
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
