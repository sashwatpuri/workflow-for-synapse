"""
Data Preprocessing Module
Smart Farming Prediction System

This module handles:
1. Data loading and cleaning
2. Feature engineering
3. Data transformation
4. Train-test splitting
5. Feature scaling and normalization
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Main data preprocessing class"""
    
    def __init__(self, filepath: str):
        """
        Initialize preprocessor
        
        Args:
            filepath: Path to CSV file
        """
        self.filepath = filepath
        self.df = None
        self.df_processed = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV"""
        print("Loading data...")
        self.df = pd.read_csv(self.filepath)
        print(f"Loaded {len(self.df)} records with {len(self.df.columns)} columns")
        return self.df
    
    def clean_data(self) -> pd.DataFrame:
        """Clean and validate data"""
        print("\nCleaning data...")
        
        # Check for missing values
        missing = self.df.isnull().sum()
        if missing.any():
            print(f"Missing values found:\n{missing[missing > 0]}")
            # Fill missing values
            self.df = self.df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            print(f"Removing {duplicates} duplicate records")
            self.df = self.df.drop_duplicates()
        
        # Validate ranges
        self.df['soil_moisture_%'] = self.df['soil_moisture_%'].clip(0, 100)
        self.df['soil_pH'] = self.df['soil_pH'].clip(0, 14)
        self.df['humidity_%'] = self.df['humidity_%'].clip(0, 100)
        self.df['NDVI_index'] = self.df['NDVI_index'].clip(0, 1)
        
        print("Data cleaning complete")
        return self.df
    
    def engineer_features(self) -> pd.DataFrame:
        """Engineer features for ML models"""
        print("\nEngineering features...")
        
        df = self.df.copy()
        
        # Convert dates
        df['sowing_date'] = pd.to_datetime(df['sowing_date'], format='%d-%m-%Y')
        df['harvest_date'] = pd.to_datetime(df['harvest_date'], format='%d-%m-%Y')
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y')
        
        # === Temporal Features ===
        df['days_since_sowing'] = (df['timestamp'] - df['sowing_date']).dt.days
        df['days_until_harvest'] = (df['harvest_date'] - df['timestamp']).dt.days
        df['growth_progress'] = df['days_since_sowing'] / df['total_days']
        df['growth_progress'] = df['growth_progress'].clip(0, 1)
        
        # Growth stage categorical
        df['growth_stage'] = pd.cut(
            df['growth_progress'],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['Initial', 'Development', 'Mid', 'Late'],
            include_lowest=True
        )
        
        # Month and season
        df['month'] = df['timestamp'].dt.month
        df['season'] = df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # === Environmental Features ===
        
        # Evapotranspiration proxy (simplified Penman-Monteith)
        df['et_proxy'] = 0.1 * df['temperature_C'] * (1 - df['humidity_%'] / 100) * df['sunlight_hours']
        df['et_proxy'] = df['et_proxy'].clip(0, 20)
        
        # Vapor Pressure Deficit (VPD)
        # VPD = (1 - RH/100) * 0.611 * exp(17.27 * T / (T + 237.3))
        df['vpd'] = (1 - df['humidity_%'] / 100) * 0.611 * np.exp(
            17.27 * df['temperature_C'] / (df['temperature_C'] + 237.3)
        )
        
        # Heat stress index
        df['heat_stress'] = ((df['temperature_C'] - 25).clip(0) / 10).clip(0, 1)
        
        # Water balance proxy
        df['water_balance'] = df['rainfall_mm'] - df['et_proxy']
        
        # === Soil Features ===
        
        # Optimal moisture ranges by crop
        crop_moisture_min = {
            'Wheat': 25, 'Rice': 40, 'Maize': 30, 'Cotton': 25, 'Soybean': 30
        }
        crop_moisture_max = {
            'Wheat': 40, 'Rice': 60, 'Maize': 45, 'Cotton': 40, 'Soybean': 45
        }
        
        df['moisture_min'] = df['crop_type'].map(crop_moisture_min)
        df['moisture_max'] = df['crop_type'].map(crop_moisture_max)
        df['moisture_optimal'] = (df['moisture_min'] + df['moisture_max']) / 2
        
        # Moisture deficit and surplus
        df['moisture_deficit'] = (df['moisture_min'] - df['soil_moisture_%']).clip(lower=0)
        df['moisture_surplus'] = (df['soil_moisture_%'] - df['moisture_max']).clip(lower=0)
        
        # Water stress indicator (0 = no stress, 1 = severe stress)
        df['water_stress'] = (df['moisture_deficit'] / df['moisture_min']).clip(0, 1)
        
        # Soil pH deviation from optimal (6.0-7.0)
        df['ph_deviation'] = np.abs(df['soil_pH'] - 6.5)
        
        # === Crop Health Features ===
        
        # NDVI-based health score
        df['ndvi_health_score'] = df['NDVI_index'] * 100
        
        # Disease severity encoding
        disease_severity = {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
        df['disease_severity'] = df['crop_disease_status'].map(disease_severity)
        
        # Combined health index
        df['crop_health_index'] = (
            0.6 * df['ndvi_health_score'] +
            0.4 * (100 - df['disease_severity'] * 25)
        )
        
        # === Agricultural Practice Features ===
        
        # Irrigation efficiency
        irrigation_efficiency = {
            'Drip': 0.90, 'Sprinkler': 0.75, 'Manual': 0.60, 'None': 0.0
        }
        df['irrigation_efficiency'] = df['irrigation_type'].map(irrigation_efficiency)
        
        # Fertilizer impact (proxy)
        fertilizer_impact = {'Organic': 0.8, 'Inorganic': 1.0, 'Mixed': 0.9}
        df['fertilizer_impact'] = df['fertilizer_type'].map(fertilizer_impact)
        
        # Pesticide intensity (normalized)
        df['pesticide_intensity'] = (df['pesticide_usage_ml'] - df['pesticide_usage_ml'].min()) / \
                                    (df['pesticide_usage_ml'].max() - df['pesticide_usage_ml'].min())
        
        # === Interaction Features ===
        
        # Temperature × Humidity interaction
        df['temp_humidity_interaction'] = df['temperature_C'] * df['humidity_%'] / 100
        
        # Moisture × Growth stage interaction
        df['moisture_growth_interaction'] = df['soil_moisture_%'] * df['growth_progress']
        
        # NDVI × Water stress interaction
        df['ndvi_stress_interaction'] = df['NDVI_index'] * (1 - df['water_stress'])
        
        # === Target Variables ===
        
        # Binary: Irrigation needed (primary target)
        df['irrigation_needed'] = (df['soil_moisture_%'] < df['moisture_min']).astype(int)
        
        # Continuous: Irrigation amount needed (liters per hectare)
        # Assuming 1mm of water = 10,000 liters per hectare
        df['irrigation_amount_needed'] = df['moisture_deficit'] * 10000
        
        # Probability of rain (based on humidity and temperature)
        df['rain_probability'] = 1 / (1 + np.exp(-(0.05 * df['humidity_%'] - 0.02 * df['temperature_C'] - 2)))
        
        # === Lag Features (for time series) ===
        
        # Sort by farm and timestamp
        df = df.sort_values(['farm_id', 'timestamp'])
        
        # Create lag features (previous moisture, rainfall)
        df['moisture_lag_1'] = df.groupby('farm_id')['soil_moisture_%'].shift(1)
        df['rainfall_lag_1'] = df.groupby('farm_id')['rainfall_mm'].shift(1)
        
        # Fill NaN lag features with current values
        df['moisture_lag_1'] = df['moisture_lag_1'].fillna(df['soil_moisture_%'])
        df['rainfall_lag_1'] = df['rainfall_lag_1'].fillna(df['rainfall_mm'])
        
        # Moisture change rate
        df['moisture_change_rate'] = df['soil_moisture_%'] - df['moisture_lag_1']
        
        # === Rolling Statistics ===
        
        # 7-day rolling averages (simulated - in real scenario would use actual time series)
        df['rainfall_7day_avg'] = df.groupby('farm_id')['rainfall_mm'].transform(
            lambda x: x.rolling(window=min(7, len(x)), min_periods=1).mean()
        )
        df['temperature_7day_avg'] = df.groupby('farm_id')['temperature_C'].transform(
            lambda x: x.rolling(window=min(7, len(x)), min_periods=1).mean()
        )
        
        print(f"Feature engineering complete. Total features: {len(df.columns)}")
        self.df_processed = df
        return df
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        print("\nEncoding categorical variables...")
        
        categorical_cols = ['region', 'crop_type', 'irrigation_type', 
                          'fertilizer_type', 'crop_disease_status', 
                          'growth_stage', 'season']
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        return df
    
    def prepare_features(self) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare final feature set for modeling"""
        print("\nPreparing features for modeling...")
        
        # Select feature columns
        feature_cols = [
            # Environmental
            'soil_moisture_%', 'soil_pH', 'temperature_C', 'rainfall_mm',
            'humidity_%', 'sunlight_hours', 'NDVI_index',
            
            # Engineered environmental
            'et_proxy', 'vpd', 'heat_stress', 'water_balance', 'rain_probability',
            
            # Soil
            'moisture_deficit', 'moisture_surplus', 'water_stress', 'ph_deviation',
            
            # Crop health
            'ndvi_health_score', 'disease_severity', 'crop_health_index',
            
            # Agricultural practices
            'irrigation_efficiency', 'fertilizer_impact', 'pesticide_intensity',
            
            # Temporal
            'days_since_sowing', 'days_until_harvest', 'growth_progress',
            'month',
            
            # Interactions
            'temp_humidity_interaction', 'moisture_growth_interaction',
            'ndvi_stress_interaction',
            
            # Lag features
            'moisture_lag_1', 'rainfall_lag_1', 'moisture_change_rate',
            
            # Rolling stats
            'rainfall_7day_avg', 'temperature_7day_avg',
            
            # Encoded categoricals
            'region_encoded', 'crop_type_encoded', 'irrigation_type_encoded',
            'fertilizer_type_encoded', 'crop_disease_status_encoded',
            'growth_stage_encoded', 'season_encoded'
        ]
        
        # Filter to existing columns
        feature_cols = [col for col in feature_cols if col in self.df_processed.columns]
        
        X = self.df_processed[feature_cols].copy()
        
        # Handle any remaining NaN
        X = X.fillna(X.mean())
        
        print(f"Final feature set: {len(feature_cols)} features")
        return X, feature_cols
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """Split data into train and test sets"""
        print(f"\nSplitting data (test_size={test_size})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple:
        """Scale features using StandardScaler"""
        print("\nScaling features...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        return X_train_scaled, X_test_scaled
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of processed data"""
        if self.df_processed is None:
            return {}
        
        return {
            'total_records': len(self.df_processed),
            'total_features': len(self.df_processed.columns),
            'farms': self.df_processed['farm_id'].nunique(),
            'regions': self.df_processed['region'].nunique(),
            'crops': self.df_processed['crop_type'].nunique(),
            'irrigation_needed_pct': self.df_processed['irrigation_needed'].mean() * 100,
            'avg_soil_moisture': self.df_processed['soil_moisture_%'].mean(),
            'avg_water_stress': self.df_processed['water_stress'].mean(),
            'avg_crop_health': self.df_processed['crop_health_index'].mean()
        }
    
    def run_full_pipeline(self) -> Tuple:
        """Run complete preprocessing pipeline"""
        print("="*70)
        print("SMART FARMING DATA PREPROCESSING PIPELINE")
        print("="*70)
        
        # Load and clean
        self.load_data()
        self.clean_data()
        
        # Engineer features
        self.engineer_features()
        self.encode_categorical(self.df_processed)
        
        # Prepare features
        X, feature_cols = self.prepare_features()
        y = self.df_processed['irrigation_needed']
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Summary
        print("\n" + "="*70)
        print("PREPROCESSING COMPLETE")
        print("="*70)
        summary = self.get_data_summary()
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor(
        r"c:\Users\sashwat puri sachdev\OneDrive\Documents\synapse pro project\Smart_Farming_Crop_Yield_2024.csv"
    )
    
    X_train, X_test, y_train, y_test, feature_cols = preprocessor.run_full_pipeline()
    
    print(f"\nFeature columns ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, 1):
        print(f"{i:2d}. {col}")
    
    print(f"\nData shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
