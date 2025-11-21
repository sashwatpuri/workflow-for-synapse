"""
Probability & Random Processes Module
Smart Farming Prediction System

This module implements:
1. Markov Process for soil moisture prediction
2. Poisson Process for rainfall modeling
3. Gaussian Noise modeling for sensors
4. Monte Carlo Simulation for future predictions
5. Bayesian Inference for irrigation probability
6. Probability distributions and uncertainty quantification

Mathematical Models:
- Markov Chain: P(X_t+1 | X_t, X_t-1, ...) = P(X_t+1 | X_t)
- Poisson Process: P(N(t) = k) = (λt)^k * e^(-λt) / k!
- Gaussian Noise: N(μ, σ²)
- Bayes' Theorem: P(A|B) = P(B|A) * P(A) / P(B)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import factorial
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class ProbabilityDistribution:
    """Represents a probability distribution"""
    mean: float
    std: float
    distribution_type: str  # 'normal', 'poisson', 'exponential'
    samples: np.ndarray = None
    
    def generate_samples(self, n_samples: int = 1000):
        """Generate random samples from the distribution"""
        if self.distribution_type == 'normal':
            self.samples = np.random.normal(self.mean, self.std, n_samples)
        elif self.distribution_type == 'poisson':
            self.samples = np.random.poisson(self.mean, n_samples)
        elif self.distribution_type == 'exponential':
            self.samples = np.random.exponential(self.mean, n_samples)
        return self.samples
    
    def pdf(self, x):
        """Probability density function"""
        if self.distribution_type == 'normal':
            return stats.norm.pdf(x, self.mean, self.std)
        elif self.distribution_type == 'poisson':
            return stats.poisson.pmf(x, self.mean)
        elif self.distribution_type == 'exponential':
            return stats.expon.pdf(x, scale=self.mean)
    
    def cdf(self, x):
        """Cumulative distribution function"""
        if self.distribution_type == 'normal':
            return stats.norm.cdf(x, self.mean, self.std)
        elif self.distribution_type == 'poisson':
            return stats.poisson.cdf(x, self.mean)
        elif self.distribution_type == 'exponential':
            return stats.expon.cdf(x, scale=self.mean)


class MarkovChainMoisture:
    """
    Markov Chain model for soil moisture prediction
    
    States: Discretized moisture levels
    Transition Matrix: P(state_t+1 | state_t)
    
    Assumptions:
    - Markov property: Future depends only on present
    - Stationary transitions: Probabilities don't change over time
    """
    
    def __init__(self, n_states: int = 10):
        """
        Initialize Markov Chain
        
        Args:
            n_states: Number of discrete moisture states (0-100% divided into bins)
        """
        self.n_states = n_states
        self.state_bins = np.linspace(0, 100, n_states + 1)
        self.transition_matrix = np.zeros((n_states, n_states))
        self.state_probabilities = np.ones(n_states) / n_states  # Uniform initial
        self.is_fitted = False
    
    def _discretize_moisture(self, moisture: float) -> int:
        """Convert continuous moisture to discrete state"""
        state = np.digitize(moisture, self.state_bins) - 1
        return min(max(state, 0), self.n_states - 1)
    
    def fit(self, moisture_sequence: List[float]):
        """
        Estimate transition matrix from historical data
        
        Args:
            moisture_sequence: Time series of soil moisture values
        """
        # Count transitions
        transition_counts = np.zeros((self.n_states, self.n_states))
        
        for i in range(len(moisture_sequence) - 1):
            current_state = self._discretize_moisture(moisture_sequence[i])
            next_state = self._discretize_moisture(moisture_sequence[i + 1])
            transition_counts[current_state, next_state] += 1
        
        # Normalize to get probabilities
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.transition_matrix = transition_counts / row_sums
        
        # Calculate stationary distribution
        self._calculate_stationary_distribution()
        
        self.is_fitted = True
    
    def _calculate_stationary_distribution(self):
        """Calculate stationary distribution π where π = π * P"""
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        
        # Find eigenvector corresponding to eigenvalue 1
        idx = np.argmax(np.abs(eigenvalues - 1.0) < 1e-10)
        stationary = np.real(eigenvectors[:, idx])
        self.state_probabilities = stationary / stationary.sum()
    
    def predict_next(self, current_moisture: float) -> ProbabilityDistribution:
        """
        Predict next moisture state probability distribution
        
        Args:
            current_moisture: Current soil moisture value
            
        Returns:
            Probability distribution of next moisture state
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        current_state = self._discretize_moisture(current_moisture)
        next_state_probs = self.transition_matrix[current_state]
        
        # Calculate expected value and variance
        state_centers = (self.state_bins[:-1] + self.state_bins[1:]) / 2
        expected_moisture = np.sum(next_state_probs * state_centers)
        variance = np.sum(next_state_probs * (state_centers - expected_moisture) ** 2)
        std_dev = np.sqrt(variance)
        
        return ProbabilityDistribution(
            mean=expected_moisture,
            std=std_dev,
            distribution_type='normal'
        )
    
    def predict_n_steps(self, current_moisture: float, n_steps: int) -> List[ProbabilityDistribution]:
        """
        Predict moisture distribution for n future time steps
        
        Args:
            current_moisture: Current soil moisture
            n_steps: Number of steps ahead to predict
            
        Returns:
            List of probability distributions for each future step
        """
        predictions = []
        current_state = self._discretize_moisture(current_moisture)
        state_distribution = np.zeros(self.n_states)
        state_distribution[current_state] = 1.0
        
        state_centers = (self.state_bins[:-1] + self.state_bins[1:]) / 2
        
        for step in range(n_steps):
            # Propagate distribution forward
            state_distribution = state_distribution @ self.transition_matrix
            
            # Calculate statistics
            expected_moisture = np.sum(state_distribution * state_centers)
            variance = np.sum(state_distribution * (state_centers - expected_moisture) ** 2)
            std_dev = np.sqrt(variance)
            
            predictions.append(ProbabilityDistribution(
                mean=expected_moisture,
                std=std_dev,
                distribution_type='normal'
            ))
        
        return predictions


class PoissonRainfallModel:
    """
    Poisson Process for rainfall event modeling
    
    Models rainfall as random events occurring at rate λ
    P(k events in time t) = (λt)^k * e^(-λt) / k!
    
    Assumptions:
    - Events are independent
    - Events occur at constant average rate
    - Two events cannot occur simultaneously
    """
    
    def __init__(self):
        self.lambda_rate = 0.0  # Events per day
        self.mean_rainfall = 0.0  # mm per event
        self.is_fitted = False
    
    def fit(self, rainfall_data: List[float], time_period_days: int):
        """
        Estimate Poisson rate from historical data
        
        Args:
            rainfall_data: List of daily rainfall amounts (mm)
            time_period_days: Number of days in the observation period
        """
        # Count rainfall events (days with rainfall > 0)
        rainfall_events = [r for r in rainfall_data if r > 0]
        n_events = len(rainfall_events)
        
        # Estimate rate (events per day)
        self.lambda_rate = n_events / time_period_days
        
        # Estimate mean rainfall per event
        if n_events > 0:
            self.mean_rainfall = np.mean(rainfall_events)
        else:
            self.mean_rainfall = 0.0
        
        self.is_fitted = True
    
    def predict_probability(self, n_events: int, time_days: float = 1.0) -> float:
        """
        Calculate probability of exactly n rainfall events in given time
        
        Args:
            n_events: Number of events
            time_days: Time period in days
            
        Returns:
            Probability P(N(t) = n)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        lambda_t = self.lambda_rate * time_days
        prob = (lambda_t ** n_events) * np.exp(-lambda_t) / factorial(n_events)
        return prob
    
    def predict_cumulative_probability(self, max_events: int, time_days: float = 1.0) -> float:
        """
        Calculate probability of at most max_events in given time
        
        Args:
            max_events: Maximum number of events
            time_days: Time period in days
            
        Returns:
            Cumulative probability P(N(t) <= max_events)
        """
        cumulative_prob = sum(
            self.predict_probability(k, time_days)
            for k in range(max_events + 1)
        )
        return cumulative_prob
    
    def predict_expected_rainfall(self, time_days: float = 1.0) -> ProbabilityDistribution:
        """
        Predict expected rainfall amount in given time period
        
        Args:
            time_days: Time period in days
            
        Returns:
            Probability distribution of total rainfall
        """
        expected_events = self.lambda_rate * time_days
        expected_rainfall = expected_events * self.mean_rainfall
        
        # Variance of compound Poisson process
        variance = expected_events * (self.mean_rainfall ** 2)
        std_dev = np.sqrt(variance)
        
        return ProbabilityDistribution(
            mean=expected_rainfall,
            std=std_dev,
            distribution_type='normal'
        )
    
    def simulate_rainfall(self, n_days: int, n_simulations: int = 1) -> np.ndarray:
        """
        Simulate rainfall using Poisson process
        
        Args:
            n_days: Number of days to simulate
            n_simulations: Number of simulation runs
            
        Returns:
            Array of shape (n_simulations, n_days) with daily rainfall
        """
        simulations = np.zeros((n_simulations, n_days))
        
        for sim in range(n_simulations):
            for day in range(n_days):
                # Number of events on this day
                n_events = np.random.poisson(self.lambda_rate)
                
                # Total rainfall (sum of event amounts)
                if n_events > 0:
                    # Each event has exponentially distributed amount
                    rainfall_amounts = np.random.exponential(self.mean_rainfall, n_events)
                    simulations[sim, day] = rainfall_amounts.sum()
        
        return simulations


class GaussianSensorNoise:
    """
    Gaussian noise model for sensor measurements
    
    True value = Measured value + N(0, σ²)
    """
    
    def __init__(self, noise_std: float = 0.5):
        """
        Initialize noise model
        
        Args:
            noise_std: Standard deviation of Gaussian noise
        """
        self.noise_std = noise_std
        self.noise_mean = 0.0
    
    def add_noise(self, true_values: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to true values
        
        Args:
            true_values: Array of true sensor values
            
        Returns:
            Noisy measurements
        """
        noise = np.random.normal(self.noise_mean, self.noise_std, len(true_values))
        return true_values + noise
    
    def estimate_true_value(self, measurements: np.ndarray) -> Tuple[float, float]:
        """
        Estimate true value from noisy measurements using MLE
        
        Args:
            measurements: Array of noisy measurements
            
        Returns:
            (estimated_mean, confidence_interval_width)
        """
        estimated_mean = np.mean(measurements)
        
        # 95% confidence interval
        n = len(measurements)
        std_error = self.noise_std / np.sqrt(n)
        ci_width = 1.96 * std_error
        
        return estimated_mean, ci_width
    
    def kalman_filter(self, measurements: np.ndarray, 
                     process_variance: float = 0.1) -> np.ndarray:
        """
        Apply Kalman filter to reduce noise
        
        Args:
            measurements: Noisy measurements
            process_variance: Variance of process model
            
        Returns:
            Filtered estimates
        """
        n = len(measurements)
        estimates = np.zeros(n)
        estimate_variance = np.zeros(n)
        
        # Initialize
        estimates[0] = measurements[0]
        estimate_variance[0] = self.noise_std ** 2
        
        for i in range(1, n):
            # Prediction
            predicted_estimate = estimates[i-1]
            predicted_variance = estimate_variance[i-1] + process_variance
            
            # Update
            kalman_gain = predicted_variance / (predicted_variance + self.noise_std ** 2)
            estimates[i] = predicted_estimate + kalman_gain * (measurements[i] - predicted_estimate)
            estimate_variance[i] = (1 - kalman_gain) * predicted_variance
        
        return estimates


class MonteCarloSimulator:
    """
    Monte Carlo simulation for future soil moisture prediction
    
    Simulates multiple possible futures considering:
    - Rainfall uncertainty (Poisson process)
    - Evapotranspiration variability
    - Irrigation decisions
    - Sensor noise
    """
    
    def __init__(self, 
                 markov_model: MarkovChainMoisture,
                 rainfall_model: PoissonRainfallModel,
                 noise_model: GaussianSensorNoise):
        self.markov_model = markov_model
        self.rainfall_model = rainfall_model
        self.noise_model = noise_model
    
    def simulate_moisture_trajectory(self,
                                    initial_moisture: float,
                                    n_days: int,
                                    et_rate: float = 2.0,
                                    irrigation_threshold: float = 25.0,
                                    irrigation_amount: float = 10.0) -> np.ndarray:
        """
        Simulate single moisture trajectory
        
        Args:
            initial_moisture: Starting soil moisture (%)
            n_days: Number of days to simulate
            et_rate: Evapotranspiration rate (mm/day)
            irrigation_threshold: Moisture level triggering irrigation
            irrigation_amount: Amount of water added by irrigation (mm)
            
        Returns:
            Array of daily moisture values
        """
        moisture = np.zeros(n_days + 1)
        moisture[0] = initial_moisture
        
        for day in range(n_days):
            # Current moisture
            current = moisture[day]
            
            # Rainfall (Poisson process)
            n_rain_events = np.random.poisson(self.rainfall_model.lambda_rate)
            rainfall = 0.0
            if n_rain_events > 0:
                rainfall = np.sum(np.random.exponential(
                    self.rainfall_model.mean_rainfall, n_rain_events
                ))
            
            # Evapotranspiration loss
            et_loss = et_rate
            
            # Irrigation decision
            irrigation = 0.0
            if current < irrigation_threshold:
                irrigation = irrigation_amount
            
            # Update moisture (simplified water balance)
            # Δθ = (P + I - ET) / soil_depth
            # Assuming soil depth = 100mm for normalization
            moisture_change = (rainfall + irrigation - et_loss) / 10.0
            
            # Add Markov transition uncertainty
            markov_pred = self.markov_model.predict_next(current)
            markov_noise = np.random.normal(0, markov_pred.std * 0.1)
            
            # Update
            moisture[day + 1] = current + moisture_change + markov_noise
            moisture[day + 1] = np.clip(moisture[day + 1], 0, 100)
        
        return moisture
    
    def run_monte_carlo(self,
                       initial_moisture: float,
                       n_days: int,
                       n_simulations: int = 1000,
                       **kwargs) -> Dict:
        """
        Run Monte Carlo simulation
        
        Args:
            initial_moisture: Starting moisture
            n_days: Simulation horizon
            n_simulations: Number of simulation runs
            **kwargs: Additional parameters for simulate_moisture_trajectory
            
        Returns:
            Dictionary with statistics and trajectories
        """
        trajectories = np.zeros((n_simulations, n_days + 1))
        
        for sim in range(n_simulations):
            trajectories[sim] = self.simulate_moisture_trajectory(
                initial_moisture, n_days, **kwargs
            )
        
        # Calculate statistics
        mean_trajectory = np.mean(trajectories, axis=0)
        std_trajectory = np.std(trajectories, axis=0)
        percentile_5 = np.percentile(trajectories, 5, axis=0)
        percentile_95 = np.percentile(trajectories, 95, axis=0)
        
        # Probability of irrigation needed
        irrigation_threshold = kwargs.get('irrigation_threshold', 25.0)
        prob_irrigation_needed = np.mean(trajectories < irrigation_threshold, axis=0)
        
        return {
            'trajectories': trajectories,
            'mean': mean_trajectory,
            'std': std_trajectory,
            'percentile_5': percentile_5,
            'percentile_95': percentile_95,
            'prob_irrigation_needed': prob_irrigation_needed,
            'n_simulations': n_simulations,
            'n_days': n_days
        }


class BayesianIrrigationPredictor:
    """
    Bayesian inference for irrigation probability
    
    P(irrigation_needed | observations) = 
        P(observations | irrigation_needed) * P(irrigation_needed) / P(observations)
    """
    
    def __init__(self):
        self.prior_irrigation_prob = 0.3  # Prior belief
        self.likelihood_params = {}
    
    def set_prior(self, prior_prob: float):
        """Set prior probability of irrigation being needed"""
        self.prior_irrigation_prob = prior_prob
    
    def fit_likelihood(self, df: pd.DataFrame, target_col: str = 'irrigation_needed'):
        """
        Estimate likelihood P(observations | irrigation_needed)
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
        """
        # Separate data by class
        irrigated = df[df[target_col] == 1]
        not_irrigated = df[df[target_col] == 0]
        
        # Estimate Gaussian likelihood for continuous features
        features = ['soil_moisture_%', 'temperature_C', 'humidity_%', 'rainfall_mm']
        
        self.likelihood_params = {
            'irrigated': {},
            'not_irrigated': {}
        }
        
        for feature in features:
            self.likelihood_params['irrigated'][feature] = {
                'mean': irrigated[feature].mean(),
                'std': irrigated[feature].std()
            }
            self.likelihood_params['not_irrigated'][feature] = {
                'mean': not_irrigated[feature].mean(),
                'std': not_irrigated[feature].std()
            }
    
    def predict_probability(self, observations: Dict[str, float]) -> float:
        """
        Predict P(irrigation_needed | observations) using Bayes' theorem
        
        Args:
            observations: Dictionary of feature values
            
        Returns:
            Posterior probability of irrigation being needed
        """
        # Calculate likelihoods
        likelihood_irrigated = 1.0
        likelihood_not_irrigated = 1.0
        
        for feature, value in observations.items():
            if feature in self.likelihood_params['irrigated']:
                params_irr = self.likelihood_params['irrigated'][feature]
                params_not = self.likelihood_params['not_irrigated'][feature]
                
                # Gaussian likelihood
                likelihood_irrigated *= stats.norm.pdf(
                    value, params_irr['mean'], params_irr['std']
                )
                likelihood_not_irrigated *= stats.norm.pdf(
                    value, params_not['mean'], params_not['std']
                )
        
        # Apply Bayes' theorem
        numerator = likelihood_irrigated * self.prior_irrigation_prob
        denominator = (likelihood_irrigated * self.prior_irrigation_prob + 
                      likelihood_not_irrigated * (1 - self.prior_irrigation_prob))
        
        if denominator == 0:
            return self.prior_irrigation_prob
        
        posterior = numerator / denominator
        return posterior


def demonstrate_probability_models():
    """Demonstrate all probability models"""
    print("=== Probability & Random Processes Demo ===\n")
    
    # 1. Markov Chain for Soil Moisture
    print("1. Markov Chain - Soil Moisture Prediction")
    print("-" * 50)
    
    # Generate sample moisture data
    np.random.seed(42)
    moisture_sequence = [35.0]
    for _ in range(100):
        change = np.random.normal(-0.5, 2.0)  # Slight downward trend with noise
        new_moisture = moisture_sequence[-1] + change
        moisture_sequence.append(np.clip(new_moisture, 0, 100))
    
    markov = MarkovChainMoisture(n_states=10)
    markov.fit(moisture_sequence)
    
    current_moisture = 30.0
    prediction = markov.predict_next(current_moisture)
    print(f"Current Moisture: {current_moisture:.2f}%")
    print(f"Predicted Next: {prediction.mean:.2f}% ± {prediction.std:.2f}%")
    
    # Multi-step prediction
    multi_step = markov.predict_n_steps(current_moisture, n_steps=7)
    print(f"\n7-Day Forecast:")
    for i, pred in enumerate(multi_step, 1):
        print(f"  Day {i}: {pred.mean:.2f}% ± {pred.std:.2f}%")
    
    # 2. Poisson Process for Rainfall
    print("\n2. Poisson Process - Rainfall Modeling")
    print("-" * 50)
    
    # Generate sample rainfall data
    rainfall_data = []
    for _ in range(365):
        if np.random.random() < 0.3:  # 30% chance of rain
            rainfall_data.append(np.random.exponential(10))
        else:
            rainfall_data.append(0.0)
    
    poisson = PoissonRainfallModel()
    poisson.fit(rainfall_data, time_period_days=365)
    
    print(f"Estimated rainfall rate: {poisson.lambda_rate:.3f} events/day")
    print(f"Mean rainfall per event: {poisson.mean_rainfall:.2f} mm")
    
    # Predict probabilities
    for n in range(4):
        prob = poisson.predict_probability(n, time_days=1.0)
        print(f"P({n} rain events in 1 day) = {prob:.4f}")
    
    # Expected rainfall
    expected = poisson.predict_expected_rainfall(time_days=7.0)
    print(f"\nExpected rainfall (7 days): {expected.mean:.2f} ± {expected.std:.2f} mm")
    
    # 3. Gaussian Sensor Noise
    print("\n3. Gaussian Sensor Noise Modeling")
    print("-" * 50)
    
    noise_model = GaussianSensorNoise(noise_std=0.8)
    true_moisture = 35.0
    measurements = noise_model.add_noise(np.full(20, true_moisture))
    
    estimated, ci = noise_model.estimate_true_value(measurements)
    print(f"True value: {true_moisture:.2f}%")
    print(f"Estimated value: {estimated:.2f}% ± {ci:.2f}% (95% CI)")
    
    # Kalman filtering
    filtered = noise_model.kalman_filter(measurements)
    print(f"Kalman filtered: {filtered[-1]:.2f}%")
    
    # 4. Monte Carlo Simulation
    print("\n4. Monte Carlo Simulation")
    print("-" * 50)
    
    mc_simulator = MonteCarloSimulator(markov, poisson, noise_model)
    
    results = mc_simulator.run_monte_carlo(
        initial_moisture=30.0,
        n_days=14,
        n_simulations=500,
        et_rate=2.5,
        irrigation_threshold=25.0,
        irrigation_amount=10.0
    )
    
    print(f"Simulations run: {results['n_simulations']}")
    print(f"Forecast horizon: {results['n_days']} days")
    print(f"\nDay 7 Forecast:")
    print(f"  Mean: {results['mean'][7]:.2f}%")
    print(f"  Std Dev: {results['std'][7]:.2f}%")
    print(f"  90% CI: [{results['percentile_5'][7]:.2f}%, {results['percentile_95'][7]:.2f}%]")
    print(f"  P(irrigation needed): {results['prob_irrigation_needed'][7]:.2%}")
    
    print("\n=== Probability Models Demo Complete ===")


if __name__ == "__main__":
    demonstrate_probability_models()
