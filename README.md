# 📚 Subject Integration Report
## Smart Farming Prediction System

**Integrated Semester-2 Project**  
**Status:** ✅ 100% Implemented

---

## 🌟 Executive Summary

This project successfully integrates concepts from four core computer science subjects to build a robust, efficient, and scalable Smart Farming Prediction System. Each subject contributes a critical layer to the system's functionality:

1.  **Probability & Random Processes:** Models environmental uncertainty and predicts future states.
2.  **Design & Analysis of Algorithms (DAA):** Optimizes resource allocation (water) and scheduling.
3.  **Object-Oriented Programming (OOP):** Provides a modular, maintainable, and scalable software architecture.
4.  **Computer Architecture (CAO):** Bridges the gap between software logic and hardware sensors/actuators.

---

## 1. 🎲 Probability & Random Processes
**Role:** Modeling Uncertainty & Prediction

The agricultural environment is inherently uncertain. We use probabilistic models to simulate weather patterns, soil moisture behavior, and crop growth to make informed predictions.

### ✅ Implemented Concepts

#### A. Markov Chains (Weather Modeling)
*   **Concept:** A stochastic model describing a sequence of possible events where the probability of each event depends only on the state attained in the previous event.
*   **Implementation:** Used to model weather transitions (e.g., Sunny → Rainy → Cloudy).
*   **Code Location:** `src/probability_models.py` -> `MarkovChainWeather` class.
*   **Matrix:**
    ```python
    # Transition Matrix
    #          Sunny   Rainy   Cloudy
    # Sunny    0.7     0.1     0.2
    # Rainy    0.3     0.5     0.2
    # Cloudy   0.4     0.3     0.3
    ```
*   **Application:** Predicts the weather for the next 7-30 days to adjust irrigation plans proactively.

#### B. Poisson Process (Rainfall Events)
*   **Concept:** Models the occurrence of events (rainfall) happening independently at a constant average rate.
*   **Implementation:** Simulates the number of rainfall events in a given period.
*   **Code Location:** `src/probability_models.py` -> `PoissonRainfall` class.
*   **Application:** Estimates the frequency of rain to determine if irrigation can be delayed.

#### C. Gaussian (Normal) Distribution (Sensor Noise)
*   **Concept:** Models random noise in sensor readings (Temperature, Moisture).
*   **Implementation:** Adds Gaussian noise ($\mu=0, \sigma=0.5$) to synthetic sensor data to test system robustness.
*   **Code Location:** `src/probability_models.py` -> `add_gaussian_noise()`.
*   **Application:** Ensures the system doesn't overreact to minor sensor fluctuations.

#### D. Monte Carlo Simulation (Risk Assessment)
*   **Concept:** Uses repeated random sampling to obtain numerical results.
*   **Implementation:** Runs 500+ simulations of soil moisture decay under different weather conditions.
*   **Code Location:** `src/probability_models.py` -> `MonteCarloSimulation` class.
*   **Application:** Calculates the **probability of crop failure** due to drought.

---

## 2. ⚡ Design & Analysis of Algorithms (DAA)
**Role:** Optimization & Efficiency

Efficiently managing limited water resources across hundreds of farms requires optimized algorithms. We implemented and analyzed multiple scheduling approaches.

### ✅ Implemented Concepts

#### A. Greedy Algorithm (Priority Scheduling)
*   **Concept:** Makes the locally optimal choice at each stage (irrigating the driest farm first) to find a global optimum.
*   **Implementation:** Sorts farms by `Priority Score` (based on moisture deficit and crop value) and allocates water until the supply runs out.
*   **Code Location:** `src/scheduling_algorithms.py` -> `GreedyScheduler` class.
*   **Complexity:**
    *   **Time:** $O(N \log N)$ (due to sorting).
    *   **Space:** $O(N)$ (to store farm list).
*   **Application:** The primary scheduler for daily irrigation.

#### B. Heap-Based Priority Queue
*   **Concept:** Uses a Binary Heap data structure to efficiently retrieve the element with the highest priority.
*   **Implementation:** Maintains a max-heap of farms based on water stress.
*   **Code Location:** `src/scheduling_algorithms.py` -> `HeapScheduler` class.
*   **Complexity:**
    *   **Insertion:** $O(\log N)$
    *   **Extraction:** $O(\log N)$
*   **Application:** Used for real-time dynamic scheduling where farm statuses change frequently.

#### C. Dynamic Programming (0/1 Knapsack Variant)
*   **Concept:** Solves complex problems by breaking them down into simpler subproblems.
*   **Implementation:** Treats water as the "knapsack capacity" and farms as "items" with value (crop yield) and weight (water needed). Maximizes total crop yield for a fixed water budget.
*   **Code Location:** `src/scheduling_algorithms.py` -> `KnapsackScheduler` class.
*   **Complexity:** $O(N \times W)$ where $W$ is total water capacity.
*   **Application:** Used when water is extremely scarce to maximize economic return.

---

## 3. 🏗️ Object-Oriented Programming (OOP)
**Role:** System Architecture & Modularity

The system is built using robust OOP principles to ensure code reusability, security, and scalability.

### ✅ Implemented Concepts

#### A. Encapsulation
*   **Concept:** Bundling data (attributes) and methods (functions) together and restricting direct access to some components.
*   **Implementation:**
    *   `Farm` class has private attributes like `__soil_moisture` and `__crop_health`.
    *   Access is controlled via getters/setters (e.g., `get_moisture()`, `update_status()`).
*   **Code Location:** `src/oop_architecture.py`.

#### B. Inheritance
*   **Concept:** Creating new classes based on existing ones to reuse code.
*   **Implementation:**
    *   **Base Class:** `Sensor` (generic sensor properties).
    *   **Derived Classes:** `MoistureSensor`, `TemperatureSensor`, `HumiditySensor` (specific implementations).
*   **Code Location:** `src/oop_architecture.py`.

#### C. Polymorphism
*   **Concept:** Using a single interface to represent different underlying forms.
*   **Implementation:**
    *   The `read_data()` method exists in all sensor classes but behaves differently for each (reading moisture vs. temperature).
    *   The `Scheduler` interface allows swapping between Greedy, Heap, and Knapsack algorithms without changing the main code.
*   **Code Location:** `src/oop_architecture.py`.

#### D. Abstraction
*   **Concept:** Hiding complex implementation details and showing only necessary features.
*   **Implementation:**
    *   The `IrrigationSystem` class provides a simple `run_daily_cycle()` method that hides the complexity of gathering sensor data, running ML models, and executing scheduling algorithms.
*   **Code Location:** `src/oop_architecture.py`.

---

## 4. 💻 Computer Architecture (CAO)
**Role:** Hardware-Software Interface

This subject bridges the gap between our Python code and the physical world of IoT sensors and actuators.

### ✅ Implemented Concepts

#### A. Sensor Data Acquisition (Input)
*   **Concept:** Mapping analog signals from sensors to digital values processed by the CPU.
*   **Implementation:**
    *   **Soil Moisture Sensor:** Maps 0-5V analog signal to 0-100% digital moisture value.
    *   **DHT11 Sensor:** Uses a specific digital protocol for Temperature/Humidity.
*   **Mapping:** Documented in `docs/CAO_Hardware_Software_Mapping.md`.

#### B. Actuator Control (Output)
*   **Concept:** Converting digital commands into physical actions.
*   **Implementation:**
    *   **Relay Module:** Receives a logic HIGH (1) signal from the GPIO pin to close the circuit and turn on the water pump.
    *   **PWM (Pulse Width Modulation):** Used for variable speed control of pumps (simulated).

#### C. Memory Hierarchy & Data Flow
*   **Concept:** How data moves from sensors -> RAM -> Cache -> CPU -> Storage.
*   **Implementation:**
    *   **L1/L2 Cache:** Optimized data structures (NumPy arrays) for fast CPU processing during model training.
    *   **RAM:** Holds the active `Farm` objects and real-time sensor data.
    *   **Disk:** Stores historical CSV data and trained `.pkl` models.

#### D. Instruction Set Architecture (ISA) Simulation
*   **Concept:** The set of instructions that the processor understands.
*   **Implementation:** The system logic is designed to be compatible with ARM-based architectures (like Raspberry Pi) often used in IoT, utilizing efficient integer arithmetic where possible.

---

## 🔗 The "Integration" Factor

How do these 4 subjects work **together**?

1.  **Sensors (CAO)** collect raw data (Temperature, Moisture).
2.  **OOP Classes** (`Sensor`, `Farm`) structure this data into usable objects.
3.  **Probability Models** analyze the data to predict future weather and risks.
4.  **ML Models** (trained on historical data) predict immediate irrigation needs.
5.  **Algorithms (DAA)** take these predictions and optimize the schedule for hundreds of farms.
6.  **Actuators (CAO)** execute the final decision to turn on pumps.

---

## 📊 Summary Table

| Subject | Key Concept | Implementation in Project |
| :--- | :--- | :--- |
| **Probability** | Markov Chains | Predicting weather transitions (Sunny/Rainy) |
| **Probability** | Monte Carlo | Simulating 30-day crop risk scenarios |
| **DAA** | Greedy Algorithm | Prioritizing farms for irrigation based on need |
| **DAA** | Time Complexity | Analyzing efficiency ($O(N \log N)$) of schedulers |
| **OOP** | Encapsulation | Protecting farm data within Classes |
| **OOP** | Inheritance | `MoistureSensor` inherits from generic `Sensor` |
| **CAO** | I/O Mapping | converting Sensor Voltage $\to$ Digital Data |
| **CAO** | Actuator Control | Logic HIGH $\to$ Relay ON $\to$ Pump ON |

---

**This document serves as the definitive guide to how your project satisfies all academic requirements.**
