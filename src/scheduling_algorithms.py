"""
Design & Analysis of Algorithms (DAA) Module
Smart Farming Irrigation Scheduling

This module implements:
1. Greedy Priority Scheduler (water-deficit based)
2. Heap-based optimization for efficient scheduling
3. Time and space complexity analysis
4. Multi-farm zone prioritization
5. Dynamic programming for optimal water allocation

Algorithms:
- Greedy Scheduler: O(n log n) with priority queue
- Dynamic Programming: O(n * W) for water allocation
- Graph-based zone optimization: O(V + E)

Complexity Analysis:
- Time: O(n log n) for sorting + O(n) for scheduling = O(n log n)
- Space: O(n) for priority queue + O(n) for schedule = O(n)
"""

import heapq
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict


@dataclass(order=True)
class IrrigationTask:
    """
    Represents an irrigation task with priority
    
    Uses dataclass with order=True for heap operations
    Priority is negative for max-heap behavior (Python has min-heap)
    """
    priority: float = field(compare=True)
    farm_id: str = field(compare=False)
    water_needed: float = field(compare=False)  # liters
    duration_minutes: int = field(compare=False)
    urgency_level: str = field(compare=False)  # 'critical', 'high', 'medium', 'low'
    timestamp: datetime = field(compare=False, default_factory=datetime.now)
    
    def __post_init__(self):
        # Negate priority for max-heap behavior
        self.priority = -abs(self.priority)


@dataclass
class ScheduleEntry:
    """Represents a scheduled irrigation event"""
    farm_id: str
    start_time: datetime
    end_time: datetime
    water_allocated: float  # liters
    priority_score: float
    zone_id: Optional[str] = None


class GreedyPriorityScheduler:
    """
    Greedy algorithm for irrigation scheduling based on priority
    
    Algorithm:
    1. Calculate priority for each farm (based on water stress, crop stage, disease)
    2. Sort farms by priority (descending)
    3. Allocate water greedily to highest priority farms first
    4. Schedule irrigation events without conflicts
    
    Time Complexity: O(n log n)
    - Sorting: O(n log n)
    - Scheduling: O(n)
    
    Space Complexity: O(n)
    - Priority queue: O(n)
    - Schedule: O(n)
    """
    
    def __init__(self, total_water_budget: float, time_window_hours: int = 24):
        """
        Initialize scheduler
        
        Args:
            total_water_budget: Total water available (liters)
            time_window_hours: Scheduling time window (hours)
        """
        self.total_water_budget = total_water_budget
        self.time_window_hours = time_window_hours
        self.schedule: List[ScheduleEntry] = []
        self.water_used = 0.0
        self.farms_scheduled = set()
    
    def calculate_priority(self, farm_data: Dict) -> float:
        """
        Calculate priority score for a farm
        
        Priority factors:
        1. Water stress (40%): (optimal_moisture - current_moisture) / optimal_moisture
        2. Crop growth stage (30%): Critical stages get higher priority
        3. Disease severity (20%): Diseased crops need more care
        4. Time since last irrigation (10%): Longer time = higher priority
        
        Args:
            farm_data: Dictionary with farm attributes
            
        Returns:
            Priority score (0-100, higher = more urgent)
        """
        priority = 0.0
        
        # Factor 1: Water stress (0-40 points)
        current_moisture = farm_data.get('current_moisture', 30.0)
        optimal_moisture = farm_data.get('optimal_moisture', 35.0)
        moisture_deficit = max(0, optimal_moisture - current_moisture)
        water_stress = (moisture_deficit / optimal_moisture) * 40
        priority += water_stress
        
        # Factor 2: Crop growth stage (0-30 points)
        growth_stage = farm_data.get('growth_stage', 'Mid')
        stage_weights = {
            'Initial': 0.6,
            'Development': 0.8,
            'Mid': 1.0,  # Most critical
            'Late': 0.7
        }
        priority += stage_weights.get(growth_stage, 0.5) * 30
        
        # Factor 3: Disease severity (0-20 points)
        disease_status = farm_data.get('disease_status', 'None')
        disease_weights = {
            'None': 0.0,
            'Mild': 0.3,
            'Moderate': 0.6,
            'Severe': 1.0
        }
        priority += disease_weights.get(disease_status, 0.0) * 20
        
        # Factor 4: Time since last irrigation (0-10 points)
        hours_since_irrigation = farm_data.get('hours_since_irrigation', 24.0)
        time_factor = min(1.0, hours_since_irrigation / 48.0)
        priority += time_factor * 10
        
        return priority
    
    def schedule_greedy(self, farms_data: List[Dict]) -> List[ScheduleEntry]:
        """
        Greedy scheduling algorithm
        
        Complexity Analysis:
        - Time: O(n log n) for sorting + O(n) for iteration = O(n log n)
        - Space: O(n) for storing schedule
        
        Args:
            farms_data: List of dictionaries with farm data
            
        Returns:
            List of scheduled irrigation events
        """
        # Step 1: Calculate priorities for all farms
        # Time: O(n)
        tasks = []
        for farm in farms_data:
            priority = self.calculate_priority(farm)
            water_needed = farm.get('water_needed', 1000.0)
            duration = farm.get('duration_minutes', 30)
            
            # Determine urgency level
            if priority >= 75:
                urgency = 'critical'
            elif priority >= 50:
                urgency = 'high'
            elif priority >= 25:
                urgency = 'medium'
            else:
                urgency = 'low'
            
            task = IrrigationTask(
                priority=priority,
                farm_id=farm['farm_id'],
                water_needed=water_needed,
                duration_minutes=duration,
                urgency_level=urgency
            )
            tasks.append(task)
        
        # Step 2: Sort by priority (descending)
        # Time: O(n log n)
        tasks.sort()  # Sorts by priority (negated for max-heap)
        
        # Step 3: Greedy allocation
        # Time: O(n)
        current_time = datetime.now()
        self.schedule = []
        self.water_used = 0.0
        self.farms_scheduled = set()
        
        for task in tasks:
            # Check if we have enough water
            if self.water_used + task.water_needed > self.total_water_budget:
                continue  # Skip this farm
            
            # Check if within time window
            end_time = current_time + timedelta(minutes=task.duration_minutes)
            if (end_time - datetime.now()).total_seconds() / 3600 > self.time_window_hours:
                continue  # Skip this farm
            
            # Allocate water
            entry = ScheduleEntry(
                farm_id=task.farm_id,
                start_time=current_time,
                end_time=end_time,
                water_allocated=task.water_needed,
                priority_score=-task.priority  # Convert back to positive
            )
            
            self.schedule.append(entry)
            self.water_used += task.water_needed
            self.farms_scheduled.add(task.farm_id)
            
            # Update current time for next task
            current_time = end_time
        
        return self.schedule
    
    def get_statistics(self) -> Dict:
        """Get scheduling statistics"""
        return {
            'total_farms_scheduled': len(self.schedule),
            'total_water_used': self.water_used,
            'water_budget': self.total_water_budget,
            'water_utilization': (self.water_used / self.total_water_budget) * 100,
            'average_priority': np.mean([s.priority_score for s in self.schedule]) if self.schedule else 0,
            'time_window_hours': self.time_window_hours
        }


class HeapBasedScheduler:
    """
    Optimized scheduler using min-heap (priority queue)
    
    Advantages over greedy:
    - Dynamic priority updates: O(log n)
    - Efficient insertion/deletion: O(log n)
    - Real-time scheduling support
    
    Time Complexity: O(n log n)
    - n insertions: O(n log n)
    - n deletions: O(n log n)
    
    Space Complexity: O(n)
    - Heap storage: O(n)
    """
    
    def __init__(self, total_water_budget: float):
        self.total_water_budget = total_water_budget
        self.task_heap: List[IrrigationTask] = []
        self.schedule: List[ScheduleEntry] = []
        self.water_used = 0.0
    
    def add_task(self, task: IrrigationTask):
        """
        Add task to priority queue
        
        Time Complexity: O(log n)
        """
        heapq.heappush(self.task_heap, task)
    
    def add_tasks_bulk(self, tasks: List[IrrigationTask]):
        """
        Add multiple tasks efficiently
        
        Time Complexity: O(n log n)
        """
        for task in tasks:
            self.add_task(task)
    
    def get_next_task(self) -> Optional[IrrigationTask]:
        """
        Get highest priority task
        
        Time Complexity: O(log n)
        """
        if self.task_heap:
            return heapq.heappop(self.task_heap)
        return None
    
    def schedule_with_heap(self) -> List[ScheduleEntry]:
        """
        Schedule irrigation using heap-based approach
        
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        current_time = datetime.now()
        self.schedule = []
        self.water_used = 0.0
        
        while self.task_heap and self.water_used < self.total_water_budget:
            task = self.get_next_task()
            
            if task is None:
                break
            
            # Check water availability
            if self.water_used + task.water_needed > self.total_water_budget:
                continue
            
            # Schedule the task
            end_time = current_time + timedelta(minutes=task.duration_minutes)
            
            entry = ScheduleEntry(
                farm_id=task.farm_id,
                start_time=current_time,
                end_time=end_time,
                water_allocated=task.water_needed,
                priority_score=-task.priority
            )
            
            self.schedule.append(entry)
            self.water_used += task.water_needed
            current_time = end_time
        
        return self.schedule
    
    def update_task_priority(self, farm_id: str, new_priority: float):
        """
        Update priority of a task (requires rebuild)
        
        Time Complexity: O(n) for search + O(n log n) for rebuild = O(n log n)
        
        Note: For frequent updates, consider using a more sophisticated
        data structure like Fibonacci heap
        """
        # Find and update task
        for i, task in enumerate(self.task_heap):
            if task.farm_id == farm_id:
                self.task_heap[i].priority = -abs(new_priority)
                break
        
        # Rebuild heap
        heapq.heapify(self.task_heap)


class DynamicProgrammingWaterAllocator:
    """
    Dynamic Programming approach for optimal water allocation
    
    Problem: Given n farms with priorities and water needs, and total water W,
    maximize total priority while staying within water budget.
    
    This is a variant of the 0/1 Knapsack problem.
    
    DP Formulation:
    dp[i][w] = maximum priority using first i farms with water budget w
    
    Recurrence:
    dp[i][w] = max(
        dp[i-1][w],  # Don't irrigate farm i
        dp[i-1][w - water[i]] + priority[i]  # Irrigate farm i
    )
    
    Time Complexity: O(n * W) where W is water budget (discretized)
    Space Complexity: O(n * W)
    """
    
    def __init__(self, water_budget: float, water_unit: float = 100.0):
        """
        Initialize DP allocator
        
        Args:
            water_budget: Total water available (liters)
            water_unit: Discretization unit for DP (liters)
        """
        self.water_budget = water_budget
        self.water_unit = water_unit
        self.W = int(water_budget / water_unit)  # Discretized budget
    
    def allocate_optimal(self, farms_data: List[Dict]) -> Tuple[List[str], float, float]:
        """
        Find optimal water allocation using DP
        
        Args:
            farms_data: List of farm dictionaries
            
        Returns:
            (selected_farm_ids, total_priority, total_water_used)
        """
        n = len(farms_data)
        
        # Extract priorities and water needs
        priorities = []
        water_needs = []
        farm_ids = []
        
        for farm in farms_data:
            # Calculate priority (reuse from greedy scheduler)
            scheduler = GreedyPriorityScheduler(self.water_budget)
            priority = scheduler.calculate_priority(farm)
            priorities.append(priority)
            
            # Discretize water need
            water_needed = farm.get('water_needed', 1000.0)
            water_needed_units = int(water_needed / self.water_unit)
            water_needs.append(water_needed_units)
            
            farm_ids.append(farm['farm_id'])
        
        # DP table
        # dp[i][w] = (max_priority, selected_farms)
        dp = [[0.0 for _ in range(self.W + 1)] for _ in range(n + 1)]
        
        # Fill DP table
        # Time: O(n * W)
        for i in range(1, n + 1):
            for w in range(self.W + 1):
                # Option 1: Don't include farm i-1
                dp[i][w] = dp[i-1][w]
                
                # Option 2: Include farm i-1 (if it fits)
                if water_needs[i-1] <= w:
                    include_priority = dp[i-1][w - water_needs[i-1]] + priorities[i-1]
                    dp[i][w] = max(dp[i][w], include_priority)
        
        # Backtrack to find selected farms
        selected_farms = []
        w = self.W
        total_priority = dp[n][w]
        
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                # Farm i-1 was included
                selected_farms.append(farm_ids[i-1])
                w -= water_needs[i-1]
        
        # Calculate total water used
        total_water = sum(
            farms_data[farm_ids.index(fid)].get('water_needed', 1000.0)
            for fid in selected_farms
        )
        
        return selected_farms, total_priority, total_water
    
    def get_complexity_analysis(self, n: int) -> Dict:
        """Return complexity analysis"""
        return {
            'algorithm': 'Dynamic Programming (0/1 Knapsack)',
            'time_complexity': f'O(n * W) = O({n} * {self.W}) = O({n * self.W})',
            'space_complexity': f'O(n * W) = O({n} * {self.W}) = O({n * self.W})',
            'n_farms': n,
            'water_budget_units': self.W,
            'optimal': True,
            'note': 'Guarantees optimal solution but slower than greedy for large W'
        }


class ZoneBasedScheduler:
    """
    Graph-based zone optimization for multi-farm scheduling
    
    Farms are grouped into zones based on:
    - Geographical proximity
    - Shared irrigation infrastructure
    - Similar crop types
    
    Uses graph algorithms for zone optimization:
    - Connected components for zone identification
    - Minimum spanning tree for infrastructure planning
    
    Time Complexity: O(V + E) for graph traversal
    """
    
    def __init__(self):
        self.zones: Dict[str, List[str]] = defaultdict(list)
        self.zone_priorities: Dict[str, float] = {}
    
    def create_zones_by_proximity(self, farms_data: List[Dict], max_distance_km: float = 10.0):
        """
        Create zones based on geographical proximity
        
        Uses simple clustering based on distance threshold
        
        Time Complexity: O(n²) for pairwise distances
        """
        from scipy.spatial.distance import pdist, squareform
        
        # Extract coordinates
        coords = []
        farm_ids = []
        for farm in farms_data:
            coords.append([farm.get('latitude', 0), farm.get('longitude', 0)])
            farm_ids.append(farm['farm_id'])
        
        coords = np.array(coords)
        
        # Calculate pairwise distances (Haversine approximation)
        # For simplicity, using Euclidean distance
        distances = squareform(pdist(coords))
        
        # Simple clustering: assign to zones
        visited = set()
        zone_id = 0
        
        for i, farm_id in enumerate(farm_ids):
            if farm_id in visited:
                continue
            
            # Create new zone
            zone_name = f"ZONE_{zone_id:03d}"
            self.zones[zone_name].append(farm_id)
            visited.add(farm_id)
            
            # Add nearby farms to same zone
            for j, other_farm_id in enumerate(farm_ids):
                if other_farm_id not in visited and distances[i][j] < max_distance_km:
                    self.zones[zone_name].append(other_farm_id)
                    visited.add(other_farm_id)
            
            zone_id += 1
    
    def calculate_zone_priorities(self, farms_data: List[Dict]):
        """
        Calculate priority for each zone
        
        Zone priority = average of farm priorities in zone
        """
        scheduler = GreedyPriorityScheduler(total_water_budget=100000)
        
        farm_priorities = {
            farm['farm_id']: scheduler.calculate_priority(farm)
            for farm in farms_data
        }
        
        for zone_name, farm_ids in self.zones.items():
            priorities = [farm_priorities[fid] for fid in farm_ids]
            self.zone_priorities[zone_name] = np.mean(priorities)
    
    def schedule_by_zones(self, farms_data: List[Dict], water_budget: float) -> List[ScheduleEntry]:
        """
        Schedule irrigation by zones (highest priority zones first)
        
        Time Complexity: O(z log z + n) where z = number of zones
        """
        # Sort zones by priority
        sorted_zones = sorted(
            self.zone_priorities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Schedule farms zone by zone
        scheduler = GreedyPriorityScheduler(water_budget)
        all_schedules = []
        
        for zone_name, zone_priority in sorted_zones:
            farm_ids_in_zone = self.zones[zone_name]
            zone_farms = [f for f in farms_data if f['farm_id'] in farm_ids_in_zone]
            
            zone_schedule = scheduler.schedule_greedy(zone_farms)
            
            # Add zone information
            for entry in zone_schedule:
                entry.zone_id = zone_name
                all_schedules.append(entry)
        
        return all_schedules


def demonstrate_scheduling_algorithms():
    """Demonstrate all scheduling algorithms"""
    print("=== DAA Scheduling Algorithms Demo ===\n")
    
    # Generate sample farm data
    np.random.seed(42)
    farms_data = []
    for i in range(20):
        farm = {
            'farm_id': f'FARM{i:03d}',
            'current_moisture': np.random.uniform(15, 45),
            'optimal_moisture': 35.0,
            'growth_stage': np.random.choice(['Initial', 'Development', 'Mid', 'Late']),
            'disease_status': np.random.choice(['None', 'Mild', 'Moderate', 'Severe']),
            'hours_since_irrigation': np.random.uniform(0, 72),
            'water_needed': np.random.uniform(500, 2000),
            'duration_minutes': np.random.randint(20, 60),
            'latitude': np.random.uniform(20, 35),
            'longitude': np.random.uniform(70, 90)
        }
        farms_data.append(farm)
    
    # 1. Greedy Priority Scheduler
    print("1. Greedy Priority Scheduler")
    print("-" * 60)
    
    greedy = GreedyPriorityScheduler(total_water_budget=20000, time_window_hours=24)
    schedule = greedy.schedule_greedy(farms_data)
    
    stats = greedy.get_statistics()
    print(f"Farms scheduled: {stats['total_farms_scheduled']}")
    print(f"Water used: {stats['total_water_used']:.2f} L ({stats['water_utilization']:.1f}%)")
    print(f"Average priority: {stats['average_priority']:.2f}")
    
    print(f"\nTop 5 scheduled farms:")
    for entry in schedule[:5]:
        print(f"  {entry.farm_id}: Priority={entry.priority_score:.2f}, "
              f"Water={entry.water_allocated:.0f}L, "
              f"Time={entry.start_time.strftime('%H:%M')}-{entry.end_time.strftime('%H:%M')}")
    
    print(f"\nComplexity Analysis:")
    print(f"  Time: O(n log n) = O({len(farms_data)} log {len(farms_data)}) ≈ O({len(farms_data) * np.log2(len(farms_data)):.0f})")
    print(f"  Space: O(n) = O({len(farms_data)})")
    
    # 2. Heap-Based Scheduler
    print("\n2. Heap-Based Scheduler")
    print("-" * 60)
    
    heap_scheduler = HeapBasedScheduler(total_water_budget=20000)
    
    # Create tasks
    tasks = []
    for farm in farms_data:
        priority = greedy.calculate_priority(farm)
        task = IrrigationTask(
            priority=priority,
            farm_id=farm['farm_id'],
            water_needed=farm['water_needed'],
            duration_minutes=farm['duration_minutes'],
            urgency_level='high' if priority > 50 else 'medium'
        )
        tasks.append(task)
    
    heap_scheduler.add_tasks_bulk(tasks)
    heap_schedule = heap_scheduler.schedule_with_heap()
    
    print(f"Farms scheduled: {len(heap_schedule)}")
    print(f"Water used: {heap_scheduler.water_used:.2f} L")
    print(f"\nHeap operations:")
    print(f"  Insert: O(log n)")
    print(f"  Extract-max: O(log n)")
    print(f"  Total: O(n log n)")
    
    # 3. Dynamic Programming Allocator
    print("\n3. Dynamic Programming Optimal Allocator")
    print("-" * 60)
    
    dp_allocator = DynamicProgrammingWaterAllocator(water_budget=20000, water_unit=100)
    selected_farms, total_priority, total_water = dp_allocator.allocate_optimal(farms_data[:10])  # Use subset for speed
    
    print(f"Optimal selection: {len(selected_farms)} farms")
    print(f"Total priority: {total_priority:.2f}")
    print(f"Total water: {total_water:.2f} L")
    print(f"Selected farms: {', '.join(selected_farms[:5])}...")
    
    complexity = dp_allocator.get_complexity_analysis(10)
    print(f"\nComplexity:")
    print(f"  Time: {complexity['time_complexity']}")
    print(f"  Space: {complexity['space_complexity']}")
    print(f"  Optimal: {complexity['optimal']}")
    
    # 4. Zone-Based Scheduler
    print("\n4. Zone-Based Scheduler")
    print("-" * 60)
    
    zone_scheduler = ZoneBasedScheduler()
    zone_scheduler.create_zones_by_proximity(farms_data, max_distance_km=5.0)
    zone_scheduler.calculate_zone_priorities(farms_data)
    
    print(f"Zones created: {len(zone_scheduler.zones)}")
    for zone_name, farm_ids in list(zone_scheduler.zones.items())[:3]:
        priority = zone_scheduler.zone_priorities[zone_name]
        print(f"  {zone_name}: {len(farm_ids)} farms, Priority={priority:.2f}")
    
    zone_schedule = zone_scheduler.schedule_by_zones(farms_data, water_budget=20000)
    print(f"\nTotal farms scheduled: {len(zone_schedule)}")
    
    print("\n=== Scheduling Algorithms Demo Complete ===")


if __name__ == "__main__":
    demonstrate_scheduling_algorithms()
