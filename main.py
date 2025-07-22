import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from typing import List, Tuple, Dict
from dataclasses import dataclass
from scipy.spatial.distance import euclidean
import math

@dataclass
class UAVState:
    """UAV state representation"""
    x: float = 0.0
    y: float = 0.0
    h: float = 50.0
    vxy: float = 10.0
    psi: float = 0.0
    lambda_: float = 0.0

@dataclass
class Obstacle:
    """Obstacle representation"""
    x: float
    y: float
    radius: float
    height: float = 100.0

class UAVModel:
    """UAV dynamic model (Section 3.1)"""
    
    def __init__(self):
        # Autopilot time constants
        self.tau_v = 1.0      # Mach-hold autopilot
        self.tau_psi = 0.75   # Heading-hold autopilot  
        self.tau_h = 1.0      # Altitude-hold autopilot
        self.tau_lambda = 0.3 # Altitude rate autopilot
        
        # Constraints
        self.vxy_max = 15.0
        self.vxy_min = 5.0
        self.lambda_max = 5.0
        self.lambda_min = -5.0
        self.n_max = 10.0  # max lateral overload
        self.g = 10.0      # gravitational acceleration
        
    def update_state(self, state: UAVState, control_inputs: Dict, dt: float) -> UAVState:
        """Update UAV state using Equation 3"""
        vxy_c = control_inputs.get('vxy_c', state.vxy)
        psi_c = control_inputs.get('psi_c', state.psi)
        h_c = control_inputs.get('h_c', state.h)
        
        # State derivatives
        x_dot = state.vxy * np.cos(state.psi)
        y_dot = state.vxy * np.sin(state.psi)
        h_dot = state.lambda_
        vxy_dot = (vxy_c - state.vxy) / self.tau_v
        psi_dot = (psi_c - state.psi) / self.tau_psi
        lambda_dot = (h_c - state.h) / self.tau_h - state.lambda_ / self.tau_lambda
        
        # Apply constraints
        psi_dot = np.clip(psi_dot, -self.n_max * self.g / state.vxy, 
                         self.n_max * self.g / state.vxy)
        
        # Update state
        new_state = UAVState(
            x=state.x + x_dot * dt,
            y=state.y + y_dot * dt,
            h=state.h + h_dot * dt,
            vxy=np.clip(state.vxy + vxy_dot * dt, self.vxy_min, self.vxy_max),
            psi=state.psi + psi_dot * dt,
            lambda_=np.clip(state.lambda_ + lambda_dot * dt, self.lambda_min, self.lambda_max)
        )
        
        return new_state

class FlockingController:
    """Self-propelled flocking model (Section 3.2)"""
    
    def __init__(self):
        # Parameters
        self.R1_comm = 20.0      # communication range
        self.R1_lim = 2.0        # collision avoidance range
        self.R_desire = 10.0     # desired flocking distance
        self.Kf = 0.1           # flocking control strength
        self.Kc = 100000.0      # collision avoidance strength
        self.Ka_vn = 0.1        # velocity alignment strength
        self.Ka_he = 30.0       # altitude alignment strength
        self.Kve = 10.0         # vertical velocity alignment strength
        
    def calculate_flocking_acceleration(self, uav_state: UAVState, 
                                      neighbors: List[UAVState],
                                      expected_velocity: np.ndarray,
                                      expected_altitude: float) -> np.ndarray:
        """Calculate desired flocking acceleration (Equation 5)"""
        
        # Get neighbors within communication range
        nearby_neighbors = []
        for neighbor in neighbors:
            dist = np.sqrt((uav_state.x - neighbor.x)**2 + (uav_state.y - neighbor.y)**2)
            if dist <= self.R1_comm and dist > 0:
                nearby_neighbors.append((neighbor, dist))
        
        # Horizontal control components
        ff = self._flocking_geometry_control(uav_state, nearby_neighbors)
        fc = self._collision_avoidance_control(uav_state, nearby_neighbors)
        fa_vn = self._alignment_control(uav_state, nearby_neighbors)
        
        horizontal_acc = ff + fc + fa_vn
        
        # Vertical control
        vertical_acc = (self.Ka_he * (expected_altitude - uav_state.h) + 
                       self.Kve * (expected_velocity[2] - uav_state.lambda_))
        
        return np.array([horizontal_acc[0], horizontal_acc[1], vertical_acc])
    
    def _flocking_geometry_control(self, uav_state: UAVState, 
                                 neighbors: List[Tuple[UAVState, float]]) -> np.ndarray:
        """Flocking geometry control (Equation 6)"""
        force = np.array([0.0, 0.0])
        
        for neighbor, dist in neighbors:
            if dist <= self.R1_comm:
                direction = np.array([neighbor.x - uav_state.x, neighbor.y - uav_state.y])
                weight = 1.0  # w_ji weight (simplified)
                force += weight * direction * (1 - (self.R_desire / dist)**2)
        
        return self.Kf * force
    
    def _collision_avoidance_control(self, uav_state: UAVState,
                                   neighbors: List[Tuple[UAVState, float]]) -> np.ndarray:
        """Collision avoidance control (Equation 7)"""
        force = np.array([0.0, 0.0])
        
        for neighbor, dist in neighbors:
            if dist <= self.R1_lim:
                direction = np.array([uav_state.x - neighbor.x, uav_state.y - neighbor.y])
                if dist > 0:
                    force += (1/dist - 1/self.R1_lim)**2 * direction / dist
        
        return self.Kc * force
    
    def _alignment_control(self, uav_state: UAVState,
                         neighbors: List[Tuple[UAVState, float]]) -> np.ndarray:
        """Velocity alignment control (Equation 8)"""
        force = np.array([0.0, 0.0])
        
        if len(neighbors) > 0:
            avg_velocity = np.array([0.0, 0.0])
            for neighbor, dist in neighbors:
                if dist <= self.R1_comm:
                    neighbor_vel = np.array([neighbor.vxy * np.cos(neighbor.psi),
                                           neighbor.vxy * np.sin(neighbor.psi)])
                    avg_velocity += neighbor_vel
            
            if len(neighbors) > 0:
                avg_velocity /= len(neighbors)
                current_vel = np.array([uav_state.vxy * np.cos(uav_state.psi),
                                      uav_state.vxy * np.sin(uav_state.psi)])
                force = avg_velocity - current_vel
        
        return self.Ka_vn * force

class ObstacleAvoidance:
    """Obstacle avoidance model (Section 3.3)"""
    
    def __init__(self):
        self.R2_comm = 105.0    # perception range
        self.R2_lim = 10.0      # minimum allowable distance
        self.theta_lim = np.pi/2  # field of view
        
    def calculate_obstacle_avoidance_velocity(self, uav_state: UAVState,
                                            obstacles: List[Obstacle],
                                            expected_velocity: np.ndarray) -> np.ndarray:
        """Calculate desired obstacle avoidance velocity (Section 3.3)"""
        
        # Step 1: Identify obstacles in attention zone
        detected_obstacles = self._detect_obstacles(uav_state, obstacles, expected_velocity)
        
        if not detected_obstacles:
            # No obstacles, return expected velocity direction
            psi_m = np.arctan2(expected_velocity[1], expected_velocity[0])
            speed = np.linalg.norm(expected_velocity[:2])
            return np.array([speed * np.cos(psi_m), speed * np.sin(psi_m)])
        
        # Step 2: Find nearest obstacle (Equation 9)
        nearest_obstacle_idx = self._find_nearest_obstacle(uav_state, detected_obstacles)
        
        # Step 3: Calculate steering angle
        if len(detected_obstacles) == 1:
            # Single obstacle case
            obstacle = detected_obstacles[0]
            theta_m = np.arctan2(obstacle.y - uav_state.y, obstacle.x - uav_state.x)
            # Adjust to avoid obstacle (simplified)
            theta_m += np.pi/4  # Turn away from obstacle
        else:
            # Multiple obstacles - find largest gap
            theta_m = self._find_largest_gap(uav_state, detected_obstacles, expected_velocity)
        
        # Calculate desired velocity (Equation 11)
        speed = np.linalg.norm(expected_velocity[:2])
        return np.array([speed * np.cos(theta_m), speed * np.sin(theta_m)])
    
    def _detect_obstacles(self, uav_state: UAVState, obstacles: List[Obstacle],
                         expected_velocity: np.ndarray) -> List[Obstacle]:
        """Detect obstacles in attention zone"""
        detected = []
        
        # Current heading
        current_heading = np.arctan2(expected_velocity[1], expected_velocity[0])
        
        for obstacle in obstacles:
            # Distance to obstacle
            dist = np.sqrt((obstacle.x - uav_state.x)**2 + (obstacle.y - uav_state.y)**2)
            
            # Angle to obstacle
            angle_to_obstacle = np.arctan2(obstacle.y - uav_state.y, obstacle.x - uav_state.x)
            angle_diff = abs(angle_to_obstacle - current_heading)
            
            # Normalize angle difference
            while angle_diff > np.pi:
                angle_diff -= 2*np.pi
            angle_diff = abs(angle_diff)
            
            # Check if in attention zone
            if (dist <= self.R2_comm + obstacle.radius + self.R2_lim and angle_diff <= self.theta_lim):
                detected.append(obstacle)
        
        return detected
    
    def _find_nearest_obstacle(self, uav_state: UAVState, obstacles: List[Obstacle]) -> int:
        """Find nearest obstacle index (Equation 9)"""
        min_dist = float('inf')
        nearest_idx = 0
        
        for i, obstacle in enumerate(obstacles):
            dist = np.sqrt((obstacle.x - uav_state.x)**2 + (obstacle.y - uav_state.y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        return nearest_idx
    
    def _find_largest_gap(self, uav_state: UAVState, obstacles: List[Obstacle],
                         expected_velocity: np.ndarray) -> float:
        """Find largest gap between obstacles"""
        # Simplified gap finding - return direction that maximizes clearance
        best_angle = np.arctan2(expected_velocity[1], expected_velocity[0])
        max_clearance = 0
        
        # Sample different angles
        for angle in np.linspace(-np.pi, np.pi, 36):
            clearance = self._calculate_clearance(uav_state, obstacles, angle)
            if clearance > max_clearance:
                max_clearance = clearance
                best_angle = angle
        
        return best_angle
    
    def _calculate_clearance(self, uav_state: UAVState, obstacles: List[Obstacle],
                           angle: float) -> float:
        """Calculate clearance in given direction"""
        min_clearance = float('inf')
        
        for obstacle in obstacles:
            # Vector from UAV to obstacle
            to_obstacle = np.array([obstacle.x - uav_state.x, obstacle.y - uav_state.y])
            
            # Direction vector
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            # Distance to obstacle center
            dist_to_center = np.linalg.norm(to_obstacle)
            
            # Angle between direction and obstacle
            if dist_to_center > 0:
                dot_product = np.dot(to_obstacle, direction) / dist_to_center
                clearance = dist_to_center * np.sqrt(1 - dot_product**2) - obstacle.radius
                min_clearance = min(min_clearance, clearance)
        
        return max(0, min_clearance)

class CostFunctions:
    """Performance criteria (Section 3.4)"""
    
    def __init__(self):
        self.f1 = 1.0  # weight for flocking geometry alignment
        self.f2 = 1.0  # weight for velocity alignment
    
    def calculate_cost1(self, uav_state: UAVState, expected_velocity: np.ndarray,
                       detected_obstacles: List[Obstacle]) -> float:
        """First objective function (Equation 12)"""
        ve_12 = expected_velocity[:2]
        ve_norm = np.linalg.norm(ve_12)
        
        if len(detected_obstacles) > 0:  # Obstacles detected
            position = np.array([uav_state.x, uav_state.y])
            if ve_norm > 0:
                return -np.dot(position, ve_12) / ve_norm
            else:
                print("non paper")
                return 0.0
        else:  # No obstacles
            current_velocity = np.array([uav_state.vxy * np.cos(uav_state.psi),
                                       uav_state.vxy * np.sin(uav_state.psi)])
            return abs(expected_velocity[0] - current_velocity[0]) + abs(expected_velocity[1] - current_velocity[1])
    
    def calculate_cost2(self, uav_state: UAVState, neighbors: List[UAVState]) -> float:
        """Second objective function (Equation 13)"""
        cost = 0.0
        
        for neighbor in neighbors:
            dist = np.sqrt((uav_state.x - neighbor.x)**2 + (uav_state.y - neighbor.y)**2)
            if dist <= 20.0:  # R1_comm
                # Distance alignment
                distance_error = abs(10.0 - dist)  # R_desire = 10.0
                
                # Velocity alignment
                current_vel = np.array([uav_state.vxy * np.cos(uav_state.psi),
                                      uav_state.vxy * np.sin(uav_state.psi)])
                neighbor_vel = np.array([neighbor.vxy * np.cos(neighbor.psi),
                                       neighbor.vxy * np.sin(neighbor.psi)])
                velocity_error = abs(neighbor_vel[0] - current_vel[0]) + abs(neighbor_vel[1] - current_vel[1])
                
                cost += self.f1 * distance_error + self.f2 * velocity_error
        
        return cost
    
    def calculate_cost3(self, uav_state: UAVState, obstacles: List[Obstacle]) -> float:
        """Third objective function - obstacle collision (Equation 14)"""
        for obstacle in obstacles:
            dist = np.sqrt((uav_state.x - obstacle.x)**2 + (uav_state.y - obstacle.y)**2)
            if dist <= obstacle.radius + 10.0:  # R2_lim = 10.0
                return 1.0
        return 0.0
    
    def calculate_cost4(self, uav_state: UAVState, neighbors: List[UAVState]) -> float:
        """Fourth objective function - UAV collision (Equation 15)"""
        for neighbor in neighbors:
            dist = np.sqrt((uav_state.x - neighbor.x)**2 + (uav_state.y - neighbor.y)**2)
            if dist <= 2.0:  # R1_lim = 2.0
                return 1.0
        return 0.0

class ModifiedMPIO:
    """Modified Multi-objective Pigeon-Inspired Optimization (Section 4)"""
    
    def __init__(self, n_pigeons=58, max_iter=20, n_reduced=2, pl=0.9):
        self.N = n_pigeons
        self.Nc3_max = max_iter
        self.Nd = n_reduced
        self.pl = pl  # percentage of general leaders
        self.R = 0.3  # map and compass factor
        self.ft = 3.0  # transition factor
        self.e = 0.01  # learning error
        self.sl = 2    # learning strength
        self.V_upper = 0.05
        self.V_lower = -0.05
        
    def optimize(self, objective_function, dimension=5) -> Tuple[np.ndarray, List[float]]:
        """Main optimization loop"""
        # Initialize population
        current_pop_size = self.N
        positions = np.random.rand(current_pop_size, dimension)
        velocities = np.random.uniform(self.V_lower, self.V_upper, (current_pop_size, dimension))
        
        # Historical information set
        archive = []
        
        for iteration in range(self.Nc3_max):
            # Evaluate objectives
            objectives = []
            for i in range(current_pop_size):
                try:
                    obj = objective_function(positions[i])
                    objectives.append(obj)
                except Exception as e:
                    # Handle potential errors in objective function
                    objectives.append([1000.0, 1000.0])
            
            # Pareto sorting
            ranks = self._pareto_sorting(objectives)
            
            # Calculate landmark (Equation 18)
            pareto_front = [i for i, rank in enumerate(ranks) if rank == 1]
            if pareto_front:
                landmark = np.mean(positions[pareto_front], axis=0)
            else:
                landmark = np.mean(positions, axis=0)
                print("non paper1")
            
            # Update archive
            archive.extend([positions[i] for i in pareto_front])
            if archive:
                archive_objectives = [objective_function(pos) for pos in archive]
                archive_ranks = self._pareto_sorting(archive_objectives)
                archive = [pos for pos, rank in zip(archive, archive_ranks) if rank == 1]
            
            # Select global best
            if archive:
                global_best = random.choice(archive)
            else:
                print("non paper2")
                global_best = positions[0] if len(positions) > 0 else np.random.rand(dimension)
            
            # Update positions
            new_positions = np.copy(positions)
            new_velocities = np.copy(velocities)
            
            n_leaders = max(1, int(self.pl * current_pop_size))
            
            for i in range(current_pop_size):
                if i >= len(ranks):
                    print("non paper4")
                    continue
                    
                if ranks[i] <= n_leaders:  # General leader
                    # Update using map & compass + landmark operators (Equation 19-20)
                    if iteration > 0:  # Avoid log(0)
                        transition_factor1 = 1 - np.log(iteration + 1) / np.log(self.Nc3_max)
                        transition_factor2 = np.log(iteration + 1) / np.log(self.Nc3_max)
                    else:
                        transition_factor1 = 1.0
                        transition_factor2 = 0.0
                    
                    new_velocities[i] = (np.exp(-self.R * iteration) * velocities[i] +
                                       np.random.rand() * self.ft * transition_factor1 * 
                                       (global_best - positions[i]) +
                                       np.random.rand() * self.ft * transition_factor2 * 
                                       (landmark - positions[i]))
                    
                    new_velocities[i] = np.clip(new_velocities[i], self.V_lower, self.V_upper)
                    new_positions[i] = positions[i] + new_velocities[i]
                    new_positions[i] = np.clip(new_positions[i], 0, 1)
                    
                else:  # Ordinary follower
                    # Hierarchical learning (Equation 21)
                    for _ in range(self.sl):
                        # Find valid leaders (pigeons with better ranks)
                        valid_leaders = [j for j in range(current_pop_size) if ranks[j] < ranks[i]]
                        
                        if valid_leaders:
                            j = random.choice(valid_leaders)
                            d_star = random.randint(0, dimension - 1)
                            new_positions[i][d_star] = positions[j][d_star] + self.e * np.random.rand()
                            new_positions[i][d_star] = np.clip(new_positions[i][d_star], 0, 1)
                
                # Pseudocode Step 19-21: Dominance check (immediately after update)
                try:
                    old_obj = objective_function(positions[i])
                    new_obj = objective_function(new_positions[i])
                    
                    # Paper logic: If X^(Nc-1)_i ≺ X^Nc_i Then X^Nc_i = X^(Nc-1)_i
                    if self._dominates(old_obj, new_obj):  # Old dominates new
                        new_positions[i] = positions[i]     # Keep old position
                        new_velocities[i] = velocities[i]   # Keep old velocity
                except Exception:
                    # If evaluation fails, keep old position
                    print("non paper4")
                    new_positions[i] = positions[i]
                    new_velocities[i] = velocities[i]
            
            # Apply all updates after dominance checks
            positions = new_positions
            velocities = new_velocities
            
            # Reduce population (only if we have enough pigeons)
            if iteration < self.Nc3_max - 1 and current_pop_size > self.Nd:
                # Sort by rank (worst first) and remove worst pigeons
                to_remove = sorted(range(current_pop_size), key=lambda x: ranks[x], reverse=True)[:self.Nd]
                
                # Make sure we don't remove more than available
                to_remove = to_remove[:min(self.Nd, current_pop_size - 1)]
                
                if to_remove:
                    positions = np.delete(positions, to_remove, axis=0)
                    velocities = np.delete(velocities, to_remove, axis=0)
                    current_pop_size -= len(to_remove)
        
        # Return Pareto front
        if len(positions) == 0:
            return np.random.rand(dimension), []
            
        final_objectives = []
        for i in range(len(positions)):
            try:
                obj = objective_function(positions[i])
                final_objectives.append(obj)
            except Exception:
                final_objectives.append([1000.0, 1000.0])
        
        final_ranks = self._pareto_sorting(final_objectives)
        pareto_front = [positions[i] for i, rank in enumerate(final_ranks) if rank == 1]
        
        if pareto_front:
            # Select solution with minimum Cost2 (Equation 24)
            best_idx = 0
            min_cost2 = float('inf')
            for i, pos in enumerate(pareto_front):
                try:
                    obj = objective_function(pos)
                    if len(obj) > 1 and obj[1] < min_cost2:
                        min_cost2 = obj[1]
                        best_idx = i
                except Exception:
                    continue
            return pareto_front[best_idx], final_objectives
        else:
            return positions[0], final_objectives
    
    def _pareto_sorting(self, objectives: List[List[float]]) -> List[int]:
        """Pareto ranking of solutions"""
        n = len(objectives)
        ranks = [0] * n
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        
        # Find domination relationships
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self._dominates(objectives[i], objectives[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(objectives[j], objectives[i]):
                        domination_count[i] += 1
        
        # Assign ranks
        current_front = []
        for i in range(n):
            if domination_count[i] == 0:
                ranks[i] = 1
                current_front.append(i)
        
        rank = 1
        while current_front:
            next_front = []
            for i in current_front:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        ranks[j] = rank + 1
                        next_front.append(j)
            current_front = next_front
            rank += 1
        
        return ranks
    
    def _dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        """Check if obj1 dominates obj2 (for minimization)"""
        better_in_any = False
        for i in range(len(obj1)):
            if obj1[i] > obj2[i]:
                return False
            elif obj1[i] < obj2[i]:
                better_in_any = True
        return better_in_any

class UAVFlockingSystem:
    """Main UAV flocking system (Section 5)"""
    
    def __init__(self, n_uavs=5):
        self.n_uavs = n_uavs
        self.uav_model = UAVModel()
        self.flocking_controller = FlockingController()
        self.obstacle_avoidance = ObstacleAvoidance()
        self.cost_functions = CostFunctions()
        self.mpio = ModifiedMPIO()
        
        # Expected parameters
        self.expected_velocity = np.array([10.0, 0.0, 0.0])  # [vx, vy, vz]
        self.expected_altitude = 50.0
        
        # Initialize UAVs
        self.uav_states = self._initialize_uavs()
        
        # Initialize obstacles (from Table 1)
        self.obstacles = [
            Obstacle(120, 120, 5),
            Obstacle(240, 75, 5),
            Obstacle(350, 40, 5),
            Obstacle(240, 155, 5),
            Obstacle(360, 110, 5),
            Obstacle(350, 180, 5)
        ]
        
    def _initialize_uavs(self) -> List[UAVState]:
        """Initialize UAV states (from Table 1)"""
        initial_positions = [
            (14.6929, 107.3676, 68.1682),
            (21.2809, 116.6406, 34.8423),
            (20.3911, 113.6529, 24.6351),
            (3.5699, 108.9509, 96.377),
            (10.2116, 111.558, 30.1431)
        ]
        
        states = []
        for i in range(self.n_uavs):
            if i < len(initial_positions):
                x, y, h = initial_positions[i]
            else:
                x, y, h = np.random.uniform(0, 50), np.random.uniform(100, 150), np.random.uniform(30, 70)
            
            states.append(UAVState(x=x, y=y, h=h, vxy=10.0, psi=0.0, lambda_=0.0))
        
        return states
    
    def simulate(self, max_time=50.0, dt=0.5) -> Dict:
        """Main simulation loop"""
        time_steps = int(max_time / dt)
        
        # Storage for results
        trajectory = {i: {'x': [], 'y': [], 'h': [], 'vxy': [], 'psi': [], 'lambda': []} 
                     for i in range(self.n_uavs)}
        cost_history = {'cost1': [], 'cost2': [], 'cost3': [], 'cost4': []}
        
        for step in range(time_steps):
            current_time = step * dt
            
            # Update each UAV
            for i in range(self.n_uavs):
                # Get neighbors (all other UAVs)
                neighbors = [state for j, state in enumerate(self.uav_states) if j != i]
                
                # Detect obstacles
                detected_obstacles = self.obstacle_avoidance._detect_obstacles(
                    self.uav_states[i], self.obstacles, self.expected_velocity)
                
                # Define objective function for MPIO
                def objective_function(weights):
                    try:
                        # Apply weights to control components (simplified for stability)
                        w_f, w_c, w_a, w_o, w_i = weights
                        
                        # Calculate costs with current weights
                        cost1 = self.cost_functions.calculate_cost1(
                            self.uav_states[i], self.expected_velocity, detected_obstacles)
                        cost2 = self.cost_functions.calculate_cost2(self.uav_states[i], neighbors)
                        cost3 = self.cost_functions.calculate_cost3(self.uav_states[i], self.obstacles)
                        cost4 = self.cost_functions.calculate_cost4(self.uav_states[i], neighbors)
                        
                        # Hard constraints
                        if cost3 > 0 or cost4 > 0:
                            return [1000.0, 1000.0]  # Penalty for constraint violation
                        
                        # Apply weights to modify cost functions
                        weighted_cost1 = cost1 * w_f
                        weighted_cost2 = cost2 * w_c
                        
                        return [weighted_cost1, weighted_cost2]
                    except Exception:
                        return [1000.0, 1000.0]
                
                # Optimize weights using Modified MPIO (less frequently for performance)
                if step % 10 == 0 and step > 0:  # Optimize every 10 steps after initial steps
                    try:
                        # Use smaller MPIO for computational efficiency
                        small_mpio = ModifiedMPIO(n_pigeons=20, max_iter=5, n_reduced=1, pl=0.8)
                        optimal_weights, _ = small_mpio.optimize(objective_function, dimension=5)
                    except Exception:
                        optimal_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # Default weights
                else:
                    optimal_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # Default weights
                
                # Calculate flocking acceleration
                try:
                    flocking_acc = self.flocking_controller.calculate_flocking_acceleration(
                        self.uav_states[i], neighbors, self.expected_velocity, self.expected_altitude)
                except Exception:
                    flocking_acc = np.array([0.0, 0.0, 0.0])
                
                # Calculate obstacle avoidance velocity
                try:
                    obstacle_vel = self.obstacle_avoidance.calculate_obstacle_avoidance_velocity(
                        self.uav_states[i], self.obstacles, self.expected_velocity)
                except Exception:
                    obstacle_vel = self.expected_velocity[:2]
                
                # Combine control inputs (Equation 25)
                current_vel = np.array([self.uav_states[i].vxy * np.cos(self.uav_states[i].psi),
                                      self.uav_states[i].vxy * np.sin(self.uav_states[i].psi)])
                
                # Apply weights safely
                try:
                    weight_factor = min(1.0, max(0.1, optimal_weights[4])) if len(optimal_weights) > 4 else 1.0
                    control_input = flocking_acc[:2] + weight_factor * (obstacle_vel - current_vel)
                except Exception:
                    control_input = flocking_acc[:2]
                
                # Convert to autopilot commands (Equation 26)
                try:
                    vxy_c = (self.uav_model.tau_v * 
                            (control_input[0] * np.cos(self.uav_states[i].psi) + 
                             control_input[1] * np.sin(self.uav_states[i].psi)) + 
                            self.uav_states[i].vxy)
                    
                    if self.uav_states[i].vxy > 0:
                        psi_c = (self.uav_model.tau_psi / self.uav_states[i].vxy * 
                                (control_input[1] * np.cos(self.uav_states[i].psi) - 
                                 control_input[0] * np.sin(self.uav_states[i].psi)) + 
                                self.uav_states[i].psi)
                    else:
                        psi_c = self.uav_states[i].psi
                    
                    h_c = (self.uav_states[i].h + 
                          self.uav_model.tau_h / self.uav_model.tau_lambda * self.uav_states[i].lambda_ + 
                          self.uav_model.tau_h * flocking_acc[2])
                except Exception:
                    vxy_c = self.uav_states[i].vxy
                    psi_c = self.uav_states[i].psi
                    h_c = self.uav_states[i].h
                
                # Apply dead zone thresholds
                if abs(vxy_c - np.linalg.norm(self.expected_velocity[:2])) < 0.25:
                    vxy_c = np.linalg.norm(self.expected_velocity[:2])
                if abs(psi_c - np.arctan2(self.expected_velocity[1], self.expected_velocity[0])) < 0.1:
                    psi_c = np.arctan2(self.expected_velocity[1], self.expected_velocity[0])
                
                # Update UAV state
                control_inputs = {'vxy_c': vxy_c, 'psi_c': psi_c, 'h_c': h_c}
                try:
                    self.uav_states[i] = self.uav_model.update_state(self.uav_states[i], control_inputs, dt)
                except Exception:
                    # Keep current state if update fails
                    pass
                
                # Store trajectory
                trajectory[i]['x'].append(self.uav_states[i].x)
                trajectory[i]['y'].append(self.uav_states[i].y)
                trajectory[i]['h'].append(self.uav_states[i].h)
                trajectory[i]['vxy'].append(self.uav_states[i].vxy)
                trajectory[i]['psi'].append(self.uav_states[i].psi)
                trajectory[i]['lambda'].append(self.uav_states[i].lambda_)
            
            # Calculate and store cost functions
            total_costs = [0.0, 0.0, 0.0, 0.0]
            for i in range(self.n_uavs):
                try:
                    neighbors = [state for j, state in enumerate(self.uav_states) if j != i]
                    detected_obstacles = self.obstacle_avoidance._detect_obstacles(
                        self.uav_states[i], self.obstacles, self.expected_velocity)
                    
                    costs = [
                        self.cost_functions.calculate_cost1(self.uav_states[i], self.expected_velocity, detected_obstacles),
                        self.cost_functions.calculate_cost2(self.uav_states[i], neighbors),
                        self.cost_functions.calculate_cost3(self.uav_states[i], self.obstacles),
                        self.cost_functions.calculate_cost4(self.uav_states[i], neighbors)
                    ]
                    
                    for j in range(4):
                        total_costs[j] += costs[j]
                except Exception:
                    # If cost calculation fails, add penalty
                    total_costs[0] += 100.0
                    total_costs[1] += 100.0
            
            cost_history['cost1'].append(total_costs[0])
            cost_history['cost2'].append(total_costs[1])
            cost_history['cost3'].append(total_costs[2])
            cost_history['cost4'].append(total_costs[3])
            
            if step % 10 == 0:
                print(f"Time: {current_time:.1f}s, Costs: {[f'{c:.2f}' for c in total_costs]}")
        
        return {
            'trajectory': trajectory,
            'costs': cost_history,
            'obstacles': self.obstacles,
            'time_vector': np.arange(0, max_time, dt)
        }
    
    def plot_results(self, results: Dict):
        """Plot simulation results"""
        fig = plt.figure(figsize=(15, 12))
        
        # 3D trajectory plot
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        colors = ['b', 'r', 'g', 'm', 'c']
        for i in range(self.n_uavs):
            traj = results['trajectory'][i]
            ax1.plot(traj['x'], traj['y'], traj['h'], colors[i % len(colors)], 
                    label=f'UAV {i+1}', linewidth=2)
        
        # Plot obstacles
        for obs in results['obstacles']:
            theta = np.linspace(0, 2*np.pi, 100)
            x_circle = obs.x + obs.radius * np.cos(theta)
            y_circle = obs.y + obs.radius * np.sin(theta)
            z_circle = np.zeros_like(x_circle)
            ax1.plot(x_circle, y_circle, z_circle, 'k-', linewidth=2)
            ax1.plot(x_circle, y_circle, z_circle + obs.height, 'k-', linewidth=2)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Altitude (m)')
        ax1.set_title('3D UAV Trajectories')
        ax1.legend()
        
        # Top-down view
        ax2 = fig.add_subplot(2, 3, 2)
        for i in range(self.n_uavs):
            traj = results['trajectory'][i]
            ax2.plot(traj['x'], traj['y'], colors[i % len(colors)], 
                    label=f'UAV {i+1}', linewidth=2)
        
        # Plot obstacles
        for obs in results['obstacles']:
            circle = plt.Circle((obs.x, obs.y), obs.radius, fill=False, color='black', linewidth=2)
            ax2.add_patch(circle)
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Top-down View')
        ax2.legend()
        ax2.grid(True)
        ax2.axis('equal')
        
        # Altitude vs time
        ax3 = fig.add_subplot(2, 3, 3)
        for i in range(self.n_uavs):
            traj = results['trajectory'][i]
            ax3.plot(results['time_vector'], traj['h'], colors[i % len(colors)], 
                    label=f'UAV {i+1}', linewidth=2)
        ax3.axhline(y=self.expected_altitude, color='k', linestyle='--', label='Expected')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Altitude (m)')
        ax3.set_title('Altitude vs Time')
        ax3.legend()
        ax3.grid(True)
        
        # Horizontal airspeed vs time
        ax4 = fig.add_subplot(2, 3, 4)
        for i in range(self.n_uavs):
            traj = results['trajectory'][i]
            ax4.plot(results['time_vector'], traj['vxy'], colors[i % len(colors)], 
                    label=f'UAV {i+1}', linewidth=2)
        ax4.axhline(y=np.linalg.norm(self.expected_velocity[:2]), color='k', linestyle='--', label='Expected')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Horizontal Speed (m/s)')
        ax4.set_title('Horizontal Airspeed vs Time')
        ax4.legend()
        ax4.grid(True)
        
        # Cost functions vs time
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(results['time_vector'], results['costs']['cost1'], 'b-', label='Cost 1', linewidth=2)
        ax5.plot(results['time_vector'], results['costs']['cost2'], 'r-', label='Cost 2', linewidth=2)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Cost Value')
        ax5.set_title('Soft Constraints (Cost 1 & 2)')
        ax5.legend()
        ax5.grid(True)
        
        # Hard constraints vs time
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.plot(results['time_vector'], results['costs']['cost3'], 'g-', label='Cost 3 (Obstacle)', linewidth=2)
        ax6.plot(results['time_vector'], results['costs']['cost4'], 'm-', label='Cost 4 (Collision)', linewidth=2)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Violation Count')
        ax6.set_title('Hard Constraints (Cost 3 & 4)')
        ax6.legend()
        ax6.grid(True)
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    print("Initializing UAV Flocking System with Modified MPIO...")
    try:
        system = UAVFlockingSystem(n_uavs=5)
        
        print("Starting simulation...")
        results = system.simulate(max_time=50.0, dt=0.1)  # Shorter simulation for testing
        
        print("Plotting results...")
        system.plot_results(results)
        
        print("Simulation completed successfully!")
        
    except Exception as e:
        print(f"Error occurred during simulation: {e}")
        print("Try running with reduced parameters or check your Python environment.")

