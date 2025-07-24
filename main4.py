import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from typing import List, Tuple, Dict
import copy

class UAV:
    """
    UAV Model - Section 3.1, Equation (3)
    Represents a single UAV with autopilot models
    """
    def __init__(self, x: float, y: float, h: float, vxy: float = 10.0, psi: float = 0.0, lam: float = 0.0):
        # Position and states
        self.x = x # x position
        self.y = y # y position  
        self.h = h # altitude
        self.vxy = vxy # horizontal airspeed
        self.psi = psi # yaw angle
        self.lam = lam # altitude rate
        
        # Autopilot time constants
        self.tau_v = 1.0 # Mach-hold autopilot time constant
        self.tau_psi = 0.75 # heading-hold autopilot time constant
        self.tau_h = 1.0 # altitude-hold autopilot time constant
        self.tau_lam = 0.3 # altitude rate time constant
        
        # Control limits from Equation (4)
        self.vxy_max = 15.0
        self.vxy_min = 5.0
        self.lam_max = 5.0
        self.lam_min = -5.0
        self.n_max = 10.0 # maximum lateral overload
        self.g = 10.0 # gravitational acceleration
        
        # Control inputs
        self.vxy_c = vxy
        self.psi_c = psi
        self.h_c = h
        
        # Influence weight vector for MPIO
        self.w = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        
    def update_dynamics(self, dt: float):
        """
        Update UAV dynamics using Equation (3)
        Step 11: Calculate UAV states at next time
        """
        # Equation (3) - UAV model dynamics
        self.x += self.vxy * np.cos(self.psi) * dt
        self.y += self.vxy * np.sin(self.psi) * dt
        self.h += self.lam * dt
        
        # Autopilot dynamics
        self.vxy += (1/self.tau_v) * (self.vxy_c - self.vxy) * dt
        self.psi += (1/self.tau_psi) * (self.psi_c - self.psi) * dt
        self.lam += (1/self.tau_h) * (self.h_c - self.h) * dt - (1/self.tau_lam) * self.lam * dt
        
        # Apply constraints from Equation (4)
        self.vxy = np.clip(self.vxy, self.vxy_min, self.vxy_max)
        self.lam = np.clip(self.lam, self.lam_min, self.lam_max)
        
        # Psi rate constraint
        max_psi_rate = self.n_max * self.g / self.vxy
        psi_rate = (self.psi_c - self.psi) / self.tau_psi
        if abs(psi_rate) > max_psi_rate:
            self.psi_c = self.psi + np.sign(psi_rate) * max_psi_rate * self.tau_psi

class Obstacle:
    """
    Obstacle representation for environment
    """
    def __init__(self, x: float, y: float, radius: float, height: float = 100.0):
        self.x = x
        self.y = y  
        self.radius = radius
        self.height = height

class FlockingController:
    """
    Self-propelled flocking model - Section 3.2
    Implements Equations (5)-(8)
    """
    def __init__(self):
        # Flocking parameters
        self.R1_comm = 20.0 # horizontal communication range
        self.R1_lim = 2.0 # maximum range of collision avoidance
        self.R_desire = 10.0 # desired flocking distance
        
        # Control strengths
        self.Kf = 0.1 # flocking control strength
        self.Kc = 100000.0 # collision avoidance strength
        self.Ka_vn = 0.1 # velocity alignment strength
        self.Ka_he = 30.0 # altitude alignment strength
        self.Kv_e = 10.0 # expected vertical velocity strength
        
        # Expected states
        self.h_e = 50.0 # expected altitude
        self.v_e = np.array([10.0, 0.0, 0.0]) # expected velocity
        
    def calculate_flocking_velocity(self, uav: UAV, neighbors: List[UAV]) -> np.ndarray:
        """
        Calculate desired flocking velocity using Equation (5)
        Step 3: Calculate desired flocking acceleration
        """
        vf = np.zeros(3)
        
        # Horizontal flocking control (k=1,2 in Equation 5)
        f_f = self.flocking_geometry_control(uav, neighbors) # Equation (6)
        f_c = self.collision_avoidance_control(uav, neighbors) # Equation (7)
        f_a_vn = self.alignment_control(uav, neighbors) # Equation (8)
        
        vf[0] = f_f[0] + f_c[0] + f_a_vn[0]
        vf[1] = f_f[1] + f_c[1] + f_a_vn[1]
        
        # Vertical flocking control (k=3 in Equation 5)
        vf[2] = self.Ka_he * (self.h_e - uav.h) + self.Kv_e * (self.v_e[2] - uav.lam)
        
        return vf
    
    def flocking_geometry_control(self, uav: UAV, neighbors: List[UAV]) -> np.ndarray:
        """
        Flocking geometry control component - Equation (6)
        """
        f_f = np.zeros(2)
        
        for neighbor in neighbors:
            d_ij = self.horizontal_distance(uav, neighbor)
            if d_ij <= self.R1_comm and d_ij > 0:
                w_ji = 1.0 # influence weight (simplified)
                
                direction = np.array([neighbor.x - uav.x, neighbor.y - uav.y])
                factor = 1 - (self.R_desire / d_ij)**2
                
                f_f += self.Kf * w_ji * direction * factor
        
        return f_f
    
    def collision_avoidance_control(self, uav: UAV, neighbors: List[UAV]) -> np.ndarray:
        """
        Collision avoidance control component - Equation (7)
        """
        f_c = np.zeros(2)
        
        for neighbor in neighbors:
            d_ij = self.horizontal_distance(uav, neighbor)
            if d_ij <= self.R1_lim and d_ij > 0:
                direction = np.array([uav.x - neighbor.x, uav.y - neighbor.y])
                direction_norm = np.linalg.norm(direction)
                
                if direction_norm > 0:
                    repulsion = (1/direction_norm - 1/self.R1_lim)**2
                    f_c += self.Kc * repulsion * (direction / direction_norm)
        
        return f_c
    
    def alignment_control(self, uav: UAV, neighbors: List[UAV]) -> np.ndarray:
        """
        Velocity alignment control component - Equation (8)
        """
        f_a_vn = np.zeros(2)
        
        for neighbor in neighbors:
            d_ij = self.horizontal_distance(uav, neighbor)
            if d_ij <= self.R1_comm:
                w_ji = 1.0 # influence weight (simplified)
                
                neighbor_vel = np.array([neighbor.vxy * np.cos(neighbor.psi),
                                       neighbor.vxy * np.sin(neighbor.psi)])
                uav_vel = np.array([uav.vxy * np.cos(uav.psi),
                                  uav.vxy * np.sin(uav.psi)])
                
                f_a_vn += self.Ka_vn * w_ji * (neighbor_vel - uav_vel)
        
        return f_a_vn
    
    def horizontal_distance(self, uav1: UAV, uav2: UAV) -> float:
        """Calculate horizontal distance between two UAVs"""
        return np.sqrt((uav1.x - uav2.x)**2 + (uav1.y - uav2.y)**2)

class ObstacleAvoidance:
    """
    Obstacle avoidance model - Section 3.3
    Implements Equations (9)-(11)
    """
    def __init__(self):
        self.R2_comm = 105.0 # horizontal perception range
        self.R2_lim = 10.0 # minimum allowable distance to obstacles
        self.theta_lim = np.pi/2 # field of view
        
    def calculate_obstacle_avoidance_velocity(self, uav: UAV, obstacles: List[Obstacle], 
                                            expected_vel: np.ndarray) -> np.ndarray:
        """
        Calculate desired obstacle avoidance velocity
        Step 3: Calculate desired velocity for obstacle avoidance
        """
        # Step 1: Identify obstacles in attention zone
        detected_obstacles = self.detect_obstacles(uav, obstacles)
        
        if not detected_obstacles:
            # No obstacles detected, return expected velocity direction
            psi_m = np.arctan2(expected_vel[1], expected_vel[0])
            vo = np.array([np.linalg.norm(expected_vel[:2]) * np.cos(psi_m),
                          np.linalg.norm(expected_vel[:2]) * np.sin(psi_m)])
            return vo
        
        # Step 1: Identify nearest obstacle - Equation (9)
        nearest_idx = self.identify_nearest_obstacle(uav, detected_obstacles)
        
        # Step 2: Identify largest visual gap - Equation (10)
        if len(detected_obstacles) > 1:
            gap_idx = self.identify_largest_gap(uav, detected_obstacles, nearest_idx)
        else:
            gap_idx = nearest_idx
        
        # Step 3: Calculate desired velocity - Equation (11)
        theta_m = self.calculate_steering_angle(uav, detected_obstacles, nearest_idx, gap_idx, expected_vel)
        
        w_ii = 1.0 # influence weight of obstacle avoidance
        vo = np.array([w_ii * np.linalg.norm(expected_vel[:2]) * np.cos(theta_m),
                      w_ii * np.linalg.norm(expected_vel[:2]) * np.sin(theta_m)])
        
        return vo
    
    def detect_obstacles(self, uav: UAV, obstacles: List[Obstacle]) -> List[Obstacle]:
        """Detect obstacles within attention zone"""
        detected = []
        
        for obstacle in obstacles:
            distance = np.sqrt((uav.x - obstacle.x)**2 + (uav.y - obstacle.y)**2)
            
            # Check if obstacle is within perception range
            if distance <= self.R2_comm:
                # Check if obstacle is within field of view
                angle_to_obstacle = np.arctan2(obstacle.y - uav.y, obstacle.x - uav.x)
                angle_diff = abs(angle_to_obstacle - uav.psi)
                angle_diff = min(angle_diff, 2*np.pi - angle_diff) # Normalize to [0, pi]
                
                if angle_diff <= self.theta_lim:
                    detected.append(obstacle)
        
        return detected
    
    def identify_nearest_obstacle(self, uav: UAV, obstacles: List[Obstacle]) -> int:
        """Equation (9): Find index of nearest obstacle"""
        min_distance = float('inf')
        nearest_idx = 0
        
        for i, obstacle in enumerate(obstacles):
            distance = np.sqrt((uav.x - obstacle.x)**2 + (uav.y - obstacle.y)**2) - obstacle.radius
            if distance < min_distance:
                min_distance = distance
                nearest_idx = i
        
        return nearest_idx
    
    def identify_largest_gap(self, uav: UAV, obstacles: List[Obstacle], nearest_idx: int) -> int:
        """Equation (10): Find obstacle corresponding to largest visual gap"""
        # Simplified gap calculation
        max_gap = 0
        gap_idx = nearest_idx
        
        for i, obstacle in enumerate(obstacles):
            if i != nearest_idx:
                gap = self.calculate_gap(uav, obstacles[nearest_idx], obstacle)
                if gap > max_gap:
                    max_gap = gap
                    gap_idx = i
        
        return gap_idx
    
    def calculate_gap(self, uav: UAV, obs1: Obstacle, obs2: Obstacle) -> float:
        """Calculate gap between two obstacles"""
        distance = np.sqrt((obs1.x - obs2.x)**2 + (obs1.y - obs2.y)**2)
        return distance - obs1.radius - obs2.radius
    
    def calculate_steering_angle(self, uav: UAV, obstacles: List[Obstacle], 
                               nearest_idx: int, gap_idx: int, expected_vel: np.ndarray) -> float:
        """Calculate steering angle for obstacle avoidance"""
        if len(obstacles) == 1:
            # Single obstacle case
            obs = obstacles[nearest_idx]
            theta_m = np.arctan2(obs.y - uav.y, obs.x - uav.x) + np.pi # Opposite direction
        else:
            # Multiple obstacles - steer towards gap
            obs1 = obstacles[nearest_idx]
            obs2 = obstacles[gap_idx]
            
            # Calculate midpoint of gap
            mid_x = (obs1.x + obs2.x) / 2
            mid_y = (obs1.y + obs2.y) / 2
            theta_m = np.arctan2(mid_y - uav.y, mid_x - uav.x)
        
        return theta_m

class PerformanceCriteria:
    """
    Performance criteria - Section 3.4
    Implements Equations (12)-(17)
    """
    def __init__(self):
        self.f1 = 1.0 # weight for flocking geometry
        self.f2 = 1.0 # weight for velocity alignment
        
    def calculate_objectives(self, uav: UAV, neighbors: List[UAV], obstacles: List[Obstacle], 
                           expected_vel: np.ndarray, R1_comm: float, R1_lim: float, 
                           R2_lim: float, R_desire: float) -> Tuple[float, float, float, float]:
        """
        Calculate all four objective functions
        Used in Step 7: Calculate objective function value
        """
        # Detect obstacles in attention zone
        detected_obstacles = []
        for obs in obstacles:
            distance = np.sqrt((uav.x - obs.x)**2 + (uav.y - obs.y)**2)
            if distance <= 105.0: # R2_comm
                detected_obstacles.append(obs)
        
        # Cost1 - Equation (12): Passage/alignment objective
        if detected_obstacles: # A_io not empty
            projection = (uav.x * expected_vel[0] + uav.y * expected_vel[1]) / np.linalg.norm(expected_vel[:2])
            cost1 = -projection
        else: # A_io is empty
            uav_vel_x = uav.vxy * np.cos(uav.psi)
            uav_vel_y = uav.vxy * np.sin(uav.psi)
            cost1 = abs(expected_vel[0] - uav_vel_x) + abs(expected_vel[1] - uav_vel_y)
        
        # Cost2 - Equation (13): Flocking quality and velocity alignment
        cost2 = 0.0
        for neighbor in neighbors:
            d_ij = np.sqrt((uav.x - neighbor.x)**2 + (uav.y - neighbor.y)**2)
            if d_ij <= R1_comm:
                # Flocking geometry term
                geometry_term = self.f1 * abs(R_desire - d_ij)
                
                # Velocity alignment term
                neighbor_vel_x = neighbor.vxy * np.cos(neighbor.psi)
                neighbor_vel_y = neighbor.vxy * np.sin(neighbor.psi)
                uav_vel_x = uav.vxy * np.cos(uav.psi)
                uav_vel_y = uav.vxy * np.sin(uav.psi)
                
                velocity_term = self.f2 * (abs(neighbor_vel_x - uav_vel_x) + abs(neighbor_vel_y - uav_vel_y))
                
                cost2 += geometry_term + velocity_term
        
        # Cost3 - Equation (14): Obstacle avoidance constraint (hard)
        cost3 = 0.0
        for obs in obstacles:
            d_io = np.sqrt((uav.x - obs.x)**2 + (uav.y - obs.y)**2) - obs.radius
            if d_io <= R2_lim:
                cost3 = 1.0
                break
        
        # Cost4 - Equation (15): Collision avoidance constraint (hard)
        cost4 = 0.0
        for neighbor in neighbors:
            d_ij = np.sqrt((uav.x - neighbor.x)**2 + (uav.y - neighbor.y)**2)
            if d_ij <= R1_lim:
                cost4 = 1.0
                break
        
        return cost1, cost2, cost3, cost4

class ModifiedMPIO:
    """
    Modified Multi-Objective Pigeon-Inspired Optimization - Section 4
    Implements the hierarchical learning algorithm
    """
    def __init__(self, N: int = 58, Nc3_max: int = 20, Nd: int = 2):
        # Algorithm parameters from Table 2
        self.N = N # number of pigeons
        self.Nc3_max = Nc3_max # maximum iteration
        self.Nd = Nd # reduced number of pigeons per iteration
        self.R = 0.3 # map and compass factor
        self.ft = 3.0 # transition factor
        self.pl = 0.9 # percentage of general leaders
        self.e = 0.01 # learning error
        self.sl = 2 # learning strength
        
        # Velocity bounds
        self.V_U = 0.05
        self.V_L = -0.05
        
    def optimize(self, objective_func, dimension: int) -> Tuple[np.ndarray, List[float]]:
        """
        Main optimization loop
        Steps 4-9: Modified MPIO algorithm execution
        """
        # Step 4: Initialize positions and velocities
        positions = np.random.rand(self.N, dimension) # X^1
        velocities = np.random.uniform(self.V_L, self.V_U, (self.N, dimension)) # V^1
        
        # Historical information set A
        historical_set = []
        
        current_N = self.N
        
        for nc in range(1, self.Nc3_max + 1): # Step 5: Nc iterations
            # Evaluate objective functions
            objectives = []
            for i in range(current_N):
                obj_values = objective_func(positions[i])
                objectives.append(obj_values)
            
            # Step 1: Pareto sorting and ranking
            pareto_fronts = self.pareto_sorting(objectives)
            ranks = self.assign_ranks(pareto_fronts, current_N)
            
            # Step 2: Calculate landmark - Equation (18)
            X_center = self.calculate_landmark(positions, pareto_fronts[0], current_N)
            

            # Step 2: Update historical set A
            historical_set.extend([positions[i] for i in pareto_fronts[0]])
            historical_objectives = [objective_func(pos) for pos in historical_set]
            historical_fronts = self.pareto_sorting(historical_objectives)
            
            # Keep only Pareto frontier in historical set
            historical_set = [historical_set[i] for i in historical_fronts[0]]
            
            # Select current global best position X^g
            if historical_set:
                X_g = random.choice(historical_set)
            else:
                X_g = positions[0]
            
            # Step 4: Update pigeon positions
            new_positions = np.zeros_like(positions[:current_N])
            new_velocities = np.zeros_like(velocities[:current_N])
            
            for i in range(current_N):
                if ranks[i] <= int(self.pl * current_N): # General leaders
                    # Update using Equations (19) and (20)
                    new_velocities[i] = self.update_leader_velocity(
                        velocities[i], positions[i], X_g, X_center, nc)
                    new_positions[i] = self.update_leader_position(
                        positions[i], new_velocities[i])
                else: # Ordinary followers
                    # Update using Equation (21) - hierarchical learning
                    new_positions[i] = self.hierarchical_learning(
                        positions, ranks, i, dimension)
                    new_velocities[i] = velocities[i] # Keep velocity unchanged
                
                # Step 5: Check domination
                old_obj = objective_func(positions[i])
                new_obj = objective_func(new_positions[i])
                
                if self.dominates(old_obj, new_obj):
                    new_positions[i] = positions[i]
                    new_velocities[i] = velocities[i]
            
            positions[:current_N] = new_positions
            velocities[:current_N] = new_velocities
            
            # Step 6: Reduce population size
            if nc < self.Nc3_max:
                if current_N > self.Nd:
                    # Remove worst pigeons
                    worst_indices = list(range(current_N - self.Nd, current_N))
                    positions = np.delete(positions, worst_indices, axis=0)
                    velocities = np.delete(velocities, worst_indices, axis=0)
                    current_N -= self.Nd

        # Return best solution
        final_objectives = [objective_func(positions[i]) for i in range(current_N)]
        final_fronts = self.pareto_sorting(final_objectives)
        
        if final_fronts[0]:
            best_idx = final_fronts[0][0]
            return positions[best_idx], final_objectives[best_idx]
        else:
            return positions[0], final_objectives[0]
    
    def update_leader_velocity(self, velocity: np.ndarray, position: np.ndarray, 
                              X_g: np.ndarray, X_center: np.ndarray, nc: int) -> np.ndarray:
        """
        Update velocity for general leaders - Equation (19)
        """
        rand1 = np.random.rand(len(velocity))
        rand2 = np.random.rand(len(velocity))
        
        # Equation (19)
        new_velocity = (np.exp(-self.R * nc) * velocity + 
                       rand1 * self.ft * (1 - np.log(nc) / np.log(self.Nc3_max)) * (X_g - position) +
                       rand2 * self.ft * (np.log(nc) / np.log(self.Nc3_max)) * (X_center - position))
        
        # Apply velocity bounds
        new_velocity = np.clip(new_velocity, self.V_L, self.V_U)
        
        return new_velocity
    
    def update_leader_position(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """
        Update position for general leaders - Equation (20)
        """
        new_position = position + velocity
        
        # Apply position bounds [0, 1]
        new_position = np.clip(new_position, 0.0, 1.0)
        
        return new_position
    
    def hierarchical_learning(self, positions: np.ndarray, ranks: List[int], 
                            current_idx: int, dimension: int) -> np.ndarray:
        """
        Hierarchical learning for followers - Equation (21)
        """
        new_position = positions[current_idx].copy()
        
        # Repeat learning sl times
        for _ in range(self.sl):
            # Select dimension to learn - d* = [rand * D]
            d_star = random.randint(0, dimension - 1)
            
            # Select pigeon to learn from - with higher rank (lower rank number)
            valid_teachers = [i for i in range(len(positions)) if ranks[i] < ranks[current_idx]]
            
            if valid_teachers:
                # j satisfies N_j^o = [rand * (N_i^o - 1)]
                max_teacher_rank = min(ranks[current_idx] - 1, len(valid_teachers) - 1)
                teacher_rank = random.randint(0, max_teacher_rank)
                
                # Find pigeon with this rank
                teacher_idx = next((i for i, r in enumerate(ranks) if r == teacher_rank), valid_teachers[0])
                
                # Equation (21): Learn from teacher
                new_position[d_star] = positions[teacher_idx][d_star] + self.e * np.random.rand()
        
        # Apply bounds
        new_position = np.clip(new_position, 0.0, 1.0)
        
        return new_position
    
    def calculate_landmark(self, positions: np.ndarray, pareto_front: List[int], 
                          current_N: int) -> np.ndarray:
        """
        Calculate landmark position - Equation (18)
        """
        if not pareto_front:
            return np.mean(positions[:current_N], axis=0)
        
        landmark_positions = positions[pareto_front]
        return np.mean(landmark_positions, axis=0)
    
    def pareto_sorting(self, objectives: List[Tuple[float, float, float, float]]) -> List[List[int]]:
        """
        Pareto sorting for multi-objective optimization
        Returns list of fronts, each containing indices of solutions
        """
        n = len(objectives)
        fronts = []
        domination_count = [0] * n # Number of solutions that dominate solution i
        dominated_solutions = [[] for _ in range(n)] # Solutions dominated by solution i
        # Calculate domination relationships
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self.dominates(objectives[i], objectives[j]):
                        dominated_solutions[i].append(j)
                    elif self.dominates(objectives[j], objectives[i]):
                        domination_count[i] += 1
        
        # Find first front
        first_front = []
        for i in range(n):
            if domination_count[i] == 0:
                first_front.append(i)
        
        fronts.append(first_front)
        
        # Find subsequent fronts
        while len(fronts[-1]) > 0:
            next_front = []
            for i in fronts[-1]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            
            if next_front:
                fronts.append(next_front)
            else:
                break
        
        return fronts[:-1] if not fronts[-1] else fronts
    
    def dominates(self, obj1: Tuple[float, float, float, float], 
                 obj2: Tuple[float, float, float, float]) -> bool:
        """
        Check if obj1 dominates obj2 in multi-objective sense
        For objectives: minimize Cost1, Cost2; Cost3=0, Cost4=0 (hard constraints)
        """
        # Hard constraints must be satisfied
        if obj1[2] > 0 or obj1[3] > 0: # obj1 violates hard constraints
            return False
        if obj2[2] > 0 or obj2[3] > 0: # obj2 violates hard constraints
            return obj1[2] == 0 and obj1[3] == 0 # obj1 is feasible

        # Both are feasible, compare soft objectives
        better_in_any = False
        for i in range(2): # Only compare first two objectives (soft constraints)
            if obj1[i] > obj2[i]: # obj1 is worse in this objective
                return False
            elif obj1[i] < obj2[i]: # obj1 is better in this objective
                better_in_any = True
        
        return better_in_any
    
    def assign_ranks(self, fronts: List[List[int]], current_N: int) -> List[int]:
        """
        Assign ranks to all solutions based on Pareto fronts
        """
        ranks = [0] * current_N
        
        for front_idx, front in enumerate(fronts):
            for solution_idx in front:
                ranks[solution_idx] = front_idx
        
        return ranks

class UAVFlockingSystem:
    """
    Main UAV Flocking Control System - Section 5
    Implements the complete distributed flocking algorithm
    """
    def __init__(self, initial_positions: List[Tuple[float, float, float]], 
                 obstacles: List[Tuple[float, float, float]]):
        # Step 1: Initialize UAV states
        self.uavs = []
        for i, (x, y, h) in enumerate(initial_positions):
            uav = UAV(x, y, h)
            self.uavs.append(uav)
        
        # Initialize obstacles
        self.obstacles = []
        for x, y, r in obstacles:
            self.obstacles.append(Obstacle(x, y, r))
        
        # Initialize controllers
        self.flocking_controller = FlockingController()
        self.obstacle_avoidance = ObstacleAvoidance()
        self.performance_criteria = PerformanceCriteria()
        self.mpio = ModifiedMPIO()
        
        # Simulation parameters
        self.dt = 0.5 # sampling time
        self.T_max = 49.5 # maximum simulation time
        self.u_lim = 0.25 # dead zone threshold
        self.V_lim_xy_c = 0.25 # allowable control error
        self.psi_lim_c = 0.1 # allowable control error
        
        # Store trajectory data
        self.trajectories = {i: {'x': [], 'y': [], 'h': [], 't': []} for i in range(len(self.uavs))}
        
    def run_simulation(self) -> Dict:
        """
        Main simulation loop - Algorithm steps from Section 5
        """
        t = 0.0 # Step 1: Current simulation time t = 0
        
        while t < self.T_max: # Step 13: Check time completion
            # Step 2: i = 1 (iterate through all UAVs)
            for i, uav in enumerate(self.uavs):
                # Get neighbors (all other UAVs)
                neighbors = [other_uav for j, other_uav in enumerate(self.uavs) if j != i]
                
                # Step 3: Calculate desired flocking acceleration and obstacle avoidance
                vf_i = self.flocking_controller.calculate_flocking_velocity(uav, neighbors)
                vo_i = self.obstacle_avoidance.calculate_obstacle_avoidance_velocity(
                    uav, self.obstacles, self.flocking_controller.v_e)
                
                # Step 4-10: Use Modified MPIO to find optimal influence weight vector
                def objective_function(w):
                    """Objective function for MPIO optimization"""
                    # Temporarily set influence weights
                    original_w = uav.w.copy()
                    uav.w = w
                    
                    # Calculate objectives using Equations (12)-(15)
                    cost1, cost2, cost3, cost4 = self.performance_criteria.calculate_objectives(
                        uav, neighbors, self.obstacles, self.flocking_controller.v_e,
                        self.flocking_controller.R1_comm, self.flocking_controller.R1_lim,
                        self.obstacle_avoidance.R2_lim, self.flocking_controller.R_desire)
                    
                    # Restore original weights
                    uav.w = original_w
                    
                    return (cost1, cost2, cost3, cost4)
                
                # Run MPIO optimization
                optimal_w, best_objectives = self.mpio.optimize(objective_function, len(uav.w))
                
                # Step 10: Update influence weight vector
                uav.w = optimal_w
                
                # Step 11: Calculate control inputs using Equation (25)
                u_i = self.calculate_control_input(vf_i, vo_i, uav)
                
                # Apply dead zone threshold
                for j in range(3):
                    if abs(u_i[j]) < self.u_lim:
                        u_i[j] = 0.0
                
                # Calculate autopilot control inputs using Equation (26)
                self.calculate_autopilot_inputs(uav, u_i)
                
                # Store trajectory data
                self.trajectories[i]['x'].append(uav.x)
                self.trajectories[i]['y'].append(uav.y)
                self.trajectories[i]['h'].append(uav.h)
                self.trajectories[i]['t'].append(t)

            # Step 11: Update UAV dynamics for all UAVs
            for uav in self.uavs:
                uav.update_dynamics(self.dt)
            
            # Step 13: Advance time
            t += self.dt
            print(f"Time: {t:.1f}s")

        # Step 14: Output results
        return self.trajectories
    
    def calculate_control_input(self, vf_i: np.ndarray, vo_i: np.ndarray, uav: UAV) -> np.ndarray:
        """
        Calculate control input using Equation (25)
        Step 11: Calculate control input u_i
        """
        # Current velocity
        v_i = np.array([
            uav.vxy * np.cos(uav.psi), # x-velocity
            uav.vxy * np.sin(uav.psi), # y-velocity
            uav.lam # z-velocity (altitude rate)
        ])
        
        # Equation (25)
        u_i = np.array([
            vf_i[0] + (vo_i[0] - v_i[0]),
            vf_i[1] + (vo_i[1] - v_i[1]),
            vf_i[2]
        ])
        
        return u_i
    
    def calculate_autopilot_inputs(self, uav: UAV, u_i: np.ndarray):
        """
        Calculate autopilot control inputs using Equation (26)
        Step 11: Calculate control inputs of UAV autopilots
        """
        # Equation (26)
        V_xy_c = uav.tau_v * (u_i[0] * np.cos(uav.psi) + u_i[1] * np.sin(uav.psi)) + uav.vxy
        psi_c = (uav.tau_psi / uav.vxy) * (u_i[1] * np.cos(uav.psi) - u_i[0] * np.sin(uav.psi)) + uav.psi
        h_c = uav.h + (uav.tau_h / uav.tau_lam) * uav.lam + uav.tau_h * u_i[2]
        
        # Apply control error thresholds
        expected_vxy = np.linalg.norm(self.flocking_controller.v_e[:2])
        if abs(V_xy_c - expected_vxy) < self.V_lim_xy_c:
            V_xy_c = expected_vxy
        
        psi_m = np.arctan2(self.flocking_controller.v_e[1], self.flocking_controller.v_e[0])
        if abs(psi_c - psi_m) < self.psi_lim_c:
            psi_c = psi_m
        
        # Update control inputs
        uav.vxy_c = V_xy_c
        uav.psi_c = psi_c
        uav.h_c = h_c
    
    def plot_results(self):
        """Plot simulation results"""
        fig = plt.figure(figsize=(15, 12))
        
        # 3D trajectory plot
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, uav in enumerate(self.uavs):
            traj = self.trajectories[i]
            ax1.plot(traj['x'], traj['y'], traj['h'], color=colors[i % len(colors)], 
                    label=f'UAV {i+1}', linewidth=2)
            ax1.scatter(traj['x'][0], traj['y'][0], traj['h'][0], 
                       color=colors[i % len(colors)], marker='o', s=100)
        
        # Plot obstacles
        for obs in self.obstacles:
            theta = np.linspace(0, 2*np.pi, 100)
            x_circle = obs.x + obs.radius * np.cos(theta)
            y_circle = obs.y + obs.radius * np.sin(theta)
            z_bottom = np.zeros_like(theta)
            z_top = np.full_like(theta, obs.height)
            
            ax1.plot(x_circle, y_circle, z_bottom, 'k-', linewidth=2)
            ax1.plot(x_circle, y_circle, z_top, 'k-', linewidth=2)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Altitude (m)')
        ax1.set_title('3D UAV Trajectories')
        ax1.legend()
        
        # Top-down view
        ax2 = fig.add_subplot(2, 3, 2)
        for i, uav in enumerate(self.uavs):
            traj = self.trajectories[i]
            ax2.plot(traj['x'], traj['y'], color=colors[i % len(colors)], 
                    label=f'UAV {i+1}', linewidth=2)
            ax2.scatter(traj['x'][0], traj['y'][0], color=colors[i % len(colors)], 
                       marker='o', s=100)
        
        # Plot obstacles (top view)
        for obs in self.obstacles:
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
        for i, uav in enumerate(self.uavs):
            traj = self.trajectories[i]
            ax3.plot(traj['t'], traj['h'], color=colors[i % len(colors)], 
                    label=f'UAV {i+1}', linewidth=2)
        
        ax3.axhline(y=self.flocking_controller.h_e, color='black', linestyle='--', 
                   label='Expected altitude')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Altitude (m)')
        ax3.set_title('Altitude vs Time')
        ax3.legend()
        ax3.grid(True)
        
        # Horizontal speed vs time
        ax4 = fig.add_subplot(2, 3, 4)
        for i, uav in enumerate(self.uavs):
            traj = self.trajectories[i]
            speeds = []
            for j in range(len(traj['t'])):
                # Reconstruct speed from stored data (simplified)
                if j < len(traj['t']) - 1:
                    dx = traj['x'][j+1] - traj['x'][j]
                    dy = traj['y'][j+1] - traj['y'][j]
                    dt = traj['t'][j+1] - traj['t'][j]
                    if dt > 0:
                        speed = np.sqrt(dx**2 + dy**2) / dt
                    else:
                        speed = 10.0 # default
                else:
                    speed = speeds[-1] if speeds else 10.0
                speeds.append(speed)
            
            ax4.plot(traj['t'], speeds, color=colors[i % len(colors)], 
                    label=f'UAV {i+1}', linewidth=2)
        
        expected_speed = np.linalg.norm(self.flocking_controller.v_e[:2])
        ax4.axhline(y=expected_speed, color='black', linestyle='--', 
                   label='Expected speed')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Horizontal Speed (m/s)')
        ax4.set_title('Horizontal Speed vs Time')
        ax4.legend()
        ax4.grid(True)
        
        # Distance between UAVs
        ax5 = fig.add_subplot(2, 3, 5)
        for i in range(len(self.uavs)):
            for j in range(i+1, len(self.uavs)):
                distances = []
                traj_i = self.trajectories[i]
                traj_j = self.trajectories[j]
                
                for k in range(len(traj_i['t'])):
                    dist = np.sqrt((traj_i['x'][k] - traj_j['x'][k])**2 + 
                                 (traj_i['y'][k] - traj_j['y'][k])**2)
                    distances.append(dist)
                
                ax5.plot(traj_i['t'], distances, label=f'UAV {i+1} - UAV {j+1}', linewidth=2)
        
        ax5.axhline(y=self.flocking_controller.R_desire, color='black', linestyle='--', 
                   label='Desired distance')
        ax5.axhline(y=self.flocking_controller.R1_lim, color='red', linestyle='--', 
                   label='Collision limit')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Distance (m)')
        ax5.set_title('Inter-UAV Distances')
        ax5.legend()
        ax5.grid(True)
        
        # Minimum distance to obstacles
        ax6 = fig.add_subplot(2, 3, 6)
        for i, uav in enumerate(self.uavs):
            traj = self.trajectories[i]
            min_distances = []
            
            for j in range(len(traj['t'])):
                min_dist = float('inf')
                for obs in self.obstacles:
                    dist = np.sqrt((traj['x'][j] - obs.x)**2 + (traj['y'][j] - obs.y)**2) - obs.radius
                    min_dist = min(min_dist, dist)
                min_distances.append(max(min_dist, 0)) # Avoid negative distances
            
            ax6.plot(traj['t'], min_distances, color=colors[i % len(colors)], 
                    label=f'UAV {i+1}', linewidth=2)
        
        ax6.axhline(y=self.obstacle_avoidance.R2_lim, color='red', linestyle='--', 
                   label='Safety limit')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Min Distance to Obstacles (m)')
        ax6.set_title('Minimum Distance to Obstacles')
        ax6.legend()
        ax6.grid(True)
        
        plt.tight_layout()
        plt.show()

# Example usage and simulation
def main():
    """
    Main function to run the UAV flocking simulation
    Uses parameters from Table 1 in the paper
    """
    # Initial UAV positions from Table 1
    initial_positions = [
        (14.6929, 107.3676, 68.1682), # UAV 1
        (21.2809, 116.6406, 34.8423), # UAV 2
        (20.3911, 113.6529, 24.6351), # UAV 3
        (3.5699, 108.9509, 96.377), # UAV 4
        (10.2116, 111.558, 30.1431), # UAV 5
    ]
    
    # Obstacle positions from Table 1
    obstacles = [
        (120, 120, 5), # Obstacle 1
        (240, 75, 5), # Obstacle 2
        (350, 40, 5), # Obstacle 3
        (240, 155, 5), # Obstacle 4
        (360, 110, 5), # Obstacle 5
        (350, 180, 5), # Obstacle 6
    ]
    
    print("Initializing UAV Flocking System...")
    print("=================================")
    print("Algorithm Steps Mapping:")
    print("Step 1: Initialize UAV states - UAVFlockingSystem.__init__()")
    print("Step 2-3: Calculate flocking/obstacle avoidance - FlockingController, ObstacleAvoidance")
    print("Step 4-10: Modified MPIO optimization - ModifiedMPIO.optimize()")
    print("Step 11: Calculate control inputs - Equations (25)-(26)")
    print("Step 12-13: Iterate through UAVs and time - Main simulation loop")
    print("Step 14: Output results - Plotting and analysis")
    print("=================================\n")
    
    # Create and run simulation
    system = UAVFlockingSystem(initial_positions, obstacles)
    
    print("Starting simulation...")
    trajectories = system.run_simulation()
    
    print("Simulation completed!")
    print("Plotting results...")
    
    # Plot results
    system.plot_results()
    
    # Print final statistics
    print("\nFinal Statistics:")
    print("================")
    for i, uav in enumerate(system.uavs):
        print(f"UAV {i+1} final position: ({uav.x:.2f}, {uav.y:.2f}, {uav.h:.2f})")
        
        # Check if UAV avoided all obstacles
        min_dist_to_obstacles = float('inf')
        for obs in system.obstacles:
            dist = np.sqrt((uav.x - obs.x)**2 + (uav.y - obs.y)**2) - obs.radius
            min_dist_to_obstacles = min(min_dist_to_obstacles, dist)
        
        safety_status = "SAFE" if min_dist_to_obstacles > system.obstacle_avoidance.R2_lim else "DANGER"
        print(f" Min distance to obstacles: {min_dist_to_obstacles:.2f}m ({safety_status})")
    
    # Check final formation
    distances = []
    for i in range(len(system.uavs)):
        for j in range(i+1, len(system.uavs)):
            dist = np.sqrt((system.uavs[i].x - system.uavs[j].x)**2 + 
                          (system.uavs[i].y - system.uavs[j].y)**2)
            distances.append(dist)
    
    avg_distance = np.mean(distances)
    print(f"\nAverage inter-UAV distance: {avg_distance:.2f}m")
    print(f"Desired formation distance: {system.flocking_controller.R_desire:.2f}m")
    print(f"Formation quality: {abs(avg_distance - system.flocking_controller.R_desire):.2f}m deviation")

if __name__ == "__main__":
    main()
