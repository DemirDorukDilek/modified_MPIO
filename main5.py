from dataclasses import dataclass
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

class NSGA2Sorter:
    @staticmethod
    def _pareto_sorting(objectives: List[List[float]]) -> List[int]:
        """Pareto ranking of solutions"""
        n = len(objectives)
        ranks = [0] * n
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if NSGA2Sorter._dominates(objectives[i], objectives[j]):
                        dominated_solutions[i].append(j)
                    elif NSGA2Sorter._dominates(objectives[j], objectives[i]):
                        domination_count[i] += 1
        
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
    
    @staticmethod
    def _dominates(obj1: List[float], obj2: List[float]) -> bool:
        """Check if obj1 dominates obj2 (for minimization)"""
        better_in_any = False
        for i in range(len(obj1)):
            if obj1[i] > obj2[i]:
                return False
            elif obj1[i] < obj2[i]:
                better_in_any = True
        return better_in_any
    
    @staticmethod
    def _calculate_crowding_distance(objectives: List[List[float]], 
                                   front_indices: List[int]) -> List[float]:
        """Calculate crowding distance for solutions in the same front"""
        n = len(front_indices)
        if n <= 2:
            return [float('inf')] * n
        
        distances = [0.0] * n
        n_objectives = len(objectives[0])
        
        for obj_idx in range(n_objectives):
            sorted_indices = sorted(range(n),
                                  key=lambda i: objectives[front_indices[i]][obj_idx])
            
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            obj_min = objectives[front_indices[sorted_indices[0]]][obj_idx]
            obj_max = objectives[front_indices[sorted_indices[-1]]][obj_idx]
            obj_range = obj_max - obj_min
            
            if obj_range > 0:
                for i in range(1, n - 1):
                    idx = sorted_indices[i]
                    prev_idx = sorted_indices[i - 1]
                    next_idx = sorted_indices[i + 1]
                    
                    prev_val = objectives[front_indices[prev_idx]][obj_idx]
                    next_val = objectives[front_indices[next_idx]][obj_idx]
                    
                    distances[idx] += (next_val - prev_val) / obj_range
        
        return distances
    
    @staticmethod
    def _get_fronts(objectives: List[List[float]], ranks: List[int]) -> List[List[int]]:
        """Group solutions by their Pareto rank"""
        max_rank = max(ranks)
        fronts = [[] for _ in range(max_rank)]
        
        for i, rank in enumerate(ranks):
            fronts[rank - 1].append(i)
        
        return fronts
    
    @staticmethod
    def nsga2_sort(objectives: List[List[float]]) -> Tuple[List[int], List[int], List[float]]:
        """Complete NSGA-II sorting"""
        ranks = NSGA2Sorter._pareto_sorting(objectives)
        fronts = NSGA2Sorter._get_fronts(objectives, ranks)
        
        idxs, ranks_out, dists = [], [], []
        
        for front_idx, front in enumerate(fronts):
            if not front:
                continue
            
            distances = NSGA2Sorter._calculate_crowding_distance(objectives, front)
            
            for i, sol_idx in enumerate(front):
                rank = front_idx + 1
                crowding_dist = distances[i]
                idxs.append(sol_idx)
                ranks_out.append(rank)
                dists.append(crowding_dist)
        
        return idxs, ranks_out, dists

class UAV:
    def __init__(self, xyz, Vxy, psi, lambda_):
        self.xyz = xyz
        self.Vxy = Vxy
        self.psi = psi
        self.lambda_ = lambda_

    @property
    def xy(self):
        return self.xyz[:2]
    
    @property
    def x(self): 
        return self.xyz[0]
    
    @property
    def y(self): 
        return self.xyz[1]
    
    @property
    def h(self):
        return self.xyz[2]

class Obstacle:
    def __init__(self, xy, radius, height):
        self.xy = xy
        self.radius = radius
        self.height = height

    @property
    def x(self): 
        return self.xy[0]
    
    @property
    def y(self): 
        return self.xy[1]

class UAV_model:
    def __init__(self, Vxy_min=5.0, Vxy_max=15.0, lambda_min=-5.0, lambda_max=5.0,
                 tau_v=1.0, tau_psi=0.75, tau_h=1.0, tau_lambda=0.3, n_max=10, g=10):
        self.Vxy_min = Vxy_min
        self.Vxy_max = Vxy_max
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.tau_v = tau_v
        self.tau_psi = tau_psi
        self.tau_h = tau_h
        self.tau_lambda = tau_lambda
        self.n_max = n_max
        self.g = g
    
    def update_state(self, uav: UAV, control_inputs: dict, dt: float):
        Vxy_c = control_inputs.get('Vxy_c', uav.Vxy)
        psi_c = control_inputs.get('psi_c', uav.psi)
        h_c = control_inputs.get('h_c', uav.h)

        # Autopilots (Equation 3)
        x_dot = uav.Vxy * np.cos(uav.psi)
        y_dot = uav.Vxy * np.sin(uav.psi)
        h_dot = uav.lambda_
        Vxy_dot = (Vxy_c - uav.Vxy) / self.tau_v
        psi_dot = (psi_c - uav.psi) / self.tau_psi
        lambda_dot = (h_c - uav.h) / self.tau_h - uav.lambda_ / self.tau_lambda

        # Apply constraints (Equation 4)
        psi_clip_limit = self.n_max * self.g / uav.Vxy
        psi_dot = np.clip(psi_dot, -psi_clip_limit, psi_clip_limit)

        new_state = UAV(
            np.array([uav.x + x_dot * dt, uav.y + y_dot * dt, uav.h + h_dot * dt]),
            np.clip(uav.Vxy + Vxy_dot * dt, self.Vxy_min, self.Vxy_max),
            uav.psi + psi_dot * dt,
            np.clip(uav.lambda_ + lambda_dot * dt, self.lambda_min, self.lambda_max)
        )

        return new_state

class FlockingModel:
    def __init__(self, R1_comm=20.0, Rlim_1=2.0, R_desire=10.0, Kf=0.1, Kc=100000.0, 
                 Ka_vn=0.1, Ka_he=30.0, Kv_e=10.0):
        self.Rcomm_1 = R1_comm
        self.Rlim_1 = Rlim_1
        self.R_desire = R_desire
        self.Kf = Kf
        self.Kc = Kc
        self.Ka_vn = Ka_vn
        self.Ka_he = Ka_he
        self.Kv_e = Kv_e
    
    def calculate_acc(self, uav: UAV, uavs: List[UAV], weights: List[float], ve: np.ndarray, he: float):
        # Initialize forces (Equations 6, 7, 8)
        force_f = np.zeros(2)
        force_c = np.zeros(2)
        force_a_vn = np.zeros(2)
        
        for idx, o_uav in enumerate(uavs):
            if uav is o_uav: 
                continue
                
            dist = np.linalg.norm(uav.xy - o_uav.xy)
            
            # Flocking control and alignment (within communication range)
            if dist <= self.Rcomm_1:
                # Flocking geometry control (Equation 6)
                w_ji = weights[idx] if idx < len(weights) else 1.0
                force_f += w_ji * (o_uav.xy - uav.xy) * (1 - (self.R_desire / dist)**2)
                
                # Alignment control (Equation 8)
                o_uav_vel = np.array([o_uav.Vxy * np.cos(o_uav.psi), o_uav.Vxy * np.sin(o_uav.psi)])
                uav_vel = np.array([uav.Vxy * np.cos(uav.psi), uav.Vxy * np.sin(uav.psi)])
                force_a_vn += w_ji * (o_uav_vel - uav_vel)
            
            # Collision avoidance (within limit range)
            if dist <= self.Rlim_1 and dist > 0:
                # Collision avoidance control (Equation 7)
                direction = (uav.xy - o_uav.xy) / dist
                force_c += (1/dist - 1/self.Rlim_1)**2 * direction

        # Combine horizontal forces
        xy_force = self.Kf * force_f + self.Kc * force_c + self.Ka_vn * force_a_vn
        
        # Vertical control (Equation 5)
        h_force = self.Ka_he * (he - uav.h) + self.Kv_e * (ve[2] - uav.lambda_)
        
        return np.array([xy_force[0], xy_force[1], h_force])

class ObstacleAvoidanceModel:
    def __init__(self, R2_comm=105.0, R2_lim=10.0, theta_lim=np.pi/2):
        self.R2_comm = R2_comm
        self.R2_lim = R2_lim
        self.theta_lim = theta_lim
    
    def get_attention_obstacles(self, uav: UAV, obstacles: List[Obstacle], ve: np.ndarray) -> List[Obstacle]:
        """Get obstacles within attention zone"""
        ve_direction = np.arctan2(ve[1], ve[0])
        attention_obstacles = []
        
        for obstacle in obstacles:
            # Distance from UAV to obstacle center
            dist = np.linalg.norm(uav.xy - obstacle.xy)
            
            # Check if within perception range
            if dist > self.R2_comm:
                continue
            
            # Check if within field of view
            obstacle_direction = np.arctan2(obstacle.y - uav.y, obstacle.x - uav.x)
            angle_diff = abs(obstacle_direction - ve_direction)
            if angle_diff > np.pi:
                angle_diff = 2*np.pi - angle_diff
            
            if angle_diff <= self.theta_lim:
                attention_obstacles.append(obstacle)
        
        return attention_obstacles
    
    def calculate_velocity(self, uav: UAV, obstacles: List[Obstacle], ve: np.ndarray, weight: float):
        """Calculate desired obstacle avoidance velocity (Section 3.3)"""
        attention_obstacles = self.get_attention_obstacles(uav, obstacles, ve[:2])
        
        if not attention_obstacles:
            # No obstacles, use expected velocity
            return ve[:2].copy()
        
        # Find nearest obstacle (Equation 9)
        nearest_idx = self.get_nearest_obstacle_idx(uav, attention_obstacles)
        
        if len(attention_obstacles) == 1:
            # Single obstacle case
            obstacle = attention_obstacles[nearest_idx]
            direction = np.arctan2(obstacle.y - uav.y, obstacle.x - uav.x)
            return np.linalg.norm(ve[:2]) * np.array([np.cos(direction), np.sin(direction)]) * weight
        else:
            # Multiple obstacles - find largest gap
            gap_direction = self.find_largest_gap(uav, attention_obstacles, nearest_idx)
            return np.linalg.norm(ve[:2]) * gap_direction * weight
    
    def get_nearest_obstacle_idx(self, uav: UAV, obstacles: List[Obstacle]) -> int:
        """Find nearest obstacle index (Equation 9)"""
        min_dist = float('inf')
        nearest_idx = 0
        
        for i, obstacle in enumerate(obstacles):
            dist = np.linalg.norm(uav.xy - obstacle.xy) - obstacle.radius
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        return nearest_idx
    
    def find_largest_gap(self, uav: UAV, obstacles: List[Obstacle], nearest_idx: int) -> np.ndarray:
        """Find direction of largest gap between obstacles"""
        # Simplified gap finding - in practice this would be more complex
        obstacle1 = obstacles[nearest_idx]
        
        max_gap = 0
        best_direction = np.array([1.0, 0.0])
        
        for i, obstacle2 in enumerate(obstacles):
            if i == nearest_idx:
                continue
            
            # Calculate gap between obstacles
            dir1 = np.arctan2(obstacle1.y - uav.y, obstacle1.x - uav.x)
            dir2 = np.arctan2(obstacle2.y - uav.y, obstacle2.x - uav.x)
            
            gap = abs(dir2 - dir1)
            if gap > np.pi:
                gap = 2*np.pi - gap
            
            if gap > max_gap:
                max_gap = gap
                avg_dir = (dir1 + dir2) / 2
                best_direction = np.array([np.cos(avg_dir), np.sin(avg_dir)])
        
        return best_direction

class PerformanceCriteria:
    def __init__(self, f1=1.0, f2=1.0, R_desire=10.0, R1_lim=2.0, R2_lim=10.0):
        self.f1 = f1
        self.f2 = f2
        self.R_desire = R_desire
        self.R1_lim = R1_lim
        self.R2_lim = R2_lim

    def cost1(self, uav: UAV, ve: np.ndarray, attention_obstacles: List[Obstacle]):
        """First objective function (Equation 12)"""
        if len(attention_obstacles) == 0:
            # No obstacles - alignment with expected velocity
            uav_vel = np.array([uav.Vxy * np.cos(uav.psi), uav.Vxy * np.sin(uav.psi)])
            return abs(ve[0] - uav_vel[0]) + abs(ve[1] - uav_vel[1])
        else:
            # Obstacles present - projection on expected velocity direction
            return -np.dot(uav.xy, ve[:2]) / np.linalg.norm(ve[:2])
    
    def cost2(self, uav: UAV, uavs: List[UAV]):
        """Second objective function (Equation 13)"""
        cost2 = 0
        for o_uav in uavs:
            if uav is o_uav: 
                continue
            
            uav_vel = np.array([uav.Vxy * np.cos(uav.psi), uav.Vxy * np.sin(uav.psi)])
            o_uav_vel = np.array([o_uav.Vxy * np.cos(o_uav.psi), o_uav.Vxy * np.sin(o_uav.psi)])
            
            dist = np.linalg.norm(uav.xy - o_uav.xy)
            cost2 += (self.f1 * abs(self.R_desire - dist) + 
                     self.f2 * (abs(o_uav_vel[0] - uav_vel[0]) + abs(o_uav_vel[1] - uav_vel[1])))
        
        return cost2
    
    def cost3(self, uav: UAV, obstacles: List[Obstacle]):
        """Third objective function - obstacle collision (Equation 14)"""
        for obstacle in obstacles:
            if np.linalg.norm(uav.xy - obstacle.xy) <= self.R2_lim + obstacle.radius:
                return 1
        return 0
    
    def cost4(self, uav: UAV, uavs: List[UAV]):
        """Fourth objective function - UAV collision (Equation 15)"""
        for o_uav in uavs:
            if uav is o_uav: 
                continue
            if np.linalg.norm(uav.xy - o_uav.xy) <= self.R1_lim:
                return 1
        return 0

class ModifiedMPIO:
    """Modified Multi-objective Pigeon-Inspired Optimization"""
    
    def __init__(self, n_pigeons=20, max_iter=20, pl=0.9, nd=2, R=0.3, ft=3, 
                 learning_error=0.01, learning_strength=2):
        self.n_pigeons = n_pigeons
        self.max_iter = max_iter
        self.pl = pl # Percentage of general leaders
        self.nd = nd # Reduced number of pigeons at each iteration
        self.R = R # Map and compass factor
        self.ft = ft # Transition factor
        self.learning_error = learning_error
        self.learning_strength = learning_strength
        self.archive = []
    
    def optimize(self, objective_func, dimension, bounds):
        """Main optimization loop"""
        # Initialize population
        X = np.random.uniform(bounds[0], bounds[1], (self.n_pigeons, dimension))
        V = np.random.uniform(-0.05, 0.05, (self.n_pigeons, dimension))
        
        current_n = self.n_pigeons
        
        for iteration in range(self.max_iter):
            # Evaluate objectives
            objectives = [objective_func(x) for x in X[:current_n]]
            
            # Pareto sorting
            _, ranks, _ = NSGA2Sorter.nsga2_sort(objectives)
            
            # Get Pareto front
            front_indices = [i for i in range(current_n) if ranks[i] == 1]
            
            # Update archive
            self.archive.extend([X[i] for i in front_indices])
            if self.archive:
                archive_objectives = [objective_func(pos) for pos in self.archive]
                archive_ranks = NSGA2Sorter._pareto_sorting(archive_objectives)
                self.archive = [pos for pos, rank in zip(self.archive, archive_ranks) if rank == 1]
            
            # Calculate center and global best
            if front_indices:
                X_center = X[front_indices].mean(axis=0)
                X_g = self.archive[np.random.randint(len(self.archive))] if self.archive else X[front_indices[0]]
            else:
                X_center = X[:current_n].mean(axis=0)
                X_g = X[0]
            
            # Update positions
            new_X = X.copy()
            new_V = V.copy()
            
            for i in range(current_n):
                rank_i = ranks[i]
                n_leaders = int(self.pl * current_n)
                
                if rank_i <= n_leaders:
                    # General leader update (Equations 19, 20)
                    log_factor = np.log(iteration + 1) / np.log(self.max_iter)
                    new_V[i] = (np.exp(-self.R * (iteration + 1)) * V[i] + 
                               np.random.random() * self.ft * (1 - log_factor) * (X_g - X[i]) +
                               np.random.random() * self.ft * log_factor * (X_center - X[i]))
                    
                    new_V[i] = np.clip(new_V[i], -0.05, 0.05)
                    new_X[i] = X[i] + new_V[i]
                    new_X[i] = np.clip(new_X[i], bounds[0], bounds[1])
                else:
                    # Follower update (Equation 21)
                    for _ in range(self.learning_strength):
                        valid_pigeons = [j for j in range(current_n) if ranks[j] < rank_i]
                        if valid_pigeons:
                            dim = np.random.randint(dimension)
                            leader_idx = np.random.choice(valid_pigeons)
                            new_X[i][dim] = np.clip(
                                X[leader_idx][dim] + self.learning_error * np.random.random(),
                                bounds[0], bounds[1]
                            )
                
                # Check if new solution dominates old one
                new_obj = objective_func(new_X[i])
                old_obj = objectives[i]
                if not NSGA2Sorter._dominates(new_obj, old_obj):
                    new_X[i] = X[i]
            
            X = new_X
            V = new_V
            
            # Reduce population size
            if iteration < self.max_iter - 1 and current_n > self.nd:
                to_remove = min(self.nd, current_n - 1)
                # Remove worst ranked pigeons
                removal_indices = sorted(range(current_n), key=lambda x: ranks[x], reverse=True)[:to_remove]
                X = np.delete(X, removal_indices, axis=0)
                V = np.delete(V, removal_indices, axis=0)
                current_n -= to_remove
        
        # Return best solution
        final_objectives = [objective_func(x) for x in X[:current_n]]
        cost2_values = [obj[1] for obj in final_objectives]
        best_idx = np.argmin(cost2_values)
        
        return X[best_idx]

class FlockingControlAlgorithm:
    def __init__(self, n_uavs=5, Tmax=49.5, dt=0.5):
        # Initial positions from Table 1
        initial_positions = [
            (14.6929, 107.3676, 68.1682),
            (21.2809, 116.6406, 34.8423),
            (20.3911, 113.6529, 24.6351),
            (3.5699, 108.9509, 96.377),
            (10.2116, 111.558, 30.1431)
        ]
        
        # Obstacle positions from Table 1
        obstacle_positions = [
            (120, 120), (240, 75), (350, 40),
            (240, 155), (360, 110), (350, 180)
        ]
        
        self.uavs = [UAV(xyz=np.array([x, y, h]), Vxy=10.0, psi=0.0, lambda_=0.0) 
                     for x, y, h in initial_positions[:n_uavs]]
        self.obstacles = [Obstacle(xy=np.array(pos), radius=5, height=100) 
                         for pos in obstacle_positions]
        
        # Models
        self.uav_model = UAV_model()
        self.flocking_model = FlockingModel()
        self.obstacle_avoidance_model = ObstacleAvoidanceModel()
        self.performance_criteria = PerformanceCriteria()
        self.mpio = ModifiedMPIO()
        
        # Expected parameters
        self.ve = np.array([10.0, 0.0, 0.0])
        self.he = 50.0
        self.weights = [[1.0] * n_uavs for _ in range(n_uavs)]
        
        # Simulation parameters
        self.n_uavs = n_uavs
        self.Tmax = Tmax
        self.dt = dt
        self.ulim = 0.25
        self.Vxy_c_lim = 0.25
        self.psi_c_lim = 0.1

    def simulate(self):
        """Main simulation loop following Algorithm in Section 5"""
        trajectory = {i: {'x': [], 'y': [], 'h': [], 'vxy': [], 'psi': [], 'lambda': []}
                     for i in range(self.n_uavs)}
        cost_history = {'cost1': [], 'cost2': [], 'cost3': [], 'cost4': []}
        
        for t in np.arange(0, self.Tmax, self.dt):
            total_costs = [0.0, 0.0, 0.0, 0.0]
            
            for idx, uav in enumerate(self.uavs):
                # Step 3: Calculate desired flocking acceleration
                vf_dot = self.flocking_model.calculate_acc(uav, self.uavs, self.weights[idx], self.ve, self.he)
                
                # Calculate desired velocity for obstacle avoidance
                vo = self.obstacle_avoidance_model.calculate_velocity(uav, self.obstacles, self.ve, self.weights[idx][idx])
                
                # Step 4-10: Modified MPIO optimization
                def objective_func(wi):
                    attention_obstacles = self.obstacle_avoidance_model.get_attention_obstacles(uav, self.obstacles, self.ve)
                    cost1 = self.performance_criteria.cost1(uav, self.ve, attention_obstacles)
                    cost2 = self.performance_criteria.cost2(uav, self.uavs)
                    cost3 = self.performance_criteria.cost3(uav, self.obstacles)
                    cost4 = self.performance_criteria.cost4(uav, self.uavs)
                    
                    # Return high cost if hard constraints violated
                    if cost3 + cost4 > 0:
                        return [1000.0, 1000.0]
                    return [cost1, cost2]
                
                # Optimize weights using Modified MPIO
                optimal_weights = self.mpio.optimize(objective_func, len(self.weights[idx]), (0, 1))
                self.weights[idx] = optimal_weights.tolist()
                
                # Step 11: Calculate control input
                uav_vel = np.array([uav.Vxy * np.cos(uav.psi), uav.Vxy * np.sin(uav.psi)])
                u = np.zeros(3)
                u[:2] = vf_dot[:2] + (vo - uav_vel)
                u[2] = vf_dot[2]
                
                # Apply dead zone
                u[np.abs(u) < self.ulim] = 0
                
                # Calculate autopilot control inputs (Equation 26)
                Vxy_c = (self.uav_model.tau_v * (u[0] * np.cos(uav.psi) + u[1] * np.sin(uav.psi)) + 
                         uav.Vxy)
                psi_c = (self.uav_model.tau_psi / uav.Vxy * (u[1] * np.cos(uav.psi) - u[0] * np.sin(uav.psi)) + 
                         uav.psi)
                h_c = (uav.h + self.uav_model.tau_h / self.uav_model.tau_lambda * uav.lambda_ + 
                       self.uav_model.tau_h * u[2])
                
                # Apply control limits
                ve_norm = np.linalg.norm(self.ve[:2])
                if abs(Vxy_c - ve_norm) < self.Vxy_c_lim:
                    Vxy_c = ve_norm
                
                ve_direction = np.arctan2(self.ve[1], self.ve[0])
                if abs(psi_c - ve_direction) < self.psi_c_lim:
                    psi_c = ve_direction
                
                # Step 11 cont: Update UAV state
                control_inputs = {'Vxy_c': Vxy_c, 'psi_c': psi_c, 'h_c': h_c}
                self.uavs[idx] = self.uav_model.update_state(uav, control_inputs, self.dt)
                
                # Record trajectory
                trajectory[idx]['x'].append(self.uavs[idx].x)
                trajectory[idx]['y'].append(self.uavs[idx].y)
                trajectory[idx]['h'].append(self.uavs[idx].h)
                trajectory[idx]['vxy'].append(self.uavs[idx].Vxy)
                trajectory[idx]['psi'].append(self.uavs[idx].psi)
                trajectory[idx]['lambda'].append(self.uavs[idx].lambda_)
                
                # Calculate costs for monitoring
                attention_obstacles = self.obstacle_avoidance_model.get_attention_obstacles(uav, self.obstacles, self.ve)
                costs = [
                    self.performance_criteria.cost1(self.uavs[idx], self.ve, attention_obstacles),
                    self.performance_criteria.cost2(self.uavs[idx], self.uavs),
                    self.performance_criteria.cost3(self.uavs[idx], self.obstacles),
                    self.performance_criteria.cost4(self.uavs[idx], self.uavs)
                ]
                
                for j in range(4):
                    total_costs[j] += costs[j]
            
            # Store cost history
            for j, cost_name in enumerate(['cost1', 'cost2', 'cost3', 'cost4']):
                cost_history[cost_name].append(total_costs[j])
            
            # Print progress
            if int(t) % 10 == 0:
                print(f"Time: {t:.1f}s, Costs: {[f'{c:.2f}' for c in total_costs]}")
        
        return {
            'trajectory': trajectory,
            'costs': cost_history,
            'obstacles': self.obstacles,
            'time_vector': np.arange(0, self.Tmax, self.dt)
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
        ax3.axhline(y=self.he, color='k', linestyle='--', label='Expected')
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
        ax4.axhline(y=np.linalg.norm(self.ve[:2]), color='k', linestyle='--', label='Expected')
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

if __name__ == "__main__":
    print("Initializing UAV Flocking System with Modified MPIO...")
    system = FlockingControlAlgorithm(5, 10.0, 0.5) # Shorter simulation for testing
    
    print("Starting simulation...")
    results = system.simulate()
    
    print("Plotting results...")
    system.plot_results(results)
    
    print("Simulation completed successfully!")
