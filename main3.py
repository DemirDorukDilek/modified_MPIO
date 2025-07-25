from dataclasses import dataclass
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


import numpy as np



class NSGA2Sorter:
    @staticmethod
    def _pareto_sorting(objectives: List[List[float]]) -> List[int]:
        """Pareto ranking of solutions (sizin kodunuz)"""
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
        """Check if obj1 dominates obj2 (for minimization) (sizin kodunuz)"""
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
    def _get_fronts(objectives: List[List[float]], 
                   ranks: List[int]) -> List[List[int]]:
        """Group solutions by their Pareto rank"""
        max_rank = max(ranks)
        fronts = [[] for _ in range(max_rank)]
        
        for i, rank in enumerate(ranks):
            fronts[rank - 1].append(i)
        
        return fronts
    

    @staticmethod
    def nsga2_sort(objectives: List[List[float]]) -> List[Tuple[int, int, float]]:
        """
        Complete NSGA-II sorting
        
        Returns:
            List of (solution_index, rank, crowding_distance) sorted by NSGA-II criteria
        """
        ranks = NSGA2Sorter._pareto_sorting(objectives)
        
        fronts = NSGA2Sorter._get_fronts(objectives, ranks)
        
        idxs,ranks,dists = [],[],[]
        
        for front_idx, front in enumerate(fronts):
            if not front:
                continue
                

            distances = NSGA2Sorter._calculate_crowding_distance(objectives, front)
            

            for i, sol_idx in enumerate(front):
                rank = front_idx + 1
                crowding_dist = distances[i]
                idxs.append(sol_idx)
                ranks.append(rank)
                dists.append(crowding_dist)

        return idxs,ranks,dists







class UAV:
    def __init__(self,xyz,Vxy,psi,lambda_):

        self.xyz= xyz
        self.Vxy= Vxy
        self.psi= psi
        self.lambda_= lambda_

    @property
    def xy(self):
        return self.xyz[:2]
    
    @property
    def x(self): return self.xyz[0]
    @property
    def y(self): return self.xyz[1]
    
    @property
    def h(self):
        return self.xyz[2]

class Obstacle:

    def __init__(self,xy,radius,height):
        self.xy = xy
        self.radius = radius
        self.height = height

    @property
    def x(self): return self.xy[0]
    @property
    def y(self): return self.xy[1]

class UAV_model:

    def __init__(self,Vxy_min=5.0,Vxy_max=15.0,lambda_min=-5.0,lambda_max=5.0,tau_v=1.0, tau_psi=0.75, tau_h=1.0, tau_lambda=0.3, n_max = 10, g = 10):
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

        # autopilots
        x_dot = uav.Vxy*np.cos(uav.psi)
        y_dot = uav.Vxy*np.sin(uav.psi)
        h_dot = uav.lambda_
        Vxy_dot = (Vxy_c - uav.Vxy) / self.tau_v
        psi_dot = (psi_c - uav.psi) / self.tau_psi
        lambda_dot = (h_c - uav.h) / self.tau_h - uav.lambda_ / self.tau_lambda

        psi_clip_dummy = self.n_max * self.g / uav.Vxy
        psi_dot = np.clip(psi_dot,-psi_clip_dummy,psi_clip_dummy)

        new_state = UAV(np.array([uav.xy[0] + x_dot * dt,
                        uav.xy[1] + y_dot * dt,
                        uav.h + h_dot * dt]),
                        np.clip(uav.Vxy + Vxy_dot * dt, self.Vxy_min, self.Vxy_max),
                        uav.psi + psi_dot * dt,
                        np.clip(uav.lambda_ + lambda_dot * dt, self.lambda_min, self.lambda_max))

        return new_state

class FlockingModel:

    def __init__(self, R1_comm = 20.0, Rlim_1 =2.0 , R_desire = 10.0, Kf = 0.1, Kc = 100_000.0 , Ka_vn = 0.1, Ka_he = 30.0 , Kv_e = 10.0 ):
        self.Rcomm_1 = R1_comm
        self.Rlim_1 = Rlim_1
        self.R_desire = R_desire
        self.Kf = Kf
        self.Kc = Kc
        self.Ka_vn = Ka_vn
        self.Ka_he = Ka_he
        self.Kv_e = Kv_e
    
    def calculate_acc(self, uav: UAV, uavs: List[UAV], weights: Dict[int,float], ve: np.ndarray, he: float):
        
        force_f = np.zeros((2,))
        force_c = np.zeros((2,))
        force_a_vn = np.zeros((2,))
        for idx,o_uav in enumerate(uavs):
            if uav is o_uav: continue
            dist = np.linalg.norm(uav.xy-o_uav.xy)
            if dist <= self.Rcomm_1:
                force_f += weights[idx]*(o_uav.xy - uav.xy)*(1 - np.power(self.R_desire/dist,2))

                o_uav_vel = np.array([o_uav.Vxy * np.cos(o_uav.psi), o_uav.Vxy * np.sin(o_uav.psi)])
                uav_vel = np.array([uav.Vxy * np.cos(uav.psi), uav.Vxy * np.sin(uav.psi)])
                force_a_vn += weights[idx] * o_uav_vel - uav_vel

            
            if dist <= self.Rlim_1:
                force_c += np.power(1/dist - 1 / self.Rlim_1,2) * uav.xy-o_uav.xy / dist

        xy_force = self.Kf * force_f + self.Kc * force_c + self.Ka_vn * force_a_vn
        h_force = self.Ka_he * (he - uav.h) + self.Kv_e * (ve[2] - uav.lambda_)
        
        return np.array([xy_force[0],xy_force[1],h_force])
        

class ObstacleAvoidanceModel:
    def __init__(self, R2_comm=105.0, R2_lim=10.0, theta_lim=np.pi/2):
        self.R2_comm = R2_comm
        self.R2_lim = R2_lim
        self.theta_lim = theta_lim
    
    def get_A0(self, uav: UAV, obstacles: List[Obstacle], ve: np.ndarray) -> List[Obstacle]:
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
        
        return attention_obstacles,2,3
    
    def calculate_force(self, uav: UAV, obstacles: List[Obstacle], ve: np.ndarray, weight: float):
        """Calculate desired obstacle avoidance velocity (Section 3.3)"""
        attention_obstacles = self.get_A0(uav, obstacles, ve[:2])[0]
        
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


# class ObstacleAvoidanceModel:

#     def __init__(self, R2_comm = 105.0 , R2_lim = 10.0 , theta_lim = np.pi/2):
#         self.R2_comm = R2_comm
#         self.R2_lim = R2_lim
#         self.theta_lim = theta_lim
    
#     def calculate_force(self, uav: UAV, obstacles: List[Obstacle], ve: np.ndarray, weight:int):

#         ve = ve[:2]

#         attention,tangets,yaws = self.get_A0(uav, obstacles, ve)

#         if len(attention) == 0:
#             vo = ve.copy()
#         else:
            
#             ind1_idx = self.get_Ind_1(uav, attention)
#             ind1 = attention[ind1_idx]

#             if len(attention) == 1:
#                 vo = ve/np.linalg.norm(ve) * np.arctan2(ind1.xy[1] - uav.xy, ind1.xy[0] - uav.xy[0])
#             else:
#                 ind2_idx,gap_angle,tanget_idx = self.get_Ind_2(UAV,ind1_idx,attention,yaws)
#                 ind2 = attention[ind2_idx]

#                 temp = ind1.xy+ind2.xy-2*uav.xy
#                 unit_vo = temp/np.linalg.norm(temp)
#                 vo = np.linalg.norm(ve)*unit_vo*weight        
#         return vo


#     def get_A0(self, uav: UAV, obstacles: List[Obstacle], ve: np.ndarray) -> Tuple[List[Obstacle],List[float],List[float]]:

#         def get_p(A,B,r):
#             dist = np.linalg.norm(A-B)
#             th = np.arccos(r / dist)
#             d = np.arctan2(A[1] - B[1], A[0] - B[0])
#             d1 = d + th
#             d2 = d - th

#             T1 = np.array([np.cos(d1),np.sin(d1)]) + B - A
#             T2 = np.array([np.cos(d2),np.sin(d2)]) + B - A
#             return T1,T2

#         yaw_ve = np.arctan2(ve[1],ve[0])

#         attention,tangents,yaws = [],[],[]

#         for obstacle in obstacles:

#             t1,t2 = get_p(uav.xy, obstacle.xy, obstacle.radius + self.R2_lim)
#             dist = np.linalg.norm(t1)
#             if dist > self.R2_comm: continue

#             yaw_t1 = np.arctan2(t1[1],t1[0])
#             yaw_t2 = np.arctan2(t2[1],t2[0])
            
#             vec = obstacle.xy - uav.xy
#             yaw_center = np.arctan2(vec[1],vec[0])

#             t1_diff = abs(yaw_t1 - yaw_ve)
#             while t1_diff > np.pi: t1_diff -= 2*np.pi

#             t2_diff = abs(yaw_t2 - yaw_ve)
#             while t2_diff > np.pi: t2_diff -= 2*np.pi

#             center_diff = abs(yaw_center - yaw_ve)
#             while center_diff > np.pi: center_diff -= 2*np.pi

#             if abs(t1_diff) < self.theta_lim or abs(t2_diff) < self.theta_lim:
#                 attention.append(obstacle)
#                 tangents.append((t1,vec,t2))
#                 yaws.append((yaw_t1,yaw_center,yaw_t2))

#         return attention,tangents,yaws
        

#     def get_Ind_1(self, uav: UAV, obstacles: List[Obstacle]) -> int:
#         """Find nearest obstacle index (Equation 9)"""
#         min_dist = float('inf')
#         nearest_idx = 0
        
#         for i, obstacle in enumerate(obstacles):
#             dist = np.linalg.norm(uav.xy-obstacle.xy)
#             if dist < min_dist:
#                 min_dist = dist
#                 nearest_idx = i
        
#         return nearest_idx
    
#     def get_Ind_2(self, uav: UAV, ind1_idx:int , obstacles: List[Obstacle], yaws:List[float]) -> List[int]:
        
#         max_gap = -1*float('inf')
#         idx = 0
#         ind_tangent = 1
        

#         for i,obstacle in enumerate(obstacles):
#             if i == ind1_idx: continue

#             if yaws[i][1] > yaws[ind1_idx][1]:
#                 gap = yaws[i][2] - yaws[ind1_idx][0]
#                 tangent = 0
#             elif yaws[i][1] < yaws[ind1_idx][1]:
#                 gap = yaws[i][0] - yaws[ind1_idx][2]
#                 tangent = 2
#             else:
#                 gap = 0
#                 tangent = 1

#             if np.abs(gap) > max_gap:
#                 idx = i
#                 max_gap = np.abs(gap)
#                 ind_tangent = tangent
        
#         return idx,gap,ind_tangent
            

    
class Cost:

    def __init__(self,f1=1.0,f2=1.0,R_desire=10.0,R1_lim=2.0,R2_lim=2.0):
        self.f1 = f1
        self.f2 = f2
        self.R_desire = R_desire
        self.R1_lim = R1_lim
        self.R2_lim = R2_lim

    def cost1(self, uav: UAV, ve: np.ndarray, attention: List[Obstacle]):
        
        if len(attention) == 0:
            uav_vel = np.array([uav.Vxy * np.cos(uav.psi), uav.Vxy * np.sin(uav.psi)])
            cost1 = np.abs(uav_vel[0] - ve[0]) + np.abs(uav_vel[1] - ve[1])
        else:
            cost1 = -uav.xy.dot(ve[:2])/np.linalg.norm(ve[:2])
        return cost1
    
    def cost2(self, uav: UAV, uavs: List[UAV]):

        cost2 = 0
        for o_uav in uavs:
            if uav is o_uav: continue
            uav_vel = np.array([uav.Vxy * np.cos(uav.psi),uav.Vxy * np.sin(uav.psi)])
            o_uav_vel = np.array([o_uav.Vxy * np.cos(o_uav.psi),o_uav.Vxy * np.sin(o_uav.psi)])

            dist = np.linalg.norm(uav.xy-o_uav.xy)
            cost2 += self.f1 * np.abs(self.R_desire - dist) + self.f2 * (np.abs(o_uav_vel[0] - uav_vel[0]) + np.abs(o_uav_vel[1] - uav_vel[1]))
        
        return cost2
    
    def cost3(self, uav: UAV, obstacles: List[Obstacle]):

        for obstacle in obstacles:
            if np.linalg.norm(uav.xy-obstacle.xy) <= self.R2_lim + obstacle.radius:
                return 1
        return 0
    
    def cost4(self, uav: UAV, uavs: List[UAV]):

        for o_uav in uavs:
            if uav is o_uav: continue
            if np.linalg.norm(uav.xy-o_uav.xy) <= self.R1_lim:
                return 1
        return 0


class MMPIO:

    def __init__(self):
        self.N = 58
        self.Nc_max_3 = 20
        self.Nd = 2
        self.Xu= 1
        self.Xl= 0
        self.Vu= 0.05
        self.Vl= -0.05
        self.R = 0.3
        self.ft = 3
        self.pl = 0.9
        self.e = 2
        self.sl = 20
    
    def run(self,objectives_func, dimension):
        positions = np.random.uniform(self.Xl, self.Xu, (self.N, dimension))
        velocities = np.random.uniform(self.Vl, self.Vu, (self.N, dimension))
        Nc = 1

        archive = []

        current_n = self.N

        while Nc <= self.Nc_max_3:
            objectives = [objectives_func(position) for position in positions]
            
            indexs, ranks, crowding_distences = NSGA2Sorter.nsga2_sort(objectives)
            fronts_idx = [i for i in range(len(ranks)) if ranks[i] == 1]

            Xcenter = positions[fronts_idx].mean(axis=0)

            archive.extend([positions[i] for i in fronts_idx])
            if archive:
                archive_objectives = [objectives_func(pos) for pos in archive]
                archive_ranks = self._pareto_sorting(archive_objectives)
                archive = [pos for pos, rank in zip(archive, archive_ranks) if rank == 1]

            Xg = np.random.choice(archive)
            
            Nc+=1
            new_positions = np.copy(positions)
            new_velocities = np.copy(velocities)
            
            i = 1
            while i <= current_n:

                if ranks[indexs.index(i)] <= np.ceil(self.pl*current_n):
                    dummy_log = np.log(Nc)/np.log(self.Nc_max_3)
                    new_velocities[i] = np.exp(-self.R*Nc)*velocities[i]+np.random.random()*self.ft*(1-dummy_log)*(Xg-positions[i]) + np.random.random()*self.ft*dummy_log*(Xcenter-positions[i])
                    new_positions[i] = positions[i] + np.clip(new_velocities[i],self.Vl,self.Vu)
                else:
                    k = 1
                    while k <= self.sl:
                        valid_pigeons = [j for j in range(current_n) if ranks[j] < ranks[i]]
                        if valid_pigeons:
                            dim = np.ceil(np.random.random()*dimension)
                            new_positions[i][dim] = np.clip(positions[np.random.choice(valid_pigeons)][dim]+ self.e*np.random.random(),self.Xl,self.Xu)
                        k+=1
                
                new_objective = objectives_func(new_positions[i])

                if NSGA2Sorter._dominates(objectives[i],new_objective):
                    new_positions[i] = positions[i]
                i+=1

            positions = new_positions
            velocities = new_velocities
            
            if Nc <= self.Nc_max_3:
                to_remove = sorted(range(current_n), key=lambda x: ranks[x], reverse=True)[:min(self.Nd, current_n)]
                positions = np.delete(positions, to_remove, axis=0)
                velocities = np.delete(velocities, to_remove, axis=0)
                current_pop_size -= len(to_remove)
            
            # Return Pareto front
        if len(positions) == 0:
            return np.random.rand(dimension), []
            
        final_objectives = []
        for i in range(len(positions)):
            try:
                obj = objectives_func(positions[i])
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
                    obj = objectives_func(pos)
                    if len(obj) > 1 and obj[1] < min_cost2:
                        min_cost2 = obj[1]
                        best_idx = i
                except Exception:
                    continue
            return pareto_front[best_idx], final_objectives
        else:
            return positions[0], final_objectives



class DataClass: pass



class FlockingControlAlgorithm:

    def __init__(self, num_pigeons, Tmax = 49.5, dt=0.5):
        
        initial_positions = [
            (14.6929, 107.3676, 68.1682),
            (21.2809, 116.6406, 34.8423),
            (20.3911, 113.6529, 24.6351),
            (3.5699, 108.9509, 96.377),
            (10.2116, 111.558, 30.1431)
        ]

        initial_positions2 = [
            (120, 75),
            (120, 155),
            (240, 75),
            (240, 155),
            (350, 40),
            (350, 180),
            (360, 110),
        ]

        self.uavs = [UAV(xyz=np.array([x,y,h]),Vxy=10.0,psi=0.0,lambda_=0.0) for i,(x,y,h) in enumerate(initial_positions)]
        self.obstacles = [Obstacle(xy = np.array(pos), radius=5 , height=100) for pos in initial_positions2]

        # models
        self.uav_model = UAV_model()
        self.flocking_model = FlockingModel()
        self.obstacle_avoidance_model = ObstacleAvoidanceModel()
        self.performance_critetia = Cost()
        self.mmpio = MMPIO()

        self.ve = np.array([10.0,0.0,0.0])
        self.he = 50.0
        self.w = [([1]*num_pigeons).copy() for _ in range(num_pigeons)]

        self.num_pigeons = num_pigeons
        self.Tmax = Tmax
        self.dt = dt

        self.ulim = 0.25
        self.Vxy_c_lim = 0.25
        self.psi_c_lim = 0.1

    def simulate(self):
        trajectory = {i: {'x': [], 'y': [], 'h': [], 'vxy': [], 'psi': [], 'lambda': []} 
                     for i in range(self.num_pigeons)}
        cost_history = {'cost1': [], 'cost2': [], 'cost3': [], 'cost4': []}
        for t in np.arange(0,self.Tmax,self.dt):
            for idx,uav in enumerate(self.uavs):
                vf_dot = self.flocking_model.calculate_acc(uav,self.uavs,self.w[idx],self.ve,self.he)
                vo = self.obstacle_avoidance_model.calculate_force(uav,self.obstacles,self.ve,self.w[idx][idx])

                def objective_func(wi):

                    cost1 = self.performance_critetia.cost1(uav,self.ve,self.obstacle_avoidance_model.get_A0(uav,self.obstacles,self.ve)[0])
                    cost2 = self.performance_critetia.cost2(uav,self.uavs)
                    cost3 = self.performance_critetia.cost3(uav,self.obstacles)
                    cost4 = self.performance_critetia.cost4(uav,self.uavs)

                    return [cost1,cost2] if cost3+cost4 == 0 else [2000.0,2000.0]
                
                mpio = DataClass()
                mpio.dimention = len(self.w[0])
                mpio.n = self.num_pigeons 
                mpio.X = np.array(self.w.copy())
                mpio.V = np.zeros(mpio.X.shape)
                mpio.Nc = 1
                
                mpio.Xl,mpio.Xu = 0,1
                mpio.Vl,mpio.Vu = -0.05,0.05
                mpio.Nc3_max = 20
                mpio.pl = 0.9
                mpio.Nd = 2
                mpio.R = 0.3
                mpio.ft = 3
                mpio.of = objective_func
                mpio.archive = []

                mpio.objectives = [mpio.of(wi) for wi in self.w]

                mpio.indexs, mpio.ranks, _ = NSGA2Sorter.nsga2_sort(mpio.objectives)
                mpio.fronts_idx = [i for i in range(len(mpio.ranks)) if mpio.ranks[i] == 1]

                mpio.Xcenter = mpio.X[mpio.fronts_idx].mean(axis=0)

                mpio.archive.extend([mpio.X[i] for i in mpio.fronts_idx])
                if mpio.archive:
                    archive_objectives = [mpio.of(pos) for pos in mpio.archive]
                    archive_ranks = NSGA2Sorter._pareto_sorting(archive_objectives)
                    mpio.archive = [pos for pos, rank in zip(mpio.archive, archive_ranks) if rank == 1]

                mpio.Xg = mpio.archive[np.random.randint(len(mpio.archive))]
                mpio.Nc+=1
                mpio.i = 1
                mpio.nX = mpio.X.copy()
                mpio.nV = mpio.V.copy()

                while mpio.i <= mpio.n:
                    if mpio.ranks[mpio.indexs.index(mpio.i-1)] < mpio.pl*mpio.n:

                        dummy_log = np.log(mpio.Nc)/np.log(mpio.Nc3_max)
                        mpio.nV[mpio.i-1] = np.exp(-mpio.R*mpio.Nc)*mpio.V[mpio.i-1]+np.random.random()*mpio.ft*(1-dummy_log)*(mpio.Xg-mpio.X[mpio.i-1]) + np.random.random()*mpio.ft*dummy_log*(mpio.Xcenter-mpio.X[mpio.i-1])
                        mpio.nX[mpio.i-1] = mpio.X[mpio.i-1] + np.clip(mpio.nV[mpio.i-1],mpio.Vl,mpio.Vu)

                    else:
                        k = 1
                        while k <= self.sl:
                            valid_pigeons = [j for j in range(mpio.n) if mpio.ranks[j] < mpio.ranks[mpio.i-1]]
                            if valid_pigeons:
                                dim = np.ceil(np.random.random()*mpio.dimention)
                                mpio.nX[mpio.i-1][dim] = np.clip(mpio.X[np.random.choice(valid_pigeons)][dim]+ self.e*np.random.random(),self.Xl,self.Xu)
                            k+=1
                    
                    mpio.new_objective = mpio.of(mpio.nX[mpio.i-1])
                    if NSGA2Sorter._dominates(mpio.objectives[mpio.i-1],mpio.new_objective):
                        mpio.nX[mpio.i-1] = mpio.X[mpio.i-1]
                        mpio.new_objective = mpio.objectives[mpio.i-1]
                    
                    mpio.X = mpio.nX.copy()
                    mpio.V = mpio.nV.copy()
                    mpio.objectives[mpio.i-1] = mpio.new_objective
                
                    mpio.i += 1

                if mpio.Nc <= mpio.Nc3_max and mpio.n > mpio.Nd:
                    to_remove = sorted(range(mpio.n), key=lambda x: mpio.ranks[x], reverse=True)[:min(mpio.Nd, mpio.n)]
                    mpio.X = np.delete(mpio.X, to_remove, axis=0)
                    mpio.V = np.delete(mpio.V, to_remove, axis=0)
                    mpio.n -= len(to_remove)
                
                mpio.final_objectives = [objective_func(x) for x in mpio.X[:mpio.n]]
                cost2_values = [obj[1] for obj in mpio.final_objectives]
                self.w[idx] = mpio.X[np.argmin(cost2_values)].tolist()

                # cost2s = [mpio.objectives[i][1] if i in mpio.fronts_idx else float("inf") for i in range(mpio.n)]
                # self.w[idx] = mpio.X[np.argmin(cost2s)]

                u = np.zeros(vf_dot.shape)
                u[:2] = vf_dot[:2] + (vo[:2]-uav.Vxy)
                u[2] = vf_dot[2]

                u[np.abs(u) < self.ulim] = 0

                Vxy_c = self.uav_model.tau_v*(u[0]*np.cos(uav.psi)+u[1]*np.sin(uav.psi)) + uav.Vxy
                psi_c = self.uav_model.tau_psi/uav.Vxy*(u[1]*np.cos(uav.psi)+u[0]*np.sin(uav.psi)) + uav.psi
                h_c = uav.h + self.uav_model.tau_h/self.uav_model.tau_lambda*uav.lambda_+self.uav_model.tau_h*u[2]
                
                dummy = np.linalg.norm(self.ve[:2])
                if np.abs(Vxy_c-dummy) < self.Vxy_c_lim: Vxy_c = dummy
                dummy = np.arctan2(self.ve[1],self.ve[0])
                if np.abs(psi_c - dummy) < self.psi_c_lim: psi_c = dummy

                new_uav = self.uav_model.update_state(uav,{"Vxy_c":Vxy_c,"psi_c":psi_c,"h_c":h_c},self.dt)
                self.uavs[idx] = new_uav

                trajectory[idx]['x'].append(self.uavs[idx].x)
                trajectory[idx]['y'].append(self.uavs[idx].y)
                trajectory[idx]['h'].append(self.uavs[idx].h)
                trajectory[idx]['vxy'].append(self.uavs[idx].Vxy)
                trajectory[idx]['psi'].append(self.uavs[idx].psi)
                trajectory[idx]['lambda'].append(self.uavs[idx].lambda_)


            # Calculate and store cost functions
            total_costs = [0.0, 0.0, 0.0, 0.0]
            for i in range(self.num_pigeons):
                try:
                    detected_obstacles = self.obstacle_avoidance_model.get_A0(self.uavs[i], self.obstacles, self.ve)[0]
                    
                    costs = [
                        self.performance_critetia.cost1(self.uavs[i], self.ve, detected_obstacles),
                        self.performance_critetia.cost2(self.uavs[i], self.uavs),
                        self.performance_critetia.cost3(self.uavs[i], self.obstacles),
                        self.performance_critetia.cost4(self.uavs[i], self.uavs)
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

            if t % 10 == 0:
                print(f"Time: {t:.1f}s, Costs: {[f'{c:.2f}' for c in total_costs]}")

    
        return self.uavs,{
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
        for i in range(self.num_pigeons):
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
        for i in range(self.num_pigeons):
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
        for i in range(self.num_pigeons):
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
        for i in range(self.num_pigeons):
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
    system = FlockingControlAlgorithm(5,49.5,0.5)
    
    print("Starting simulation...")
    results = system.simulate()[1]  # Shorter simulation for testing
    
    print("Plotting results...")
    system.plot_results(results)
    
    print("Simulation completed successfully!")
    




# class ModifiedMPIO:
#     """Modified Multi-objective Pigeon-Inspired Optimization"""
    
#     def __init__(self, n_pigeons=20, max_iter=20, pl=0.9, nd=2, R=0.3, ft=3, 
#                  learning_error=0.01, learning_strength=2):
#         self.n_pigeons = n_pigeons
#         self.max_iter = max_iter
#         self.pl = pl # Percentage of general leaders
#         self.nd = nd # Reduced number of pigeons at each iteration
#         self.R = R # Map and compass factor
#         self.ft = ft # Transition factor
#         self.learning_error = learning_error
#         self.learning_strength = learning_strength
#         self.archive = []
    
#     def optimize(self, objective_func, dimension, bounds):
#         """Main optimization loop"""
#         # Initialize population
#         X = np.random.uniform(bounds[0], bounds[1], (self.n_pigeons, dimension))
#         V = np.random.uniform(-0.05, 0.05, (self.n_pigeons, dimension))
        
#         current_n = self.n_pigeons
        
#         for iteration in range(self.max_iter):
#             # Evaluate objectives
#             objectives = [objective_func(x) for x in X[:current_n]]
            
#             # Pareto sorting
#             _, ranks, _ = NSGA2Sorter.nsga2_sort(objectives)
            
#             # Get Pareto front
#             front_indices = [i for i in range(current_n) if ranks[i] == 1]
            
#             # Update archive
#             self.archive.extend([X[i] for i in front_indices])
#             if self.archive:
#                 archive_objectives = [objective_func(pos) for pos in self.archive]
#                 archive_ranks = NSGA2Sorter._pareto_sorting(archive_objectives)
#                 self.archive = [pos for pos, rank in zip(self.archive, archive_ranks) if rank == 1]
            
#             # Calculate center and global best
#             if front_indices:
#                 X_center = X[front_indices].mean(axis=0)
#                 X_g = self.archive[np.random.randint(len(self.archive))] if self.archive else X[front_indices[0]]
#             else:
#                 X_center = X[:current_n].mean(axis=0)
#                 X_g = X[0]
            
#             # Update positions
#             new_X = X.copy()
#             new_V = V.copy()
            
#             for i in range(current_n):
#                 rank_i = ranks[i]
#                 n_leaders = int(self.pl * current_n)
                
#                 if rank_i <= n_leaders:
#                     # General leader update (Equations 19, 20)
#                     log_factor = np.log(iteration + 1) / np.log(self.max_iter)
#                     new_V[i] = (np.exp(-self.R * (iteration + 1)) * V[i] + 
#                                np.random.random() * self.ft * (1 - log_factor) * (X_g - X[i]) +
#                                np.random.random() * self.ft * log_factor * (X_center - X[i]))
                    
#                     new_V[i] = np.clip(new_V[i], -0.05, 0.05)
#                     new_X[i] = X[i] + new_V[i]
#                     new_X[i] = np.clip(new_X[i], bounds[0], bounds[1])
#                 else:
#                     # Follower update (Equation 21)
#                     for _ in range(self.learning_strength):
#                         valid_pigeons = [j for j in range(current_n) if ranks[j] < rank_i]
#                         if valid_pigeons:
#                             dim = np.random.randint(dimension)
#                             leader_idx = np.random.choice(valid_pigeons)
#                             new_X[i][dim] = np.clip(
#                                 X[leader_idx][dim] + self.learning_error * np.random.random(),
#                                 bounds[0], bounds[1]
#                             )
                
#                 # Check if new solution dominates old one
#                 new_obj = objective_func(new_X[i])
#                 old_obj = objectives[i]
#                 if not NSGA2Sorter._dominates(new_obj, old_obj):
#                     new_X[i] = X[i]
            
#             X = new_X
#             V = new_V
            
#             # Reduce population size
#             if iteration < self.max_iter - 1 and current_n > self.nd:
#                 to_remove = min(self.nd, current_n - 1)
#                 # Remove worst ranked pigeons
#                 removal_indices = sorted(range(current_n), key=lambda x: ranks[x], reverse=True)[:to_remove]
#                 X = np.delete(X, removal_indices, axis=0)
#                 V = np.delete(V, removal_indices, axis=0)
#                 current_n -= to_remove
        
#         # Return best solution
#         final_objectives = [objective_func(x) for x in X[:current_n]]
#         cost2_values = [obj[1] for obj in final_objectives]
#         best_idx = np.argmin(cost2_values)
        
#         return X[best_idx]

# class FlockingControlAlgorithm:
#     def __init__(self, n_uavs=5, Tmax=49.5, dt=0.5):
#         # Initial positions from Table 1
#         initial_positions = [
#             (14.6929, 107.3676, 68.1682),
#             (21.2809, 116.6406, 34.8423),
#             (20.3911, 113.6529, 24.6351),
#             (3.5699, 108.9509, 96.377),
#             (10.2116, 111.558, 30.1431)
#         ]
        
#         # Obstacle positions from Table 1
#         obstacle_positions = [
#             (120, 120), (240, 75), (350, 40),
#             (240, 155), (360, 110), (350, 180)
#         ]
        
#         self.uavs = [UAV(xyz=np.array([x, y, h]), Vxy=10.0, psi=0.0, lambda_=0.0) 
#                      for x, y, h in initial_positions[:n_uavs]]
#         self.obstacles = [Obstacle(xy=np.array(pos), radius=5, height=100) 
#                          for pos in obstacle_positions]
        
#         # Models
#         self.uav_model = UAV_model()
#         self.flocking_model = FlockingModel()
#         self.obstacle_avoidance_model = ObstacleAvoidanceModel()
#         self.performance_criteria = Cost()
#         self.mpio = ModifiedMPIO()
        
#         # Expected parameters
#         self.ve = np.array([10.0, 0.0, 0.0])
#         self.he = 50.0
#         self.weights = [[1.0] * n_uavs for _ in range(n_uavs)]
        
#         # Simulation parameters
#         self.n_uavs = n_uavs
#         self.Tmax = Tmax
#         self.dt = dt
#         self.ulim = 0.25
#         self.Vxy_c_lim = 0.25
#         self.psi_c_lim = 0.1

#     def simulate(self):
#         """Main simulation loop following Algorithm in Section 5"""
#         trajectory = {i: {'x': [], 'y': [], 'h': [], 'vxy': [], 'psi': [], 'lambda': []}
#                      for i in range(self.n_uavs)}
#         cost_history = {'cost1': [], 'cost2': [], 'cost3': [], 'cost4': []}
        
#         for t in np.arange(0, self.Tmax, self.dt):
#             total_costs = [0.0, 0.0, 0.0, 0.0]
            
#             for idx, uav in enumerate(self.uavs):
#                 # Step 3: Calculate desired flocking acceleration
#                 vf_dot = self.flocking_model.calculate_acc(uav, self.uavs, self.weights[idx], self.ve, self.he)
                
#                 # Calculate desired velocity for obstacle avoidance
#                 vo = self.obstacle_avoidance_model.calculate_force(uav, self.obstacles, self.ve, self.weights[idx][idx])
                
#                 # Step 4-10: Modified MPIO optimization
#                 def objective_func(wi):
#                     attention_obstacles = self.obstacle_avoidance_model.get_A0(uav, self.obstacles, self.ve)[0]
#                     cost1 = self.performance_criteria.cost1(uav, self.ve, attention_obstacles)
#                     cost2 = self.performance_criteria.cost2(uav, self.uavs)
#                     cost3 = self.performance_criteria.cost3(uav, self.obstacles)
#                     cost4 = self.performance_criteria.cost4(uav, self.uavs)
                    
#                     # Return high cost if hard constraints violated
#                     if cost3 + cost4 > 0:
#                         return [1000.0, 1000.0]
#                     return [cost1, cost2]
                
#                 # Optimize weights using Modified MPIO
#                 optimal_weights = self.mpio.optimize(objective_func, len(self.weights[idx]), (0, 1))
#                 self.weights[idx] = optimal_weights.tolist()
                
#                 # Step 11: Calculate control input
#                 uav_vel = np.array([uav.Vxy * np.cos(uav.psi), uav.Vxy * np.sin(uav.psi)])
#                 u = np.zeros(3)
#                 u[:2] = vf_dot[:2] + (vo - uav_vel)
#                 u[2] = vf_dot[2]
                
#                 # Apply dead zone
#                 u[np.abs(u) < self.ulim] = 0
                
#                 # Calculate autopilot control inputs (Equation 26)
#                 Vxy_c = (self.uav_model.tau_v * (u[0] * np.cos(uav.psi) + u[1] * np.sin(uav.psi)) + 
#                          uav.Vxy)
#                 psi_c = (self.uav_model.tau_psi / uav.Vxy * (u[1] * np.cos(uav.psi) - u[0] * np.sin(uav.psi)) + 
#                          uav.psi)
#                 h_c = (uav.h + self.uav_model.tau_h / self.uav_model.tau_lambda * uav.lambda_ + 
#                        self.uav_model.tau_h * u[2])
                
#                 # Apply control limits
#                 ve_norm = np.linalg.norm(self.ve[:2])
#                 if abs(Vxy_c - ve_norm) < self.Vxy_c_lim:
#                     Vxy_c = ve_norm
                
#                 ve_direction = np.arctan2(self.ve[1], self.ve[0])
#                 if abs(psi_c - ve_direction) < self.psi_c_lim:
#                     psi_c = ve_direction
                
#                 # Step 11 cont: Update UAV state
#                 control_inputs = {'Vxy_c': Vxy_c, 'psi_c': psi_c, 'h_c': h_c}
#                 self.uavs[idx] = self.uav_model.update_state(uav, control_inputs, self.dt)
                
#                 # Record trajectory
#                 trajectory[idx]['x'].append(self.uavs[idx].x)
#                 trajectory[idx]['y'].append(self.uavs[idx].y)
#                 trajectory[idx]['h'].append(self.uavs[idx].h)
#                 trajectory[idx]['vxy'].append(self.uavs[idx].Vxy)
#                 trajectory[idx]['psi'].append(self.uavs[idx].psi)
#                 trajectory[idx]['lambda'].append(self.uavs[idx].lambda_)
                
#                 # Calculate costs for monitoring
#                 attention_obstacles = self.obstacle_avoidance_model.get_A0(uav, self.obstacles, self.ve)[0]
#                 costs = [
#                     self.performance_criteria.cost1(self.uavs[idx], self.ve, attention_obstacles),
#                     self.performance_criteria.cost2(self.uavs[idx], self.uavs),
#                     self.performance_criteria.cost3(self.uavs[idx], self.obstacles),
#                     self.performance_criteria.cost4(self.uavs[idx], self.uavs)
#                 ]
                
#                 for j in range(4):
#                     total_costs[j] += costs[j]
            
#             # Store cost history
#             for j, cost_name in enumerate(['cost1', 'cost2', 'cost3', 'cost4']):
#                 cost_history[cost_name].append(total_costs[j])
            
#             # Print progress
#             if int(t) % 10 == 0:
#                 print(f"Time: {t:.1f}s, Costs: {[f'{c:.2f}' for c in total_costs]}")
        
#         return {
#             'trajectory': trajectory,
#             'costs': cost_history,
#             'obstacles': self.obstacles,
#             'time_vector': np.arange(0, self.Tmax, self.dt)
#         }

#     def plot_results(self, results: Dict):
#         """Plot simulation results"""
#         fig = plt.figure(figsize=(15, 12))
        
#         # 3D trajectory plot
#         ax1 = fig.add_subplot(2, 3, 1, projection='3d')
#         colors = ['b', 'r', 'g', 'm', 'c']
#         for i in range(self.n_uavs):
#             traj = results['trajectory'][i]
#             ax1.plot(traj['x'], traj['y'], traj['h'], colors[i % len(colors)],
#                     label=f'UAV {i+1}', linewidth=2)
        
#         # Plot obstacles
#         for obs in results['obstacles']:
#             theta = np.linspace(0, 2*np.pi, 100)
#             x_circle = obs.x + obs.radius * np.cos(theta)
#             y_circle = obs.y + obs.radius * np.sin(theta)
#             z_circle = np.zeros_like(x_circle)
#             ax1.plot(x_circle, y_circle, z_circle, 'k-', linewidth=2)
#             ax1.plot(x_circle, y_circle, z_circle + obs.height, 'k-', linewidth=2)
        
#         ax1.set_xlabel('X (m)')
#         ax1.set_ylabel('Y (m)')
#         ax1.set_zlabel('Altitude (m)')
#         ax1.set_title('3D UAV Trajectories')
#         ax1.legend()
        
#         # Top-down view
#         ax2 = fig.add_subplot(2, 3, 2)
#         for i in range(self.n_uavs):
#             traj = results['trajectory'][i]
#             ax2.plot(traj['x'], traj['y'], colors[i % len(colors)],
#                     label=f'UAV {i+1}', linewidth=2)
        
#         # Plot obstacles
#         for obs in results['obstacles']:
#             circle = plt.Circle((obs.x, obs.y), obs.radius, fill=False, color='black', linewidth=2)
#             ax2.add_patch(circle)
        
#         ax2.set_xlabel('X (m)')
#         ax2.set_ylabel('Y (m)')
#         ax2.set_title('Top-down View')
#         ax2.legend()
#         ax2.grid(True)
#         ax2.axis('equal')
        
#         # Altitude vs time
#         ax3 = fig.add_subplot(2, 3, 3)
#         for i in range(self.n_uavs):
#             traj = results['trajectory'][i]
#             ax3.plot(results['time_vector'], traj['h'], colors[i % len(colors)],
#                     label=f'UAV {i+1}', linewidth=2)
#         ax3.axhline(y=self.he, color='k', linestyle='--', label='Expected')
#         ax3.set_xlabel('Time (s)')
#         ax3.set_ylabel('Altitude (m)')
#         ax3.set_title('Altitude vs Time')
#         ax3.legend()
#         ax3.grid(True)
        
#         # Horizontal airspeed vs time
#         ax4 = fig.add_subplot(2, 3, 4)
#         for i in range(self.n_uavs):
#             traj = results['trajectory'][i]
#             ax4.plot(results['time_vector'], traj['vxy'], colors[i % len(colors)],
#                     label=f'UAV {i+1}', linewidth=2)
#         ax4.axhline(y=np.linalg.norm(self.ve[:2]), color='k', linestyle='--', label='Expected')
#         ax4.set_xlabel('Time (s)')
#         ax4.set_ylabel('Horizontal Speed (m/s)')
#         ax4.set_title('Horizontal Airspeed vs Time')
#         ax4.legend()
#         ax4.grid(True)
        
#         # Cost functions vs time
#         ax5 = fig.add_subplot(2, 3, 5)
#         ax5.plot(results['time_vector'], results['costs']['cost1'], 'b-', label='Cost 1', linewidth=2)
#         ax5.plot(results['time_vector'], results['costs']['cost2'], 'r-', label='Cost 2', linewidth=2)
#         ax5.set_xlabel('Time (s)')
#         ax5.set_ylabel('Cost Value')
#         ax5.set_title('Soft Constraints (Cost 1 & 2)')
#         ax5.legend()
#         ax5.grid(True)
        
#         # Hard constraints vs time
#         ax6 = fig.add_subplot(2, 3, 6)
#         ax6.plot(results['time_vector'], results['costs']['cost3'], 'g-', label='Cost 3 (Obstacle)', linewidth=2)
#         ax6.plot(results['time_vector'], results['costs']['cost4'], 'm-', label='Cost 4 (Collision)', linewidth=2)
#         ax6.set_xlabel('Time (s)')
#         ax6.set_ylabel('Violation Count')
#         ax6.set_title('Hard Constraints (Cost 3 & 4)')
#         ax6.legend()
#         ax6.grid(True)
        
#         plt.tight_layout()
#         plt.show()

# if __name__ == "__main__":
#     print("Initializing UAV Flocking System with Modified MPIO...")
#     system = FlockingControlAlgorithm(5, 10.0, 0.5) # Shorter simulation for testing
    
#     print("Starting simulation...")
#     results = system.simulate()
    
#     print("Plotting results...")
#     system.plot_results(results)
    
#     print("Simulation completed successfully!")
