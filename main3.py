from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

class UAV_state:
    """UAV state representation"""
    def __init__(self):
        self.id= 0
        self.xyz= np.zeros((3,))

        self.Vxy= 10.0
        self.psi= 0.0
        self.lambda_= 0.0

    @property
    def xy(self):
        return self.xyz[:2]
    
    @property
    def h(self):
        return self.xyz[2]

@dataclass
class Obstacle:
    """Obstacle representation"""
    xy: np.ndarray = np.zeros((2,))
    radius: float = 5.0
    height: float = 100.0

class UAV_model:

    def __init__(self,Vxy_min=5.0,Vxy_max=20.0,lambda_min=-5.0,lambda_max=5.0,tau_v=1.0, tau_psi=0.75, tau_h=1.0, tau_lambda=0.3, n_max = 10, g = 10):
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
    
    def update_state(self, state: UAV_state, control_inputs: dict, dt: float):
        Vxy_c = control_inputs.get('Vxy_c', state.vxy)
        psi_c = control_inputs.get('psi_c', state.psi)
        h_c = control_inputs.get('h_c', state.h)

        # autopilots
        x_dot = state.Vxy*np.cos(state.psi)
        y_dot = state.Vxy*np.sin(state.psi)
        h_dot = state.lambda_
        Vxy_dot = (Vxy_c - state.Vxy) / self.tau_v
        psi_dot = (psi_c - state.psi) / self.tau_psi
        lambda_dot = (h_c - state.h) / self.tau_h - state.lambda_ / self.tau_lambda

        psi_clip_dummy = self.n_max * self.g / state.Vxy
        psi_dot = np.clip(psi_dot,-psi_clip_dummy,psi_clip_dummy)

        new_state = UAV_state(state.xy[0] + x_dot * dt,
                              state.xy[1] + y_dot * dt,
                              state.h + h_dot * dt,
                              np.clip(state.Vxy + Vxy_dot * dt, self.Vxy_min, self.Vxy_max),
                              state.psi + psi_dot * dt,
                              np.clip(state.lambda_ + lambda_dot * dt, self.lambda_min, self.lambda_max))

        return new_state

class FlockingModel:

    def __init__(self, Rcomm_1 = 20.0, Rlim_1 =2.0 , R_desire = 10.0, Kf = 0.1, Kc = 100_000.0 , Ka_vn = 0.1, Ka_he = 30.0 , Kv_e = 10.0 ):
        self.Rcomm_1 = Rcomm_1
        self.Rlim_1 = Rlim_1
        self.R_desire = R_desire
        self.Kf = Kf
        self.Kc = Kc
        self.Ka_vn = Ka_vn
        self.Ka_he = Ka_he
        self.Kv_e = Kv_e
    
    def calculate_acc(self, uav_state: UAV_state, neighbors: List[UAV_state], weights: Dict[int,float], ve: np.ndarray, he: float):
        
        for neighbor in neighbors:
            force_f = np.zeros((2,))
            force_c = np.zeros((2,))
            force_a_vn = np.zeros((2,))
            dist = np.linalg.norm(uav_state.xy-neighbor.xy)
            if dist <= self.Rcomm_1:
                force_f += weights[neighbor.id]*(neighbor.xy - uav_state.xy)*(1 - np.power(self.R_desire/dist,2))
                force_a_vn += weights[neighbor.id] * neighbor.xy - neighbor.xy

            
            if dist <= self.Rlim_1:
                force_c += np.power(1/np.abs(uav_state.xy-neighbor.xy) - 1 / self.Rlim_1,2) * uav_state.x-neighbor.xy / np.abs(uav_state.xy-neighbor.xy)

        xy_force = self.Kf * force_f + self.Kc * force_c + self.Ka_vn * force_a_vn
        h_force = self.Ka_he * (he - uav_state.h) + self.Kv_e * (ve[2] - uav_state.lambda_)
        
        return np.array([xy_force[0],xy_force[1],h_force])
        

class ObstacleAvoidanceModel:

    def __init__(self, R2_comm = 105.0 , R2_lim = 10.0 , theta_lim = np.pi/2):
        self.R2_comm = R2_comm
        self.R2_lim = R2_lim
        self.theta_lim = theta_lim
    
    def calculate_force(self, uav_state: UAV_state, obstacles: List[Obstacle], ve: np.ndarray):

        ve = ve[:2]

        attention,tangets,yaws = self.get_A0(uav_state, obstacles, ve)

        if len(attention) == 0:
            vo = ve.copy()
        else:
            
            ind1_idx = self.get_Ind_1(uav_state, attention)
            ind1 = attention[ind1_idx]

            if len(attention) == 1:
                vo = ve/np.linalg.norm(ve) * np.arctan2(ind1.xy[1] - uav_state.xy, ind1.xy[0] - uav_state.xy[0])
            else:
                ind2_idx,gap_angle,tanget_idx = self.get_Ind_2(UAV_state,ind1_idx,attention,yaws)
                ind2 = attention[ind2_idx]

                temp = ind1.xy+ind2.xy-2*uav_state.xy
                unit_vo = temp/np.linalg.norm(temp)
                vo = np.linalg.norm(ve)*unit_vo # weight eklenebilir
        
        return vo


    def get_A0(self, uav_state: UAV_state, obstacles: List[Obstacle], ve: np.ndarray) -> Tuple[List[Obstacle],List[float],List[float]]:

        def get_p(A,B,r):
            dist = np.linalg.norm(A-B)
            th = np.arccos(r / dist)
            d = np.arctan2(A[1] - B[1], A[0] - B[0])
            d1 = d + th
            d2 = d - th

            T1 = np.array([np.cos(d1),np.sin(d1)]) + B - A
            T2 = np.array([np.cos(d2),np.sin(d2)]) + B - A
            return T1,T2

        yaw_ve = np.arctan2(ve[1],ve[0])

        attention,tangents,yaws = [],[],[]

        for obstacle in obstacles:

            t1,t2 = get_p(uav_state.xy, obstacle.xy, obstacle.radius + self.R2_lim)
            dist = np.linalg.norm(t1)
            if dist > self.R2_comm: continue

            yaw_t1 = np.arctan2(t1[1],t1[0])
            yaw_t2 = np.arctan2(t2[1],t2[0])
            
            vec = obstacle.xy - uav_state.xy
            yaw_center = np.arctan2(vec[1],vec[0])

            t1_diff = abs(yaw_t1 - yaw_ve)
            while t1_diff > np.pi: t1_diff -= 2*np.pi

            t2_diff = abs(yaw_t2 - yaw_ve)
            while t2_diff > np.pi: t2_diff -= 2*np.pi

            center_diff = abs(yaw_center - yaw_ve)
            while center_diff > np.pi: center_diff -= 2*np.pi

            if abs(t1_diff) < self.theta_lim or abs(t2_diff) < self.theta_lim:
                attention.append(obstacle)
                tangents.append((t1,vec,t2))
                yaws.append((yaw_t1,yaw_center,yaw_t2))

        return attention,tangents,yaws
        

    def get_Ind_1(self, uav_state: UAV_state, obstacles: List[Obstacle]) -> int:
        """Find nearest obstacle index (Equation 9)"""
        min_dist = float('inf')
        nearest_idx = 0
        
        for i, obstacle in enumerate(obstacles):
            dist = np.linalg.norm(uav_state.xy-obstacle.xy)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        return nearest_idx
    
    def get_Ind_2(self, uav_state: UAV_state, ind1_idx:int , obstacles: List[Obstacle], yaws:List[float]) -> List[int]:
        
        max_gap = -1*float('inf')
        idx = 0
        ind_tangent = 1
        

        for i,obstacle in enumerate(obstacles):
            if i == ind1_idx: continue

            if yaws[i][1] > yaws[ind1_idx][1]:
                gap = yaws[i][2] - yaws[ind1_idx][0]
                tangent = 0
            elif yaws[i][1] < yaws[ind1_idx][1]:
                gap = yaws[i][0] - yaws[ind1_idx][2]
                tangent = 2
            else:
                gap = 0
                tangent = 1

            if np.abs(gap) > max_gap:
                idx = i
                max_gap = np.abs(gap)
                ind_tangent = tangent
        
        return idx,gap,ind_tangent
            

    
class Cost:

    def __init__(self,f1=1.0,f2=1.0,R_desire=10.0,R1_lim=2.0,R2_lim=2.0):
        self.f1 = f1
        self.f2 = f2
        self.R_desire = R_desire
        self.R1_lim = R1_lim
        self.R2_lim = R2_lim

    def cost1(self, uav_state: UAV_state, ve: np.ndarray, attention: List[Obstacle]):
        
        if len(attention) == 0:
            cost1 = np.abs(uav_state.Vxy[0] - ve[0]) + np.abs(uav_state.Vxy[1] - ve[1])
        else:
            cost1 = uav_state.xy.dot(ve)/np.linalg.norm(ve)
    
    def cost2(self, uav_state: UAV_state, neighbors: List[UAV_state]):

        cost2 = 0
        for neighbor in neighbors:
            dist = np.linalg.norm(uav_state.xy-neighbor.xy)
            cost2 += self.f1 * np.abs(self.R_desire - dist) + self.f2 * (np.abs(neighbor.Vxy[0] - uav_state.Vxy[0]) + np.abs(neighbor.Vxy[1] - uav_state.Vxy[1]))
        
        return cost2
    
    def cost3(self, uav_state: UAV_state, obstacles: List[Obstacle]):

        for obstacle in obstacles:
            if np.linalg.norm(uav_state.xy-obstacle.xy) <= self.R2_lim + obstacle.radius:
                return 1
        return 0
    
    def cost4(self, uav_state: UAV_state, neighbors: List[UAV_state]):

        for neighbor in neighbors:
            if np.linalg.norm(uav_state.xy-neighbor.xy) <= self.R1_lim:
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






class FlockingControlAlgorithm:

    def __init__(self, num_pigeons):
        
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

        self.uavs = [UAV_state(id=i,xyz=np.array([x,y,h]),Vxy=10.0,psi=0.0,lambda_=0.0) for i,(x,y,h) in enumerate(initial_positions)]
        self.obstacles = [Obstacle(xy = pos, radius=5 , height=100) for pos in initial_positions2]

        # models
        self.uav_model = UAV_model()
        self.flocking_model = FlockingModel()
        self.obstacle_avoidance_model = ObstacleAvoidanceModel()
        self.cost_functions = Cost()
        self.mmpio = MMPIO()

        self.ve = np.array([10.0,0.0,0.0])
        self.he = 50.0

    def simulate(self):
        i = 1

        for uav in self.uavs:
            vf_dot = self.flocking_model.calculate_acc(uav,[u for u in self.uavs if u != uav],[],self.ve,self.he)
            vo = self.obstacle_avoidance_model.calculate_force(uav,self.obstacles,self.ve)

            def objective_func():


            self.mmpio.run()






































class NSGA2Sorter:
    @staticmethod
    def _pareto_sorting(objectives: List[List[float]]) -> List[int]:
        """Pareto ranking of solutions (sizin kodunuz)"""
        n = len(objectives)
        ranks = [0] * n
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
       
        # Find domination relationships
        for i in range(n):
            for j in range(n):
                if i != j:
                    if NSGA2Sorter._dominates(objectives[i], objectives[j]):
                        dominated_solutions[i].append(j)
                    elif NSGA2Sorter._dominates(objectives[j], objectives[i]):
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
    
    # EKSİK OLAN KISI: CROWDING DISTANCE
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
            # Sort by this objective
            sorted_indices = sorted(range(n), 
                                  key=lambda i: objectives[front_indices[i]][obj_idx])
            
            # Boundary solutions get infinite distance
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Calculate objective range
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
    
    # TAM NSGA-II SORTING
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
                
            # Calculate crowding distance for this front
            distances = NSGA2Sorter._calculate_crowding_distance(objectives, front)
            
            # Create result tuples
            for i, sol_idx in enumerate(front):
                rank = front_idx + 1
                crowding_dist = distances[i]
                idxs.append(sol_idx)
                ranks.append(rank)
                dists.append(crowding_dist)

        return idxs,ranks,dists
