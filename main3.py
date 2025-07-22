from dataclasses import dataclass
from typing import Dict, List

import numpy as np

@dataclass
class UAV_state:
    """UAV state representation"""
    id: int = 0
    xy: np.ndarray = np.zeros((2,))
    h: float = 50.0
    Vxy: float = 10.0
    psi: float = 0.0
    lambda_: float = 0.0

@dataclass
class Obstacle:
    """Obstacle representation"""
    xy: np.ndarray = np.zeros((2,))
    radius: float
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
                force_a_vn += weights[neighbor.id] * neighbor.xy - neighbor.yy

            
            if dist <= self.Rlim_1:
                force_c += np.power(1/np.abs(uav_state.xy-neighbor.xy) - 1 / self.Rlim_1,2) * uav_state.x-neighbor.xy / np.abs(uav_state.xy-neighbor.xy)

        xy_force = self.Kf * force_f + self.Kc * force_c + self.Ka_vn * force_a_vn
        h_force = self.Ka_he * (he - uav_state.h) + self.Kv_e * (ve - uav_state.lambda_)
        
        return np.array([xy_force[0],xy_force[1],h_force])
        

class ObstacleAvoidanceModel:

    def __init__(self, R2_comm = 105.0 , R2_lim = 10.0 , theta_lim = np.pi/2):
        self.R2_comm = R2_comm
        self.R2_lim = R2_lim
        self.theta_lim = theta_lim
    
    def calculate_force(self, uav_state: UAV_state, obstacles: List[Obstacle], ve: np.ndarray):

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
                vo = np.linalg.norm(ve)*unit_vo


    # def get_A0(self, uav_state: UAV_state, obstacles: List[Obstacle], ve: np.ndarray):

    #     yaw_ve = np.arctan2(ve[1],ve[0])

    #     attention = []

    #     for obstacle in obstacles:

    #         obstacle_rel_vec = obstacle.xy - uav_state.xy
    #         dist = np.linalg.norm(obstacle_rel_vec)
    #         yaw_obstacle = np.arctan2(obstacle_rel_vec[1],obstacle_rel_vec[0])

    #         angle_diff = abs(yaw_obstacle - yaw_ve)
    #         while angle_diff > np.pi: angle_diff -= 2*np.pi
    #         if dist < self.R2_comm + obstacle.radius + self.R2_lim and abs(angle_diff) < self.theta_lim:
    #             attention.append(obstacle)
        
    #     return attention
    
    def get_A0(self, uav_state: UAV_state, obstacles: List[Obstacle], ve: np.ndarray) -> List[List[Obstacle],List[float],List[float]]:

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
    
    def get_Ind_2(self, uav_state: UAV_state, ind1_idx:int , obstacles: List[Obstacle], yaws:List[float]) -> List[int,int,int]:
        
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
            

    
        