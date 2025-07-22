

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random

@dataclass
class UAVState:
    """UAV durumu"""
    x: float  # x pozisyonu
    y: float  # y pozisyonu
    h: float  # irtifa
    v_xy: float  # yatay hız
    psi: float  # yaw açısı
    lambda_: float  # dikey hız

@dataclass
class Obstacle:
    """Engel tanımı"""
    x: float
    y: float
    radius: float

class UAVModel:
    """UAV dinamik modeli (Denklem 3)"""
    def __init__(self, tau_v=1.0, tau_psi=0.75, tau_h=1.0, tau_lambda=0.3):
        self.tau_v = tau_v
        self.tau_psi = tau_psi
        self.tau_h = tau_h
        self.tau_lambda = tau_lambda
        
        # Limitler
        self.v_xy_min = 5.0
        self.v_xy_max = 15.0
        self.lambda_min = -5.0
        self.lambda_max = 5.0
        self.n_max = 10.0  # maksimum yanal yük faktörü
        self.g = 10.0
        
    def update(self, state: UAVState, v_xy_c: float, psi_c: float, h_c: float, dt: float) -> UAVState:
        """UAV durumunu güncelle"""
        # Pozisyon güncellemesi
        x_new = state.x + state.v_xy * np.cos(state.psi) * dt
        y_new = state.y + state.v_xy * np.sin(state.psi) * dt
        h_new = state.h + state.lambda_ * dt
        
        # Hız güncellemesi (1. derece sistem)
        v_xy_dot = (1/self.tau_v) * (v_xy_c - state.v_xy)
        v_xy_new = state.v_xy + v_xy_dot * dt
        v_xy_new = np.clip(v_xy_new, self.v_xy_min, self.v_xy_max)
        
        # Yaw güncellemesi (1. derece sistem)
        psi_dot = (1/self.tau_psi) * (psi_c - state.psi)
        # Dönüş hızı limiti
        psi_dot_max = self.n_max * self.g / state.v_xy
        psi_dot = np.clip(psi_dot, -psi_dot_max, psi_dot_max)
        psi_new = state.psi + psi_dot * dt
        
        # İrtifa kontrolü (2. derece sistem)
        lambda_dot = (1/self.tau_h) * (h_c - state.h) - (1/self.tau_lambda) * state.lambda_
        lambda_new = state.lambda_ + lambda_dot * dt
        lambda_new = np.clip(lambda_new, self.lambda_min, self.lambda_max)
        
        return UAVState(x_new, y_new, h_new, v_xy_new, psi_new, lambda_new)

class FlockingController:
    """Sürü kontrolü (Denklem 5-8)"""
    def __init__(self, R_comm=20.0, R_lim=2.0, R_desire=10.0):
        self.R_comm = R_comm  # iletişim menzili
        self.R_lim = R_lim  # minimum güvenli mesafe
        self.R_desire = R_desire  # istenen mesafe
        
        # Kazanç parametreleri
        self.K_f = 0.1  # sürü kontrolü
        self.K_c = 100000.0  # çarpışma önleme
        self.K_a_vn = 0.1  # hız uyumu
        self.K_a_h_e = 30.0  # irtifa uyumu
        self.K_v_e = 10.0  # dikey hız uyumu
        
        self.h_e = 50.0  # hedef irtifa
        self.v_e = np.array([10.0, 0.0, 0.0])  # hedef hız
        
    def calculate_flocking_force(self, uav_i: UAVState, neighbors: List[Tuple[UAVState, float]], 
                                 weights: np.ndarray) -> np.ndarray:
        """Sürü kuvvetini hesapla (Denklem 6-8)"""
        f_f = np.zeros(2)  # flocking geometry
        f_c = np.zeros(2)  # collision avoidance
        f_a_vn = np.zeros(2)  # velocity alignment
        
        pos_i = np.array([uav_i.x, uav_i.y])
        vel_i = np.array([uav_i.v_xy * np.cos(uav_i.psi), 
                          uav_i.v_xy * np.sin(uav_i.psi)])
        
        for (uav_j, w_j), w in zip(neighbors, weights):
            pos_j = np.array([uav_j.x, uav_j.y])
            vel_j = np.array([uav_j.v_xy * np.cos(uav_j.psi), 
                              uav_j.v_xy * np.sin(uav_j.psi)])
            
            d_ij = np.linalg.norm(pos_j - pos_i)
            
            if d_ij <= self.R_comm:
                # Sürü geometri kontrolü (Denklem 6)
                f_f += w * (pos_j - pos_i) * (1 - (self.R_desire/d_ij)**2)
                
                # Hız uyumu (Denklem 8)
                f_a_vn += w * (vel_j - vel_i)
                
            if d_ij <= self.R_lim and d_ij > 0:
                # Çarpışma önleme (Denklem 7)
                direction = (pos_i - pos_j) / d_ij
                magnitude = (1/d_ij - 1/self.R_lim)**2
                f_c += self.K_c * magnitude * direction
        
        # Yatay kanal kuvvetleri
        horizontal_force = self.K_f * f_f + f_c + self.K_a_vn * f_a_vn
        
        # Dikey kanal kontrolü
        vertical_force = self.K_a_h_e * (self.h_e - uav_i.h) + \
                        self.K_v_e * (self.v_e[2] - uav_i.lambda_)
        
        return np.array([horizontal_force[0], horizontal_force[1], vertical_force])

class ObstacleAvoidance:
    """Engel kaçınma modeli (Section 3.3)"""
    def __init__(self, R_comm=105.0, R_lim=10.0, theta_lim=np.pi/2):
        self.R_comm = R_comm  # algılama menzili
        self.R_lim = R_lim  # minimum engel mesafesi
        self.theta_lim = theta_lim  # görüş açısı
        
    def detect_obstacles(self, uav: UAVState, obstacles: List[Obstacle]) -> List[int]:
        """Görüş alanındaki engelleri tespit et"""
        detected = []
        uav_pos = np.array([uav.x, uav.y])
        
        for i, obs in enumerate(obstacles):
            obs_pos = np.array([obs.x, obs.y])
            d = np.linalg.norm(obs_pos - uav_pos)
            
            # Mesafe kontrolü
            if d > self.R_comm:
                continue
                
            # Açı kontrolü
            angle_to_obs = np.arctan2(obs.y - uav.y, obs.x - uav.x)
            angle_diff = np.abs(angle_to_obs - uav.psi)
            angle_diff = min(angle_diff, 2*np.pi - angle_diff)
            
            if angle_diff <= self.theta_lim:
                detected.append(i)
                
        return detected
    
    def find_largest_gap(self, uav: UAVState, obstacles: List[Obstacle], 
                        detected_indices: List[int]) -> float:
        """En büyük boşluğu bul ve hedef açıyı döndür"""
        if not detected_indices:
            # Engel yoksa düz devam
            return np.arctan2(self.v_e[1], self.v_e[0])
            
        if len(detected_indices) == 1:
            # Tek engel varsa kenarından geç
            obs = obstacles[detected_indices[0]]
            angle_to_obs = np.arctan2(obs.y - uav.y, obs.x - uav.x)
            # Engelin sağından veya solundan geç
            return angle_to_obs + np.pi/4  # Basit strateji
            
        # Birden fazla engel - en büyük boşluğu bul
        gaps = []
        for i in range(len(detected_indices)):
            for j in range(i+1, len(detected_indices)):
                obs1 = obstacles[detected_indices[i]]
                obs2 = obstacles[detected_indices[j]]
                
                # İki engel arasındaki açısal gap
                angle1 = np.arctan2(obs1.y - uav.y, obs1.x - uav.x)
                angle2 = np.arctan2(obs2.y - uav.y, obs2.x - uav.x)
                
                gap_angle = abs(angle2 - angle1)
                gap_center = (angle1 + angle2) / 2
                
                gaps.append((gap_angle, gap_center))
        
        # En büyük boşluğu seç
        if gaps:
            largest_gap = max(gaps, key=lambda x: x[0])
            return largest_gap[1]
        else:
            return uav.psi

class ModifiedMPIO:
    """Modifiye Çok-Amaçlı Güvercin İlhamlı Optimizasyon"""
    def __init__(self, n_pigeons=58, max_iter=20, R=0.3, f_t=3.0, p_l=0.9):
        self.n_pigeons = n_pigeons
        self.max_iter = max_iter
        self.R = R  # harita ve pusula faktörü
        self.f_t = f_t  # geçiş faktörü
        self.p_l = p_l  # lider yüzdesi
        self.e = 0.01  # öğrenme hatası
        self.s_l = 2  # öğrenme gücü
        self.N_d = 2  # her iterasyonda azaltılan güvercin sayısı
        
    def optimize(self, objective_func, n_vars, bounds):
        """Optimizasyon çalıştır"""
        # Başlangıç popülasyonu
        pigeons = np.random.uniform(bounds[0], bounds[1], (self.n_pigeons, n_vars))
        velocities = np.zeros((self.n_pigeons, n_vars))
        
        # En iyi pozisyonlar
        X_g = None
        best_fitness = float('inf')
        
        current_n = self.n_pigeons
        
        for nc in range(self.max_iter):
            # Amaç fonksiyonlarını hesapla
            fitness_values = np.array([objective_func(p) for p in pigeons[:current_n]])
            
            # Pareto sıralama (basitleştirilmiş)
            ranks = self.pareto_ranking(fitness_values)
            
            # Global en iyi güncelle
            min_idx = np.argmin(fitness_values[:, 0])  # İlk amaca göre
            if fitness_values[min_idx, 0] < best_fitness:
                best_fitness = fitness_values[min_idx, 0]
                X_g = pigeons[min_idx].copy()
            
            # X_center hesapla
            pareto_front = pigeons[ranks == 1]
            X_center = np.mean(pareto_front, axis=0)
            
            # Güvercinleri güncelle
            n_leaders = int(self.p_l * current_n)
            
            for i in range(current_n):
                if i < n_leaders:
                    # Liderler - harita ve pusula + işaret noktası
                    v_map = np.exp(-self.R * (nc+1)) * velocities[i]
                    v_compass = random.random() * self.f_t * (1 - np.log10((nc+1)/self.max_iter)) * (X_g - pigeons[i])
                    v_landmark = random.random() * self.f_t * np.log10((nc+1)/self.max_iter) * (X_center - pigeons[i])
                    
                    velocities[i] = v_map + v_compass + v_landmark
                    pigeons[i] = pigeons[i] + velocities[i]
                else:
                    # Takipçiler - hiyerarşik öğrenme
                    for _ in range(self.s_l):
                        teacher_idx = random.randint(0, i-1)
                        dim = random.randint(0, n_vars-1)
                        pigeons[i, dim] = pigeons[teacher_idx, dim] + self.e * random.random()
                
                # Sınırları kontrol et
                pigeons[i] = np.clip(pigeons[i], bounds[0], bounds[1])
            
            # Popülasyonu azalt
            if nc < self.max_iter - 1:
                current_n = max(10, current_n - self.N_d)
        
        return pigeons[0]  # En iyi çözüm
    
    def pareto_ranking(self, fitness_values):
        """Basit Pareto sıralama"""
        n = len(fitness_values)
        ranks = np.ones(n)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # i, j tarafından domine ediliyorsa
                    if all(fitness_values[j] <= fitness_values[i]) and any(fitness_values[j] < fitness_values[i]):
                        ranks[i] += 1
        
        return ranks

class UAVFlockingSystem:
    """Ana UAV sürü sistemi"""
    def __init__(self, n_uavs: int, obstacles: List[Obstacle]):
        self.n_uavs = n_uavs
        self.obstacles = obstacles
        
        # Modüller
        self.uav_models = [UAVModel() for _ in range(n_uavs)]
        self.flocking_controller = FlockingController()
        self.obstacle_avoidance = ObstacleAvoidance()
        self.optimizer = ModifiedMPIO()
        
        # Başlangıç durumları
        self.states = []
        for i in range(n_uavs):
            state = UAVState(
                x=random.uniform(0, 30),
                y=random.uniform(100, 120),
                h=random.uniform(20, 100),
                v_xy=10.0,
                psi=0.0,
                lambda_=0.0
            )
            self.states.append(state)
        
        # Ağırlık vektörleri
        self.weights = [np.ones(n_uavs) for _ in range(n_uavs)]
        
    def step(self, dt: float):
        """Bir zaman adımı simülasyonu"""
        new_states = []
        
        for i in range(self.n_uavs):
            # Komşuları bul
            neighbors = []
            for j in range(self.n_uavs):
                if i != j:
                    neighbors.append((self.states[j], self.weights[i][j]))
            
            # Sürü kuvvetini hesapla
            flocking_force = self.flocking_controller.calculate_flocking_force(
                self.states[i], neighbors, self.weights[i]
            )
            
            # Engelleri tespit et
            detected_obs = self.obstacle_avoidance.detect_obstacles(
                self.states[i], self.obstacles
            )
            
            # Engel kaçınma
            if detected_obs:
                target_angle = self.obstacle_avoidance.find_largest_gap(
                    self.states[i], self.obstacles, detected_obs
                )
                # Basit engel kaçınma kuvveti
                avoid_force = 10.0 * np.array([np.cos(target_angle), np.sin(target_angle), 0])
            else:
                avoid_force = np.zeros(3)
            
            # Toplam kuvvet
            total_force = flocking_force + avoid_force
            
            # Kontrol girişlerini hesapla
            v_xy_c = self.states[i].v_xy + dt * (total_force[0] * np.cos(self.states[i].psi) + 
                                                  total_force[1] * np.sin(self.states[i].psi))
            
            psi_c = self.states[i].psi + dt * (total_force[1] * np.cos(self.states[i].psi) - 
                                                total_force[0] * np.sin(self.states[i].psi)) / self.states[i].v_xy
            
            h_c = self.flocking_controller.h_e
            
            # UAV durumunu güncelle
            new_state = self.uav_models[i].update(
                self.states[i], v_xy_c, psi_c, h_c, dt
            )
            new_states.append(new_state)
        
        self.states = new_states
    
    def simulate(self, duration: float, dt: float):
        """Simülasyonu çalıştır"""
        steps = int(duration / dt)
        history = []
        
        for _ in range(steps):
            self.step(dt)
            history.append([(s.x, s.y, s.h) for s in self.states])
        
        return history

# Örnek kullanım
if __name__ == "__main__":
    # Engelleri tanımla
    obstacles = [
        Obstacle(120, 120, 5),
        Obstacle(240, 75, 5),
        Obstacle(350, 40, 5),
        Obstacle(240, 155, 5),
        Obstacle(360, 110, 5),
        Obstacle(350, 180, 5)
    ]
    
    # Sistemi oluştur
    system = UAVFlockingSystem(n_uavs=5, obstacles=obstacles)
    
    # Simülasyonu çalıştır
    history = system.simulate(duration=50.0, dt=0.5)
    
    # Sonuçları görselleştir
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # UAV yollarını çiz
    for i in range(system.n_uavs):
        trajectory = [(h[i][0], h[i][1], h[i][2]) for h in history]
        xs, ys, zs = zip(*trajectory)
        ax.plot(xs, ys, zs, label=f'UAV {i+1}')
    
    # Engelleri çiz
    for obs in obstacles:
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = obs.radius * np.outer(np.cos(u), np.sin(v)) + obs.x
        y = obs.radius * np.outer(np.sin(u), np.sin(v)) + obs.y
        z = 100 * np.outer(np.ones(np.size(u)), np.cos(v)) + 50
        ax.plot_surface(x, y, z, alpha=0.3, color='red')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title('UAV Flocking with Obstacle Avoidance')
    ax.legend()
    
    plt.show()
    
    print("Simülasyon tamamlandı!")