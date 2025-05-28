"""
MDP Holding Problem Implementation
Este módulo define estados, acciones, recompensas y implementa value iteration
para el problema de retención/holding en sistemas de transporte público.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Sequence
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

class TransportMDP:
    """
    Proceso de Decisión de Markov para optimización de holding en transporte público
    """
    
    def __init__(self, headway_bins: Sequence[float], holding_actions: Sequence[float]):
        """
        Inicializa el MDP
        
        Args:
            headway_bins: Intervalos para discretizar headways (en minutos)
            holding_actions: Acciones de retención disponibles (en minutos)
        """
        self.headway_bins = headway_bins
        self.holding_actions = holding_actions
        self.n_states = len(headway_bins) - 1
        self.n_actions = len(holding_actions)
        
        # Matrices del MDP
        self.transition_probs = None
        self.reward_matrix = None
        self.value_function = None
        self.policy = None
        
        # Parámetros del modelo
        self.passenger_arrival_rate = 2.0  # pasajeros por minuto
        self.waiting_cost = 1.0  # costo por minuto de espera
        self.holding_cost = 0.5  # costo por minuto de retención
        self.capacity = 150  # capacidad del vehículo
        
    def discretize_headways(self, headways: np.ndarray) -> np.ndarray:
        """
        Discretiza headways continuos en estados discretos
        
        Args:
            headways: Array de headways en minutos
            
        Returns:
            Array de estados discretizados
        """
        return np.digitize(headways, self.headway_bins) - 1
    
    def estimate_transition_probabilities(self, headway_sequences: List[List[int]]) -> np.ndarray:
        """
        Estima probabilidades de transición basadas en datos históricos
        
        Args:
            headway_sequences: Secuencias de estados de headway
            
        Returns:
            Tensor de probabilidades de transición [state, action, next_state]
        """
        # Inicializar tensor de transiciones
        self.transition_probs = np.zeros((self.n_states, self.n_actions, self.n_states))
        
        # Contar transiciones para cada acción
        transition_counts = np.zeros((self.n_states, self.n_actions, self.n_states))
        
        for sequence in headway_sequences:
            for t in range(len(sequence) - 1):
                current_state = min(sequence[t], self.n_states - 1)
                next_state = min(sequence[t + 1], self.n_states - 1)
                
                # Asumimos acción aleatoria por ahora (se puede mejorar con datos reales)
                action = np.random.randint(0, self.n_actions)
                transition_counts[current_state, action, next_state] += 1
        
        # Normalizar para obtener probabilidades
        for s in range(self.n_states):
            for a in range(self.n_actions):
                total = transition_counts[s, a, :].sum()
                if total > 0:
                    self.transition_probs[s, a, :] = transition_counts[s, a, :] / total
                else:
                    # Distribución uniforme si no hay datos
                    self.transition_probs[s, a, :] = 1.0 / self.n_states
        
        return self.transition_probs
    
    def calculate_reward_matrix(self) -> np.ndarray:
        """
        Calcula la matriz de recompensas R(s, a)
        
        Returns:
            Matriz de recompensas [state, action]
        """
        self.reward_matrix = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            # Headway promedio para este estado
            headway = (self.headway_bins[s] + self.headway_bins[s + 1]) / 2
            
            for a in range(self.n_actions):
                holding_time = self.holding_actions[a]
                
                # Costo de espera de pasajeros
                passengers_waiting = headway * self.passenger_arrival_rate
                waiting_cost = passengers_waiting * headway * self.waiting_cost
                
                # Costo de retención
                holding_cost = holding_time * self.holding_cost
                
                # Beneficio por regularidad (reducir variabilidad)
                regularity_bonus = -abs(headway - 10) * 0.1  # Penalizar desviación de 10 min
                
                # Recompensa total (negativa porque minimizamos costos)
                self.reward_matrix[s, a] = -(waiting_cost + holding_cost) + regularity_bonus
        
        return self.reward_matrix
    
    def value_iteration(self, gamma: float = 0.95, epsilon: float = 1e-6, max_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implementa el algoritmo de value iteration
        
        Args:
            gamma: Factor de descuento
            epsilon: Criterio de convergencia
            max_iterations: Máximo número de iteraciones
            
        Returns:
            Tupla (función de valor, política óptima)
        """
        if self.transition_probs is None or self.reward_matrix is None:
            raise ValueError("Debe calcular probabilidades de transición y recompensas primero")
        
        # Inicializar función de valor
        V = np.zeros(self.n_states)
        V_new = np.zeros(self.n_states)
        
        for iteration in range(max_iterations):
            for s in range(self.n_states):
                # Calcular valor para cada acción
                action_values = np.zeros(self.n_actions)
                
                for a in range(self.n_actions):
                    # Recompensa inmediata
                    immediate_reward = self.reward_matrix[s, a]
                    
                    # Valor esperado del siguiente estado
                    expected_future_value = np.sum(
                        self.transition_probs[s, a, :] * V
                    )
                    
                    action_values[a] = immediate_reward + gamma * expected_future_value
                
                # Tomar el máximo sobre acciones
                V_new[s] = np.max(action_values)
            
            # Verificar convergencia
            if np.max(np.abs(V_new - V)) < epsilon:
                print(f"Value iteration convergió en {iteration + 1} iteraciones")
                break
            
            V = V_new.copy()
        
        # Extraer política óptima
        policy = np.zeros(self.n_states, dtype=int)
        
        for s in range(self.n_states):
            action_values = np.zeros(self.n_actions)
            
            for a in range(self.n_actions):
                immediate_reward = self.reward_matrix[s, a]
                expected_future_value = np.sum(
                    self.transition_probs[s, a, :] * V
                )
                action_values[a] = immediate_reward + gamma * expected_future_value
            
            policy[s] = np.argmax(action_values)
        
        self.value_function = V
        self.policy = policy
        
        return V, policy
    
    def simulate_mdp_policy(self, initial_states: List[int], n_steps: int, n_simulations: int) -> List[List[int]]:
        """
        Simula el MDP usando la política óptima
        
        Args:
            initial_states: Estados iniciales para las simulaciones
            n_steps: Número de pasos por simulación
            n_simulations: Número de simulaciones
            
        Returns:
            Lista de secuencias de estados simuladas
        """
        if self.policy is None:
            raise ValueError("Debe ejecutar value iteration primero")
        
        simulations = []
        
        for i in range(n_simulations):
            initial_state = initial_states[i % len(initial_states)]
            sequence = self._simulate_single_trajectory(initial_state, n_steps)
            simulations.append(sequence)
        
        return simulations
    
    def _simulate_single_trajectory(self, initial_state: int, n_steps: int) -> List[int]:
        """
        Simula una trayectoria individual usando la política óptima
        """
        trajectory = [initial_state]
        current_state = initial_state
        
        for _ in range(n_steps):
            # Obtener acción según política óptima
            action = self.policy[current_state]
            
            # Transición al siguiente estado
            next_state_probs = self.transition_probs[current_state, action, :]
            next_state = np.random.choice(self.n_states, p=next_state_probs)
            
            trajectory.append(next_state)
            current_state = next_state
        
        return trajectory
    
    def evaluate_policy(self, simulations: List[List[int]]) -> Dict[str, float]:
        """
        Evalúa el desempeño de la política simulada
        
        Args:
            simulations: Lista de trayectorias simuladas
            
        Returns:
            Diccionario con métricas de evaluación
        """
        all_states = []
        for sim in simulations:
            all_states.extend(sim)
        
        # Convertir estados a headways
        headways = []
        for state in all_states:
            if state < self.n_states:
                headway = (self.headway_bins[state] + self.headway_bins[state + 1]) / 2
                headways.append(headway)
        
        headways = np.array(headways)
        
        metrics = {
            'mean_headway': np.mean(headways),
            'headway_variance': np.var(headways),
            'headway_std': np.std(headways),
            'coefficient_of_variation': np.std(headways) / np.mean(headways),
            'total_simulations': len(simulations),
            'total_steps': len(all_states)
        }
        
        return metrics
    
    def plot_policy(self):
        """
        Visualiza la política óptima
        """
        if self.policy is None:
            raise ValueError("Debe ejecutar value iteration primero")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Política óptima
        state_centers = [(self.headway_bins[i] + self.headway_bins[i + 1]) / 2 
                        for i in range(self.n_states)]
        optimal_actions = [self.holding_actions[self.policy[s]] for s in range(self.n_states)]
        
        ax1.bar(range(self.n_states), optimal_actions)
        ax1.set_xlabel('Estado (Headway discretizado)')
        ax1.set_ylabel('Acción óptima (minutos de retención)')
        ax1.set_title('Política Óptima de Retención')
        ax1.set_xticks(range(self.n_states))
        ax1.set_xticklabels([f'{state_centers[i]:.1f}' for i in range(self.n_states)], rotation=45)
        
        # Función de valor
        ax2.plot(range(self.n_states), self.value_function, 'o-')
        ax2.set_xlabel('Estado (Headway discretizado)')
        ax2.set_ylabel('Función de Valor')
        ax2.set_title('Función de Valor Óptima')
        ax2.set_xticks(range(self.n_states))
        ax2.set_xticklabels([f'{state_centers[i]:.1f}' for i in range(self.n_states)], rotation=45)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Ejemplo de uso
      # Definir espacios de estados y acciones
    headway_bins = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0]  # Bins en minutos
    holding_actions = [0.0, 1.0, 2.0, 3.0, 5.0]  # Acciones de retención en minutos
    
    # Crear MDP
    mdp = TransportMDP(headway_bins, holding_actions)
    
    # Generar datos sintéticos para prueba
    print("Generando datos sintéticos...")
    n_sequences = 100
    sequence_length = 50
    synthetic_sequences = []
    
    for _ in range(n_sequences):
        # Generar secuencia sintética de estados
        sequence = np.random.randint(0, mdp.n_states, sequence_length)
        synthetic_sequences.append(sequence.tolist())
    
    # Estimar probabilidades de transición
    print("Estimando probabilidades de transición...")
    mdp.estimate_transition_probabilities(synthetic_sequences)
    
    # Calcular matriz de recompensas
    print("Calculando matriz de recompensas...")
    mdp.calculate_reward_matrix()
    
    # Ejecutar value iteration
    print("Ejecutando value iteration...")
    V, policy = mdp.value_iteration()
    
    print(f"Política óptima: {policy}")
    print(f"Función de valor: {V}")
    
    # Simular con política óptima
    print("Simulando con política óptima...")
    initial_states = [0, 1, 2, 3, 4]
    simulations = mdp.simulate_mdp_policy(initial_states, 100, 1000)
    
    # Evaluar desempeño
    metrics = mdp.evaluate_policy(simulations)
    print("Métricas de evaluación:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
