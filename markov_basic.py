"""
Markov Chain Basic Implementation
Este módulo estima la matriz de transición P y simula cadenas de Markov
para el análisis de headways en sistemas de transporte público.
Versión mejorada con análisis específico para Metrobús CDMX.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import seaborn as sns
from pathlib import Path
import json

class MarkovChain:
    """Implementación básica de cadenas de Markov para análisis de headways"""
    
    def __init__(self):
        self.transition_matrix = None
        self.states = None
        self.state_to_index = None
        self.index_to_state = None
        
    def estimate_transition_matrix(self, sequences: List[List]) -> np.ndarray:
        """
        Estima la matriz de transición P a partir de secuencias observadas
        
        Args:
            sequences: Lista de secuencias de estados observados
            
        Returns:
            Matriz de transición estimada
        """
        # Obtener todos los estados únicos
        all_states = set()
        for seq in sequences:
            all_states.update(seq)
        
        self.states = sorted(list(all_states))
        self.state_to_index = {state: i for i, state in enumerate(self.states)}
        self.index_to_state = {i: state for i, state in enumerate(self.states)}
        
        n_states = len(self.states)
        transition_counts = np.zeros((n_states, n_states))
        
        # Contar transiciones
        for seq in sequences:
            for i in range(len(seq) - 1):
                current_state = self.state_to_index[seq[i]]
                next_state = self.state_to_index[seq[i + 1]]
                transition_counts[current_state, next_state] += 1
        
        # Normalizar para obtener probabilidades
        row_sums = transition_counts.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Evitar división por cero
        self.transition_matrix = transition_counts / row_sums[:, np.newaxis]
        
        return self.transition_matrix
    
    def simulate_chain(self, initial_state, n_steps: int) -> List:
        """
        Simula una cadena de Markov por n_steps pasos
        
        Args:
            initial_state: Estado inicial
            n_steps: Número de pasos a simular
            
        Returns:
            Lista de estados simulados
        """
        if self.transition_matrix is None:
            raise ValueError("Debe estimar la matriz de transición primero")
        
        current_state_idx = self.state_to_index[initial_state]
        sequence = [initial_state]
        
        for _ in range(n_steps):
            # Obtener probabilidades de transición del estado actual
            probs = self.transition_matrix[current_state_idx]
            
            # Seleccionar siguiente estado según las probabilidades
            next_state_idx = np.random.choice(len(self.states), p=probs)
            next_state = self.index_to_state[next_state_idx]
            
            sequence.append(next_state)
            current_state_idx = next_state_idx
        
        return sequence
    
    def simulate_multiple_chains(self, initial_states: List, n_steps: int, n_simulations: int) -> List[List]:
        """
        Simula múltiples cadenas de Markov
        
        Args:
            initial_states: Lista de estados iniciales para cada simulación
            n_steps: Número de pasos por simulación
            n_simulations: Número de simulaciones
            
        Returns:
            Lista de secuencias simuladas
        """
        simulations = []
        for i in range(n_simulations):
            initial_state = initial_states[i % len(initial_states)]
            sequence = self.simulate_chain(initial_state, n_steps)
            simulations.append(sequence)
        
        return simulations
    
    def get_stationary_distribution(self) -> np.ndarray:
        """
        Calcula la distribución estacionaria de la cadena de Markov
        
        Returns:
            Vector de distribución estacionaria
        """
        if self.transition_matrix is None:
            raise ValueError("Debe estimar la matriz de transición primero")
        
        # Resolver (P^T - I)π = 0 con la restricción sum(π) = 1
        n = self.transition_matrix.shape[0]
        A = self.transition_matrix.T - np.eye(n)
        A = np.vstack([A[:-1], np.ones(n)])
        b = np.zeros(n)
        b[-1] = 1
        
        stationary_dist = np.linalg.lstsq(A, b, rcond=None)[0]
        return stationary_dist

def load_gtfs_data(data_path: str) -> Dict[str, pd.DataFrame]:
    """
    Carga los datos GTFS desde archivos CSV
    
    Args:
        data_path: Ruta al directorio con archivos GTFS
        
    Returns:
        Diccionario con DataFrames de cada archivo GTFS
    """
    gtfs_files = [
        'agency.txt', 'routes.txt', 'trips.txt', 'calendar.txt',
        'shapes.txt', 'stops.txt', 'stop_times.txt'
    ]
    
    gtfs_data = {}
    for file in gtfs_files:
        try:
            file_path = f"{data_path}/{file}"
            gtfs_data[file.replace('.txt', '')] = pd.read_csv(file_path)
            print(f"Cargado {file}: {len(gtfs_data[file.replace('.txt', '')])} registros")
        except FileNotFoundError:
            print(f"Archivo {file} no encontrado")
    
    return gtfs_data

def calculate_headways(stop_times_df: pd.DataFrame, route_id: str = None) -> pd.DataFrame:
    """
    Calcula los headways (intervalos entre vehículos) a partir de stop_times
    
    Args:
        stop_times_df: DataFrame con tiempos de parada
        route_id: ID de ruta específica (opcional)
        
    Returns:
        DataFrame con headways calculados
    """
    # Convertir tiempos a datetime
    stop_times_df['arrival_datetime'] = pd.to_datetime(stop_times_df['arrival_time'], format='%H:%M:%S', errors='coerce')
    
    # Agrupar por parada y calcular diferencias de tiempo
    headways_list = []
    
    for stop_id in stop_times_df['stop_id'].unique():
        stop_data = stop_times_df[stop_times_df['stop_id'] == stop_id].copy()
        stop_data = stop_data.sort_values('arrival_datetime')
        
        # Calcular diferencias de tiempo consecutivas
        stop_data['headway_minutes'] = stop_data['arrival_datetime'].diff().dt.total_seconds() / 60
        
        # Filtrar headways válidos (entre 1 y 120 minutos)
        valid_headways = stop_data[(stop_data['headway_minutes'] >= 1) & 
                                 (stop_data['headway_minutes'] <= 120)]
        
        for _, row in valid_headways.iterrows():
            headways_list.append({
                'stop_id': stop_id,
                'trip_id': row['trip_id'],
                'headway_minutes': row['headway_minutes'],
                'arrival_time': row['arrival_time']
            })
    
    return pd.DataFrame(headways_list)

if __name__ == "__main__":
    # Ejemplo de uso
    data_path = "data"
    
    # Cargar datos GTFS
    print("Cargando datos GTFS...")
    gtfs_data = load_gtfs_data(data_path)
    
    # Calcular headways
    print("Calculando headways...")
    headways_df = calculate_headways(gtfs_data['stop_times'])
    
    print(f"Headways calculados: {len(headways_df)} registros")
    print(f"Headway promedio: {headways_df['headway_minutes'].mean():.2f} minutos")
    print(f"Headway mediano: {headways_df['headway_minutes'].median():.2f} minutos")
