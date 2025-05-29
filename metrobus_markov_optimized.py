"""
Versión optimizada del analizador de Markov con paralelización GPU y CPU
Incluye múltiples estrategias de optimización
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import networkx as nx
from datetime import datetime, timedelta
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import psutil
import time

# Intentar importar bibliotecas para aceleración GPU
try:
    import cupy as cp
    HAS_CUPY = True
    st.info("🚀 CuPy disponible - Aceleración GPU habilitada")
except ImportError:
    HAS_CUPY = False
    st.info("⚡ CuPy no disponible - Usando CPU solamente")

try:
    import numba
    from numba import jit, cuda
    HAS_NUMBA = True
    st.info("⚡ Numba disponible - Compilación JIT habilitada")
except ImportError:
    HAS_NUMBA = False

try:
    import dask.dataframe as dd
    from dask.distributed import Client
    HAS_DASK = True
    st.info("🔄 Dask disponible - Procesamiento paralelo habilitado")
except ImportError:
    HAS_DASK = False

class OptimizedMetrobusMarkovAnalyzer:
    """Analizador optimizado de cadenas de Markov para datos del Metrobús"""
    
    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.stops_df = None
        self.routes_df = None
        self.stop_times_df = None
        self.trips_df = None
        self.transition_matrix = None
        self.states = None
        
        # Configuración de paralelización
        self.n_cores = psutil.cpu_count(logical=False)
        self.n_threads = psutil.cpu_count(logical=True)
        
        st.info(f"💻 Sistema detectado: {self.n_cores} núcleos físicos, {self.n_threads} threads")
    
    def load_data(self):
        """Carga los datos GTFS del Metrobús de forma optimizada"""
        try:
            stops_file = self.data_path / "stops.txt"
            routes_file = self.data_path / "routes.txt"
            stop_times_file = self.data_path / "stop_times.txt"
            trips_file = self.data_path / "trips.txt"
            
            # Verificar que los archivos existan
            missing_files = []
            for file_path, name in [(stops_file, "stops.txt"), (routes_file, "routes.txt"), 
                                   (stop_times_file, "stop_times.txt"), (trips_file, "trips.txt")]:
                if not file_path.exists():
                    missing_files.append(str(file_path))
            
            if missing_files:
                st.error(f"❌ Archivos no encontrados: {', '.join(missing_files)}")
                return False
            
            # Usar Dask para carga paralela si está disponible
            if HAS_DASK:
                st.info("🔄 Cargando datos con Dask (paralelo)...")
                try:
                    self.stop_times_df = dd.read_csv(stop_times_file).compute()
                    self.stops_df = pd.read_csv(stops_file)
                    self.routes_df = pd.read_csv(routes_file)
                    self.trips_df = pd.read_csv(trips_file)
                except Exception as e:
                    st.warning(f"⚠️ Error con Dask, usando pandas: {e}")
                    self._load_with_pandas()
            else:
                self._load_with_pandas()
            
            st.success(f"✅ Datos cargados: {len(self.stops_df)} estaciones, {len(self.routes_df)} rutas")
            return True
            
        except Exception as e:
            st.error(f"Error cargando datos: {e}")
            return False
    
    def _load_with_pandas(self):
        """Carga con pandas tradicional"""
        st.info("📊 Cargando datos con pandas...")
        self.stops_df = pd.read_csv(self.data_path / "stops.txt")
        self.routes_df = pd.read_csv(self.data_path / "routes.txt")
        self.stop_times_df = pd.read_csv(self.data_path / "stop_times.txt")
        self.trips_df = pd.read_csv(self.data_path / "trips.txt")
    
    def is_data_loaded(self):
        """Verifica si los datos han sido cargados correctamente"""
        return (self.stops_df is not None and 
                self.routes_df is not None and 
                self.stop_times_df is not None and 
                self.trips_df is not None)
    
    def preprocess_data_vectorized(self, sample_size: Optional[int] = None) -> List[List[str]]:
        """Preprocesa los datos usando operaciones vectorizadas"""
        if not self.is_data_loaded():
            raise ValueError("Los datos no han sido cargados.")
        
        st.info("🚀 Usando procesamiento vectorizado optimizado...")
        
        # Filtrar por sample_size si se especifica
        if sample_size is not None:
            unique_trips = self.stop_times_df['trip_id'].unique()[:sample_size]
            filtered_df = self.stop_times_df[self.stop_times_df['trip_id'].isin(unique_trips)]
        else:
            filtered_df = self.stop_times_df
        
        # Operación vectorizada para obtener secuencias
        trip_sequences = []
        
        # Agrupar y procesar de forma vectorizada
        grouped = filtered_df.groupby('trip_id')
        
        progress_bar = st.progress(0)
        total_groups = len(grouped)
        
        for i, (trip_id, group) in enumerate(grouped):
            if i % 1000 == 0:
                progress_bar.progress(i / total_groups)
            
            if len(group) > 1:
                # Ordenar por stop_sequence y obtener la secuencia
                sequence = group.sort_values('stop_sequence')['stop_id'].tolist()
                trip_sequences.append(sequence)
        
        progress_bar.progress(1.0)
        st.success(f"✅ Procesados {len(trip_sequences):,} viajes válidos")
        
        return trip_sequences
    
    def preprocess_data_parallel(self, sample_size: Optional[int] = None, n_workers: Optional[int] = None) -> List[List[str]]:
        """Preprocesa los datos usando paralelización con multiprocessing"""
        if not self.is_data_loaded():
            raise ValueError("Los datos no han sido cargados.")
        
        if n_workers is None:
            n_workers = min(self.n_cores, 8)  # Limitar para evitar sobrecarga
        
        st.info(f"⚡ Usando procesamiento paralelo con {n_workers} workers...")
        
        # Preparar datos
        if sample_size is not None:
            unique_trips = self.stop_times_df['trip_id'].unique()[:sample_size]
        else:
            unique_trips = self.stop_times_df['trip_id'].unique()
        
        # Dividir el trabajo en chunks
        chunk_size = max(1, len(unique_trips) // n_workers)
        trip_chunks = [unique_trips[i:i + chunk_size] for i in range(0, len(unique_trips), chunk_size)]
        
        # Procesar en paralelo
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for chunk in trip_chunks:
                future = executor.submit(self._process_trip_chunk, chunk)
                futures.append(future)
            
            # Recopilar resultados
            all_sequences = []
            progress_bar = st.progress(0)
            
            for i, future in enumerate(futures):
                sequences = future.result()
                all_sequences.extend(sequences)
                progress_bar.progress((i + 1) / len(futures))
        
        st.success(f"✅ Procesados {len(all_sequences):,} viajes válidos con paralelización")
        return all_sequences
    
    def _process_trip_chunk(self, trip_ids: np.ndarray) -> List[List[str]]:
        """Procesa un chunk de trip_ids (función auxiliar para paralelización)"""
        sequences = []
        
        for trip_id in trip_ids:
            trip_data = self.stop_times_df[self.stop_times_df['trip_id'] == trip_id]
            if len(trip_data) > 1:
                sequence = trip_data.sort_values('stop_sequence')['stop_id'].tolist()
                sequences.append(sequence)
        
        return sequences
    
    @staticmethod
    @numba.jit(nopython=True) if HAS_NUMBA else lambda f: f
    def _count_transitions_numba(sequences_flat, state_indices, n_states):
        """Cuenta transiciones usando Numba para aceleración"""
        transition_counts = np.zeros((n_states, n_states), dtype=np.int64)
        
        i = 0
        while i < len(sequences_flat) - 1:
            if sequences_flat[i] != -1 and sequences_flat[i+1] != -1:  # -1 marca fin de secuencia
                from_idx = state_indices[sequences_flat[i]]
                to_idx = state_indices[sequences_flat[i+1]]
                transition_counts[from_idx, to_idx] += 1
            i += 1
        
        return transition_counts
    
    def estimate_transition_matrix_gpu(self, sequences: List[List[str]]) -> np.ndarray:
        """Estima la matriz de transición usando GPU si está disponible"""
        # Obtener estados únicos
        all_stops = set()
        for seq in sequences:
            all_stops.update(seq)
        
        self.states = sorted(list(all_stops))
        state_to_idx = {state: i for i, state in enumerate(self.states)}
        n_states = len(self.states)
        
        st.info(f"🧮 Calculando matriz de transición para {n_states} estaciones...")
        
        if HAS_CUPY and n_states > 100:  # Solo usar GPU para matrices grandes
            st.info("🚀 Usando aceleración GPU con CuPy...")
            return self._estimate_with_cupy(sequences, state_to_idx, n_states)
        elif HAS_NUMBA:
            st.info("⚡ Usando aceleración Numba...")
            return self._estimate_with_numba(sequences, state_to_idx, n_states)
        else:
            st.info("💻 Usando procesamiento CPU estándar...")
            return self._estimate_standard(sequences, state_to_idx, n_states)
    
    def _estimate_with_cupy(self, sequences, state_to_idx, n_states):
        """Implementación con CuPy (GPU)"""
        # Crear matriz de conteos en GPU
        transition_counts = cp.zeros((n_states, n_states), dtype=cp.int32)
        
        # Procesar secuencias
        for seq in sequences:
            for i in range(len(seq) - 1):
                from_idx = state_to_idx[seq[i]]
                to_idx = state_to_idx[seq[i + 1]]
                transition_counts[from_idx, to_idx] += 1
        
        # Normalizar
        row_sums = cp.sum(transition_counts, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_matrix = transition_counts / row_sums
        
        # Convertir de vuelta a numpy
        self.transition_matrix = cp.asnumpy(transition_matrix)
        return self.transition_matrix
    
    def _estimate_with_numba(self, sequences, state_to_idx, n_states):
        """Implementación con Numba (CPU optimizado)"""
        # Aplanar secuencias para Numba
        sequences_flat = []
        state_indices = np.array([state_to_idx[state] for state in self.states])
        
        for seq in sequences:
            for stop in seq:
                sequences_flat.append(state_to_idx[stop])
            sequences_flat.append(-1)  # Marcador de fin de secuencia
        
        sequences_array = np.array(sequences_flat)
        
        # Usar función optimizada con Numba
        transition_counts = self._count_transitions_numba(sequences_array, state_indices, n_states)
        
        # Normalizar
        row_sums = transition_counts.sum(axis=1)
        row_sums[row_sums == 0] = 1
        self.transition_matrix = transition_counts / row_sums[:, np.newaxis]
        
        return self.transition_matrix
    
    def _estimate_standard(self, sequences, state_to_idx, n_states):
        """Implementación estándar (fallback)"""
        transition_counts = np.zeros((n_states, n_states))
        
        for seq in sequences:
            for i in range(len(seq) - 1):
                from_idx = state_to_idx[seq[i]]
                to_idx = state_to_idx[seq[i + 1]]
                transition_counts[from_idx, to_idx] += 1
        
        # Normalizar
        row_sums = transition_counts.sum(axis=1)
        row_sums[row_sums == 0] = 1
        self.transition_matrix = transition_counts / row_sums[:, np.newaxis]
        
        return self.transition_matrix
    
    def benchmark_methods(self, sample_size: int = 1000):
        """Compara el rendimiento de diferentes métodos"""
        st.subheader("🏁 Benchmark de Métodos de Procesamiento")
        
        methods = []
        times = []
        
        # Método estándar
        start_time = time.time()
        sequences_standard = self.preprocess_data_vectorized(sample_size)
        time_standard = time.time() - start_time
        methods.append("Vectorizado")
        times.append(time_standard)
        
        # Método paralelo
        start_time = time.time()
        sequences_parallel = self.preprocess_data_parallel(sample_size)
        time_parallel = time.time() - start_time
        methods.append("Paralelo")
        times.append(time_parallel)
        
        # Mostrar resultados
        benchmark_df = pd.DataFrame({
            'Método': methods,
            'Tiempo (s)': times,
            'Speedup': [times[0] / t for t in times]
        })
        
        st.dataframe(benchmark_df)
        
        # Gráfico de comparación
        fig = px.bar(benchmark_df, x='Método', y='Tiempo (s)', 
                     title="Comparación de Rendimiento")
        st.plotly_chart(fig, use_container_width=True)
        
        return sequences_standard

# Funciones de utilidad para paralelización
def process_trip_chunk_static(args):
    """Función estática para procesamiento paralelo"""
    stop_times_df, trip_ids = args
    sequences = []
    
    for trip_id in trip_ids:
        trip_data = stop_times_df[stop_times_df['trip_id'] == trip_id]
        if len(trip_data) > 1:
            sequence = trip_data.sort_values('stop_sequence')['stop_id'].tolist()
            sequences.append(sequence)
    
    return sequences
