"""
Versión optimizada básica del analizador de Markov 
Usa paralelización CPU sin dependencias adicionales
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
import sys
import os

class FastMetrobusMarkovAnalyzer:
    """Analizador optimizado de cadenas de Markov usando solo bibliotecas estándar"""
    
    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.stops_df = None
        self.routes_df = None
        self.stop_times_df = None
        self.trips_df = None
        self.transition_matrix = None
        self.states = None
        
        # Detectar número de CPUs
        self.n_cores = mp.cpu_count()
        st.info(f"💻 Detectados {self.n_cores} núcleos de CPU para paralelización")
    
    def load_data(self):
        """Carga los datos GTFS del Metrobús"""
        try:
            stops_file = self.data_path / "stops.txt"
            routes_file = self.data_path / "routes.txt"
            stop_times_file = self.data_path / "stop_times.txt"
            trips_file = self.data_path / "trips.txt"
            
            missing_files = []
            for file_path, name in [(stops_file, "stops.txt"), (routes_file, "routes.txt"), 
                                   (stop_times_file, "stop_times.txt"), (trips_file, "trips.txt")]:
                if not file_path.exists():
                    missing_files.append(str(file_path))
            
            if missing_files:
                st.error(f"❌ Archivos no encontrados: {', '.join(missing_files)}")
                return False
            
            # Carga optimizada con tipos de datos específicos para reducir memoria
            st.info("📊 Cargando datos con optimización de memoria...")
            
            # Cargar con tipos optimizados
            dtypes = {
                'trip_id': 'category',
                'stop_id': 'category',
                'stop_sequence': 'int16'
            }
            
            self.stops_df = pd.read_csv(stops_file)
            self.routes_df = pd.read_csv(routes_file)
            self.stop_times_df = pd.read_csv(stop_times_file, dtype=dtypes)
            self.trips_df = pd.read_csv(trips_file)
            
            st.success(f"✅ Datos cargados: {len(self.stops_df)} estaciones, {len(self.routes_df)} rutas")
            return True
            
        except Exception as e:
            st.error(f"Error cargando datos: {e}")
            return False
    
    def is_data_loaded(self):
        """Verifica si los datos han sido cargados correctamente"""
        return (self.stops_df is not None and 
                self.routes_df is not None and 
                self.stop_times_df is not None and 
                self.trips_df is not None)
    
    def preprocess_data_fast(self, sample_size: Optional[int] = None, use_parallel: bool = True) -> List[List[str]]:
        """Preprocesa los datos de forma optimizada"""
        if not self.is_data_loaded():
            raise ValueError("Los datos no han sido cargados.")
        
        # Optimización 1: Filtrar datos innecesarios temprano
        df = self.stop_times_df[['trip_id', 'stop_id', 'stop_sequence']].copy()
        
        unique_trips = df['trip_id'].unique()
        
        if sample_size is not None and sample_size < len(unique_trips):
            unique_trips = unique_trips[:sample_size]
            df = df[df['trip_id'].isin(unique_trips)]
            st.info(f"🔄 Procesando {sample_size:,} viajes de {len(self.stop_times_df['trip_id'].unique()):,} totales...")
        else:
            st.info(f"🔄 Procesando TODOS los {len(unique_trips):,} viajes disponibles...")
        
        if use_parallel and len(unique_trips) > 1000:
            return self._preprocess_parallel(df, unique_trips)
        else:
            return self._preprocess_sequential(df, unique_trips)
    
    def _preprocess_sequential(self, df: pd.DataFrame, unique_trips: np.ndarray) -> List[List[str]]:
        """Procesamiento secuencial optimizado"""
        st.info("💻 Usando procesamiento secuencial optimizado...")
        
        trip_sequences = []
        progress_bar = st.progress(0)
        
        # Optimización: Pre-agrupar los datos
        grouped = df.groupby('trip_id')
        total_trips = len(unique_trips)
        
        for i, trip_id in enumerate(unique_trips):
            if i % 1000 == 0:
                progress_bar.progress(i / total_trips)
            
            try:
                group = grouped.get_group(trip_id)
                if len(group) > 1:
                    sequence = group.sort_values('stop_sequence')['stop_id'].tolist()
                    trip_sequences.append(sequence)
            except KeyError:
                continue  # Trip no encontrado en el grupo
        
        progress_bar.progress(1.0)
        st.success(f"✅ Procesados {len(trip_sequences):,} viajes válidos")
        return trip_sequences
    
    def _preprocess_parallel(self, df: pd.DataFrame, unique_trips: np.ndarray) -> List[List[str]]:
        """Procesamiento paralelo optimizado"""
        n_workers = min(self.n_cores, 8)  # Limitar workers para evitar overhead
        st.info(f"⚡ Usando procesamiento paralelo con {n_workers} workers...")
        
        # Dividir trips en chunks
        chunk_size = max(100, len(unique_trips) // n_workers)
        trip_chunks = [unique_trips[i:i + chunk_size] for i in range(0, len(unique_trips), chunk_size)]
        
        # Preparar datos para workers (convertir a tipos básicos para serialización)
        df_dict = {
            'trip_id': df['trip_id'].astype(str).tolist(),
            'stop_id': df['stop_id'].astype(str).tolist(), 
            'stop_sequence': df['stop_sequence'].tolist()
        }
        
        all_sequences = []
        progress_bar = st.progress(0)
        
        # Usar ThreadPoolExecutor para mejor rendimiento con I/O
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Enviar trabajos
            future_to_chunk = {
                executor.submit(process_chunk_worker, df_dict, chunk): chunk 
                for chunk in trip_chunks
            }
            
            # Recopilar resultados conforme van completándose
            completed = 0
            for future in as_completed(future_to_chunk):
                sequences = future.result()
                all_sequences.extend(sequences)
                completed += 1
                progress_bar.progress(completed / len(trip_chunks))
        
        st.success(f"✅ Procesados {len(all_sequences):,} viajes válidos con {n_workers} workers")
        return all_sequences
    
    def estimate_transition_matrix_fast(self, sequences: List[List[str]]) -> np.ndarray:
        """Estima la matriz de transición de forma optimizada"""
        st.info("🧮 Calculando matriz de transición...")
        
        # Optimización 1: Usar set para estados únicos más rápido
        all_stops = set()
        for seq in sequences:
            all_stops.update(seq)
        
        self.states = sorted(list(all_stops))
        n_states = len(self.states)
        
        st.info(f"📊 Matriz de {n_states}x{n_states} estaciones")
        
        # Optimización 2: Usar diccionario para mapeo O(1)
        state_to_idx = {state: i for i, state in enumerate(self.states)}
        
        # Optimización 3: Pre-asignar matriz del tamaño correcto
        transition_counts = np.zeros((n_states, n_states), dtype=np.int32)
        
        # Optimización 4: Procesar en chunks si es muy grande
        if len(sequences) > 10000:
            chunk_size = 1000
            progress_bar = st.progress(0)
            
            for i in range(0, len(sequences), chunk_size):
                chunk = sequences[i:i + chunk_size]
                for seq in chunk:
                    for j in range(len(seq) - 1):
                        from_idx = state_to_idx[seq[j]]
                        to_idx = state_to_idx[seq[j + 1]]
                        transition_counts[from_idx, to_idx] += 1
                
                progress_bar.progress(min(1.0, (i + chunk_size) / len(sequences)))
        else:
            # Procesamiento directo para datasets pequeños
            for seq in sequences:
                for i in range(len(seq) - 1):
                    from_idx = state_to_idx[seq[i]]
                    to_idx = state_to_idx[seq[i + 1]]
                    transition_counts[from_idx, to_idx] += 1
        
        # Normalizar usando vectorización de numpy
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Evitar división por cero
        self.transition_matrix = transition_counts / row_sums
        
        st.success("✅ Matriz de transición calculada")
        return self.transition_matrix
    
    def get_optimization_stats(self):
        """Muestra estadísticas de optimización"""
        st.subheader("📊 Estadísticas de Optimización")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("CPU Cores", self.n_cores)
        
        with col2:
            if self.stop_times_df is not None:
                memory_usage = self.stop_times_df.memory_usage(deep=True).sum() / 1024**2
                st.metric("Memoria (MB)", f"{memory_usage:.1f}")
        
        with col3:
            if self.states is not None:
                st.metric("Estados", len(self.states))

def process_chunk_worker(df_dict: dict, trip_chunk: np.ndarray) -> List[List[str]]:
    """Worker function para procesamiento paralelo"""
    # Reconstruir DataFrame del diccionario
    df = pd.DataFrame(df_dict)
    
    sequences = []
    
    for trip_id in trip_chunk:
        trip_id_str = str(trip_id)
        trip_data = df[df['trip_id'] == trip_id_str]
        
        if len(trip_data) > 1:
            sequence = trip_data.sort_values('stop_sequence')['stop_id'].tolist()
            sequences.append(sequence)
    
    return sequences

# Función para benchmark
def benchmark_processing(analyzer, sample_sizes=[1000, 5000, 10000]):
    """Ejecuta benchmark de diferentes tamaños de muestra"""
    st.subheader("🏁 Benchmark de Rendimiento")
    
    results = []
    
    for size in sample_sizes:
        st.write(f"Probando con {size:,} viajes...")
        
        # Secuencial
        start_time = time.time()
        seq_sequential = analyzer.preprocess_data_fast(size, use_parallel=False)
        time_sequential = time.time() - start_time
        
        # Paralelo
        start_time = time.time()
        seq_parallel = analyzer.preprocess_data_fast(size, use_parallel=True)
        time_parallel = time.time() - start_time
        
        speedup = time_sequential / time_parallel if time_parallel > 0 else 1
        
        results.append({
            'Muestra': f"{size:,}",
            'Secuencial (s)': f"{time_sequential:.2f}",
            'Paralelo (s)': f"{time_parallel:.2f}",
            'Speedup': f"{speedup:.2f}x"
        })
    
    st.dataframe(pd.DataFrame(results))
    
    return results
