"""
Aplicación interactiva para análisis de cadenas de Markov del Metrobús CDMX
Analiza patrones de transición entre estaciones usando datos GTFS
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
from typing import Dict, List, Tuple
import networkx as nx
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(
    page_title="Metrobús CDMX - Análisis Markov",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MetrobusMarkovAnalyzer:
    """Analizador de cadenas de Markov para datos del Metrobús"""
    
    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.stops_df = None
        self.routes_df = None
        self.stop_times_df = None
        self.trips_df = None
        self.transition_matrix = None
        self.states = None
    
    @st.cache_data
    def load_data(_self):
        """Carga los datos GTFS del Metrobús"""
        try:
            # Usar rutas relativas desde el directorio actual
            stops_file = _self.data_path / "stops.txt"
            routes_file = _self.data_path / "routes.txt"
            stop_times_file = _self.data_path / "stop_times.txt"
            trips_file = _self.data_path / "trips.txt"
            
            # Verificar que los archivos existan
            missing_files = []
            for file_path, name in [(stops_file, "stops.txt"), (routes_file, "routes.txt"), 
                                   (stop_times_file, "stop_times.txt"), (trips_file, "trips.txt")]:
                if not file_path.exists():
                    missing_files.append(str(file_path))
            
            if missing_files:
                st.error(f"❌ Archivos no encontrados: {', '.join(missing_files)}")
                st.error(f"📁 Directorio actual: {Path.cwd()}")
                st.error(f"🔍 Buscando en: {_self.data_path.absolute()}")
                return False
            
            _self.stops_df = pd.read_csv(stops_file)
            _self.routes_df = pd.read_csv(routes_file)
            _self.stop_times_df = pd.read_csv(stop_times_file)
            _self.trips_df = pd.read_csv(trips_file)
            
            st.success(f"✅ Datos cargados: {len(_self.stops_df)} estaciones, {len(_self.routes_df)} rutas")
            return True
        except Exception as e:
            st.error(f"Error cargando datos: {e}")
            st.error(f"Directorio actual: {Path.cwd()}")
            st.error(f"Buscando en: {_self.data_path}")
            return False
    
    def preprocess_data(self):
        """Preprocesa los datos para análisis de Markov"""
        if self.stop_times_df is None:
            st.error("❌ Datos no cargados. Ejecute load_data() primero.")
            return []
        
        # Unir datos para obtener secuencias de paradas por viaje
        trip_sequences = []
        
        # Agrupar por trip_id y ordenar por stop_sequence
        trip_ids = self.stop_times_df['trip_id'].unique()[:100]  # Limitamos para demo
        progress_bar = st.progress(0)
        
        for i, trip_id in enumerate(trip_ids):
            trip_stops = self.stop_times_df[
                self.stop_times_df['trip_id'] == trip_id
            ].sort_values('stop_sequence')
            
            if len(trip_stops) > 1:
                stop_sequence = trip_stops['stop_id'].tolist()
                trip_sequences.append(stop_sequence)
            
            progress_bar.progress((i + 1) / len(trip_ids))
        
        progress_bar.empty()
        return trip_sequences
    
    def estimate_transition_matrix(self, sequences: List[List[str]]) -> np.ndarray:
        """Estima la matriz de transición entre estaciones"""
        # Obtener todas las estaciones únicas
        all_stops = set()
        for seq in sequences:
            all_stops.update(seq)
        
        self.states = sorted(list(all_stops))
        state_to_idx = {state: i for i, state in enumerate(self.states)}
        n_states = len(self.states)
        
        # Matriz de conteos
        transition_counts = np.zeros((n_states, n_states))
        
        # Contar transiciones
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                current_state = sequence[i]
                next_state = sequence[i + 1]
                
                current_idx = state_to_idx[current_state]
                next_idx = state_to_idx[next_state]
                
                transition_counts[current_idx, next_idx] += 1
        
        # Normalizar para obtener probabilidades
        self.transition_matrix = np.zeros_like(transition_counts)
        for i in range(n_states):
            row_sum = transition_counts[i].sum()
            if row_sum > 0:
                self.transition_matrix[i] = transition_counts[i] / row_sum
        
        return self.transition_matrix
    
    def get_stop_name(self, stop_id: str) -> str:
        """Obtiene el nombre de una estación por su ID"""
        if self.stops_df is not None:
            stop_info = self.stops_df[self.stops_df['stop_id'] == stop_id]
            if not stop_info.empty:
                return stop_info.iloc[0]['stop_name']
        return stop_id
    
    def simulate_route(self, start_state: str, steps: int = 10) -> List[str]:
        """Simula una ruta usando la cadena de Markov"""
        if self.states is None or self.transition_matrix is None:
            return []
        
        if start_state not in self.states:
            return []
        
        route = [start_state]
        current_state = start_state
        
        for _ in range(steps):
            current_idx = self.states.index(current_state)
            probs = self.transition_matrix[current_idx]
            
            if probs.sum() == 0:
                break
            
            next_idx = np.random.choice(len(self.states), p=probs)
            next_state = self.states[next_idx]
            
            route.append(next_state)
            current_state = next_state
        
        return route

def create_transition_heatmap(analyzer):
    """Crea un heatmap de la matriz de transición"""
    if analyzer.transition_matrix is None or analyzer.states is None:
        st.error("❌ Matriz de transición no disponible")
        return None
    
    # Crear nombres para las estaciones
    if len(analyzer.states) <= 15:
        station_names = [analyzer.get_stop_name(state) for state in analyzer.states]
        # Truncar nombres largos
        station_names = [name[:20] + "..." if len(name) > 20 else name for name in station_names]
    else:
        station_names = [f"E{i+1}" for i in range(len(analyzer.states))]
    
    # Crear heatmap con Plotly
    fig = go.Figure(data=go.Heatmap(
        z=analyzer.transition_matrix,
        x=station_names,
        y=station_names,
        colorscale='YlOrRd',
        hoverongaps=False,
        hovertemplate='Origen: %{y}<br>Destino: %{x}<br>Probabilidad: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Matriz de Transición - Metrobús CDMX',
        xaxis_title='Estación Destino',
        yaxis_title='Estación Origen',
        height=600,
        width=800
    )
    
    return fig

def create_network_graph(analyzer):
    """Crea un grafo de red de las transiciones"""
    if analyzer.transition_matrix is None or analyzer.states is None:
        st.error("❌ Datos no disponibles para el grafo")
        return None
    
    # Crear grafo dirigido
    G = nx.DiGraph()
    
    # Agregar nodos
    for state in analyzer.states:
        station_name = analyzer.get_stop_name(state)
        G.add_node(state, name=station_name)
    
    # Agregar aristas con pesos (solo las significativas)
    threshold = 0.1  # Solo mostrar transiciones > 10%
    for i, state_from in enumerate(analyzer.states):
        for j, state_to in enumerate(analyzer.states):
            prob = analyzer.transition_matrix[i, j]
            if prob > threshold:
                G.add_edge(state_from, state_to, weight=prob)
    
    # Calcular posiciones usando algoritmo de layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Extraer coordenadas para plotly
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(analyzer.get_stop_name(node))
    
    # Crear el gráfico
    fig = go.Figure()
    
    # Agregar aristas
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                            line=dict(width=0.5, color='#888'),
                            hoverinfo='none',
                            mode='lines'))
    
    # Agregar nodos
    fig.add_trace(go.Scatter(x=node_x, y=node_y,
                            mode='markers+text',
                            hoverinfo='text',
                            text=node_text,
                            textposition="middle center",
                            hovertext=node_text,
                            marker=dict(size=20,
                                       color='lightblue',                                       line=dict(width=2, color='black'))))
    
    fig.update_layout(title='Red de Conexiones - Metrobús CDMX',
                     title_font_size=16,
                     showlegend=False,
                     hovermode='closest',
                     margin=dict(b=20,l=5,r=5,t=40),
                     annotations=[ dict(
                         text="Conexiones con probabilidad > 10%",
                         showarrow=False,
                         xref="paper", yref="paper",
                         x=0.005, y=-0.002,
                         xanchor='left', yanchor='bottom',
                         font=dict(color="gray", size=12)
                     )],
                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    
    return fig

def main():
    """Función principal de la aplicación Streamlit"""
    
    st.title("🚌 Metrobús CDMX - Análisis de Cadenas de Markov")
    st.markdown("""
    ### Análisis de patrones de transición entre estaciones
    Esta aplicación analiza los datos del Metrobús CDMX usando cadenas de Markov para identificar 
    patrones de movimiento entre estaciones y predecir rutas probables.
    """)
    
    # Sidebar para configuración
    st.sidebar.header("⚙️ Configuración")
    
    # Inicializar analizador
    analyzer = MetrobusMarkovAnalyzer()
    
    # Cargar datos
    st.header("📂 Carga de Datos")
    
    if st.button("🔄 Cargar Datos GTFS", help="Carga los archivos de datos del Metrobús"):
        with st.spinner("Cargando datos..."):
            if analyzer.load_data():
                st.session_state['data_loaded'] = True
                st.session_state['analyzer'] = analyzer
            else:
                st.session_state['data_loaded'] = False
    
    # Verificar si los datos están cargados
    if not st.session_state.get('data_loaded', False):
        st.warning("⚠️ Por favor, carga los datos primero haciendo clic en 'Cargar Datos GTFS'")
        st.stop()
    
    # Recuperar analizador de session_state
    analyzer = st.session_state['analyzer']
    
    # Procesar datos y calcular matriz
    st.header("🧮 Análisis de Markov")
    
    if st.button("▶️ Ejecutar Análisis", help="Procesa los datos y calcula la matriz de transición"):
        with st.spinner("Procesando secuencias..."):
            sequences = analyzer.preprocess_data()
            
        if sequences:
            with st.spinner("Calculando matriz de transición..."):
                analyzer.estimate_transition_matrix(sequences)
                st.session_state['analysis_done'] = True
                st.success(f"✅ Análisis completado: {len(analyzer.states)} estaciones analizadas")
        else:
            st.error("❌ No se pudieron procesar las secuencias")
    
    # Mostrar resultados si el análisis está hecho
    if st.session_state.get('analysis_done', False):
        
        # Estadísticas generales
        st.header("📊 Estadísticas Generales")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🚉 Total Estaciones", len(analyzer.states))
        
        with col2:
            st.metric("🔗 Transiciones Posibles", f"{len(analyzer.states)**2:,}")        
        with col3:
            if analyzer.transition_matrix is not None:
                non_zero = np.count_nonzero(analyzer.transition_matrix)
                st.metric("📈 Conexiones Observadas", f"{non_zero:,}")
            else:
                non_zero = 0
        
        with col4:
            if analyzer.transition_matrix is not None and len(analyzer.states) > 0:
                sparsity = (1 - non_zero / len(analyzer.states)**2) * 100
                st.metric("📊 Dispersión", f"{sparsity:.1f}%")
        
        # Tabs para diferentes análisis
        tab1, tab2, tab3, tab4 = st.tabs(["🔥 Heatmap", "🕸️ Red de Conexiones", "🎯 Simulación", "📈 Rankings"])
        
        with tab1:
            st.subheader("Matriz de Transición")
            heatmap_fig = create_transition_heatmap(analyzer)
            if heatmap_fig:
                st.plotly_chart(heatmap_fig, use_container_width=True)
            
            with st.expander("ℹ️ Información del Heatmap"):
                st.markdown("""
                - **Colores más intensos**: Mayor probabilidad de transición
                - **Eje Y**: Estación de origen
                - **Eje X**: Estación de destino
                - **Valores**: Probabilidad de ir de una estación a otra
                """)
        
        with tab2:
            st.subheader("Red de Conexiones")
            network_fig = create_network_graph(analyzer)
            if network_fig:
                st.plotly_chart(network_fig, use_container_width=True)
            
            with st.expander("ℹ️ Información de la Red"):
                st.markdown("""
                - **Nodos**: Estaciones del Metrobús
                - **Conexiones**: Transiciones con probabilidad > 10%
                - **Posición**: Determinada por algoritmo de fuerza dirigida
                """)
        
        with tab3:
            st.subheader("Simulación de Rutas")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if analyzer.states:
                    start_station = st.selectbox(
                        "🚉 Estación de inicio:",
                        options=analyzer.states,
                        format_func=lambda x: analyzer.get_stop_name(x)
                    )
                    
                    num_steps = st.slider("📏 Número de pasos:", 1, 20, 8)
                    
                    if st.button("🎲 Simular Ruta"):
                        route = analyzer.simulate_route(start_station, num_steps)
                        
                        if route:
                            st.subheader("🗺️ Ruta Simulada:")
                            for i, stop in enumerate(route):
                                name = analyzer.get_stop_name(stop)
                                if i == 0:
                                    st.write(f"🏁 **Inicio**: {name}")
                                elif i == len(route) - 1:
                                    st.write(f"🏁 **Final**: {name}")
                                else:
                                    st.write(f"   {i}. {name}")
        
        with tab4:
            st.subheader("Rankings de Estaciones")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔝 Estaciones con más salidas")
                if analyzer.transition_matrix is not None:
                    outgoing_connections = analyzer.transition_matrix.sum(axis=1)
                    top_outgoing = np.argsort(outgoing_connections)[-10:][::-1]
                    
                    for i, idx in enumerate(top_outgoing, 1):
                        station_name = analyzer.get_stop_name(analyzer.states[idx])
                        connections = outgoing_connections[idx]
                        st.write(f"{i}. **{station_name}**: {connections:.3f}")
            
            with col2:
                st.markdown("#### 🎯 Estaciones con más llegadas")
                if analyzer.transition_matrix is not None:
                    incoming_connections = analyzer.transition_matrix.sum(axis=0)
                    top_incoming = np.argsort(incoming_connections)[-10:][::-1]
                    
                    for i, idx in enumerate(top_incoming, 1):
                        station_name = analyzer.get_stop_name(analyzer.states[idx])
                        connections = incoming_connections[idx]
                        st.write(f"{i}. **{station_name}**: {connections:.3f}")
    
    # Footer
    st.markdown("---")
    st.markdown("**💡 Tip**: Los datos muestran patrones de transición basados en horarios históricos del Metrobús CDMX")

if __name__ == "__main__":
    main()
