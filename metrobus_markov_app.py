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
    
    def load_data(self):
        """Carga los datos GTFS del Metrobús"""
        try:
            # Usar rutas relativas desde el directorio actual
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
                st.error(f"📁 Directorio actual: {Path.cwd()}")
                st.error(f"🔍 Buscando en: {self.data_path.absolute()}")
                return False
            
            self.stops_df = pd.read_csv(stops_file)
            self.routes_df = pd.read_csv(routes_file)
            self.stop_times_df = pd.read_csv(stop_times_file)
            self.trips_df = pd.read_csv(trips_file)
            
            st.success(f"✅ Datos cargados: {len(self.stops_df)} estaciones, {len(self.routes_df)} rutas")
            return True
        except Exception as e:
            st.error(f"Error cargando datos: {e}")
            st.error(f"Directorio actual: {Path.cwd()}")
            st.error(f"Buscando en: {self.data_path}")            # Asegurar que las variables queden en None si hay error
            self.stops_df = None
            self.routes_df = None
            self.stop_times_df = None
            self.trips_df = None
            return False
    
    def is_data_loaded(self):
        """Verifica si los datos han sido cargados correctamente"""
        return (self.stops_df is not None and 
                self.routes_df is not None and 
                self.stop_times_df is not None and 
                self.trips_df is not None)
    
    def preprocess_data(self):
        """Preprocesa los datos para análisis de Markov"""
        if not self.is_data_loaded():
            raise ValueError("Los datos no han sido cargados. Llama a load_data() primero.")
        
        # Unir datos para obtener secuencias de paradas por viaje
        trip_sequences = []
        
        # Agrupar por trip_id y ordenar por stop_sequence
        for trip_id in self.stop_times_df['trip_id'].unique()[:100]:  # Limitamos para demo
            trip_stops = self.stop_times_df[
                self.stop_times_df['trip_id'] == trip_id
            ].sort_values('stop_sequence')
            
            if len(trip_stops) > 1:
                stop_sequence = trip_stops['stop_id'].tolist()
                trip_sequences.append(stop_sequence)
        
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
        for seq in sequences:
            for i in range(len(seq) - 1):
                from_idx = state_to_idx[seq[i]]
                to_idx = state_to_idx[seq[i + 1]]
                transition_counts[from_idx, to_idx] += 1
        
        # Normalizar para obtener probabilidades
        row_sums = transition_counts.sum(axis=1)
        # Evitar división por cero
        row_sums[row_sums == 0] = 1
        self.transition_matrix = transition_counts / row_sums[:, np.newaxis]
        
        return self.transition_matrix
    
    def get_stop_name(self, stop_id: str) -> str:
        """Obtiene el nombre de una estación por su ID"""
        if self.stops_df is not None:
            stop_info = self.stops_df[self.stops_df['stop_id'] == stop_id]
            if not stop_info.empty:
                return stop_info.iloc[0]['stop_name']
        return stop_id
    
    def simulate_markov_chain(self, start_state: str, n_steps: int = 10) -> List[str]:
        """Simula una cadena de Markov desde un estado inicial"""
        if self.transition_matrix is None or self.states is None:
            return []
        
        try:
            current_idx = self.states.index(start_state)
        except ValueError:
            return []
        
        simulation = [start_state]
        
        for _ in range(n_steps):
            # Obtener probabilidades de transición desde el estado actual
            probs = self.transition_matrix[current_idx]
            
            # Si no hay transiciones posibles, terminar
            if probs.sum() == 0:
                break
            
            # Seleccionar siguiente estado basado en probabilidades
            next_idx = np.random.choice(len(self.states), p=probs)
            next_state = self.states[next_idx]
            simulation.append(next_state)
            current_idx = next_idx
        
        return simulation

def create_transition_heatmap(analyzer, top_n=20):
    """Crea un mapa de calor de la matriz de transición"""
    if analyzer.transition_matrix is None:
        return None
    
    # Seleccionar las top_n estaciones más activas
    activity = analyzer.transition_matrix.sum(axis=1) + analyzer.transition_matrix.sum(axis=0)
    top_indices = np.argsort(activity)[-top_n:]
    
    # Submatriz
    sub_matrix = analyzer.transition_matrix[np.ix_(top_indices, top_indices)]
    top_states = [analyzer.states[i] for i in top_indices]
    top_names = [analyzer.get_stop_name(state) for state in top_states]
    
    fig = px.imshow(
        sub_matrix,
        x=top_names,
        y=top_names,
        color_continuous_scale="Viridis",
        title=f"Matriz de Transición - Top {top_n} Estaciones Más Activas",
        labels=dict(x="Estación Destino", y="Estación Origen", color="Probabilidad")
    )
    
    fig.update_layout(
        height=600,
        xaxis_tickangle=-45,
        font_size=10
    )
    
    return fig

def create_network_graph(analyzer, threshold=0.1):
    """Crea un grafo de red de las transiciones"""
    if analyzer.transition_matrix is None:
        return None
    
    # Crear grafo dirigido
    G = nx.DiGraph()
    
    # Agregar nodos y aristas
    for i, from_state in enumerate(analyzer.states):
        for j, to_state in enumerate(analyzer.states):
            prob = analyzer.transition_matrix[i, j]
            if prob > threshold:  # Solo incluir transiciones significativas
                from_name = analyzer.get_stop_name(from_state)
                to_name = analyzer.get_stop_name(to_state)
                G.add_edge(from_name, to_name, weight=prob)
    
    # Calcular posiciones usando layout spring
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Extraer coordenadas
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Crear trazas
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        marker=dict(
            size=10,
            color='lightblue',
            line=dict(width=2, color='black')
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=f'Red de Transiciones (umbral > {threshold})',
                       title_font_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[ dict(
                           text="Conexiones entre estaciones del Metrobús",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002,
                           xanchor="left", yanchor="bottom",
                           font=dict(color="black", size=12)
                       )],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    return fig

def main():
    """Función principal de la aplicación"""
    st.title("🚌 Análisis de Cadenas de Markov - Metrobús CDMX")
    st.markdown("### Análisis de patrones de transición entre estaciones")
      # Inicializar analizador
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = MetrobusMarkovAnalyzer()
    
    analyzer = st.session_state.analyzer
      # Sidebar para configuración
    st.sidebar.header("⚙️ Configuración")
      # Cargar datos
    if st.sidebar.button("🔄 Cargar Datos"):
        with st.spinner("Cargando datos del Metrobús..."):
            if analyzer.load_data():
                st.sidebar.success("✅ Datos cargados exitosamente")
                st.session_state.data_loaded = True
            else:
                st.sidebar.error("❌ Error al cargar datos")
                st.session_state.data_loaded = False
    
    # No necesitamos recuperar analyzer de session_state ya que lo tenemos
      # Verificar si los datos están cargados
    if not hasattr(st.session_state, 'data_loaded') or not st.session_state.data_loaded:
        st.info("👆 Haz clic en 'Cargar Datos' en la barra lateral para comenzar")
        return
      # Verificar que el analyzer tenga los datos cargados
    if not analyzer.is_data_loaded():
        st.error("❌ Error: Los datos no están disponibles en el analizador. Intenta cargar de nuevo.")
        st.session_state.data_loaded = False
        return
    
    # Procesar datos y estimar matriz
    if st.sidebar.button("🧮 Calcular Matriz de Transición"):
        with st.spinner("Procesando secuencias de viajes..."):
            try:
                sequences = analyzer.preprocess_data()
                st.sidebar.info(f"📊 Se procesaron {len(sequences)} secuencias de viajes")
            except ValueError as e:
                st.error(f"❌ Error al procesar datos: {e}")
                return
            
        with st.spinner("Estimando matriz de transición..."):
            analyzer.estimate_transition_matrix(sequences)
            st.sidebar.success("✅ Matriz calculada")
            st.session_state.matrix_calculated = True
      # Verificar si la matriz está calculada
    if not hasattr(st.session_state, 'matrix_calculated') or not st.session_state.matrix_calculated:
        st.info("👆 Haz clic en 'Calcular Matriz de Transición' para continuar")
        return
    
    # Verificar que los datos estén disponibles
    if analyzer.states is None or analyzer.transition_matrix is None:
        st.error("❌ Error: Los datos no están disponibles. Intenta cargar y calcular de nuevo.")
        return
    
    # Mostrar estadísticas básicas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🚉 Total Estaciones", len(analyzer.states))
    
    with col2:
        st.metric("🔗 Transiciones Posibles", f"{len(analyzer.states)**2:,}")
    
    with col3:
        non_zero = np.count_nonzero(analyzer.transition_matrix)
        st.metric("✅ Transiciones Activas", f"{non_zero:,}")
    
    with col4:
        sparsity = (1 - non_zero / len(analyzer.states)**2) * 100
        st.metric("📉 Dispersión", f"{sparsity:.1f}%")
    
    # Tabs para diferentes análisis
    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Matriz de Calor", "🕸️ Red de Conexiones", "🎯 Simulación", "📊 Estadísticas"])
    
    with tab1:
        st.subheader("Matriz de Transición")
        
        top_n = st.slider("Número de estaciones a mostrar", 10, 50, 20)
        fig_heatmap = create_transition_heatmap(analyzer, top_n)
        
        if fig_heatmap:
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.info("💡 **Interpretación**: Los colores más brillantes indican mayor probabilidad de transición entre estaciones.")
    
    with tab2:
        st.subheader("Red de Conexiones")
        
        threshold = st.slider("Umbral mínimo de probabilidad", 0.01, 0.5, 0.1, 0.01)
        fig_network = create_network_graph(analyzer, threshold)
        
        if fig_network:
            st.plotly_chart(fig_network, use_container_width=True)
        
        st.info("💡 **Interpretación**: Los nodos representan estaciones y las conexiones muestran transiciones frecuentes.")
    
    with tab3:
        st.subheader("Simulación de Rutas")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Seleccionar estación inicial
            station_names = [analyzer.get_stop_name(state) for state in analyzer.states]
            selected_station_name = st.selectbox("Estación inicial", station_names)
            
            # Encontrar el stop_id correspondiente
            selected_station_id = None
            for state in analyzer.states:
                if analyzer.get_stop_name(state) == selected_station_name:
                    selected_station_id = state
                    break
        
        with col2:
            n_steps = st.slider("Número de pasos", 5, 20, 10)
        
        if st.button("🚀 Simular Ruta"):
            if selected_station_id:
                simulation = analyzer.simulate_markov_chain(selected_station_id, n_steps)
                
                if simulation:
                    st.subheader("🛤️ Ruta Simulada:")
                    
                    # Crear DataFrame para mostrar la ruta
                    route_data = []
                    for i, stop_id in enumerate(simulation):
                        route_data.append({
                            'Paso': i + 1,
                            'Estación': analyzer.get_stop_name(stop_id),
                            'ID': stop_id
                        })
                    
                    route_df = pd.DataFrame(route_data)
                    st.dataframe(route_df, use_container_width=True)
                    
                    # Visualizar como gráfico de línea
                    fig_route = px.line(
                        route_df, 
                        x='Paso', 
                        y='Estación',
                        title="Trayectoria Simulada",
                        markers=True
                    )
                    fig_route.update_layout(height=400)
                    st.plotly_chart(fig_route, use_container_width=True)
                else:
                    st.error("No se pudo simular la ruta desde esta estación")
    
    with tab4:
        st.subheader("Estadísticas de la Matriz de Transición")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Estaciones más conectadas (outgoing)
            outgoing_connections = analyzer.transition_matrix.sum(axis=1)
            top_outgoing_idx = np.argsort(outgoing_connections)[-10:]
            
            outgoing_data = []
            for idx in reversed(top_outgoing_idx):
                station_name = analyzer.get_stop_name(analyzer.states[idx])
                outgoing_data.append({
                    'Estación': station_name,
                    'Conexiones Salientes': outgoing_connections[idx]
                })
            
            st.subheader("🔝 Top 10 - Más Conexiones Salientes")
            st.dataframe(pd.DataFrame(outgoing_data), use_container_width=True)
        
        with col2:
            # Estaciones más conectadas (incoming)
            incoming_connections = analyzer.transition_matrix.sum(axis=0)
            top_incoming_idx = np.argsort(incoming_connections)[-10:]
            
            incoming_data = []
            for idx in reversed(top_incoming_idx):
                station_name = analyzer.get_stop_name(analyzer.states[idx])
                incoming_data.append({
                    'Estación': station_name,
                    'Conexiones Entrantes': incoming_connections[idx]
                })
            
            st.subheader("🎯 Top 10 - Más Conexiones Entrantes")
            st.dataframe(pd.DataFrame(incoming_data), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            💻 Desarrollado para análisis de transporte público | 
            🚌 Datos del Metrobús CDMX | 
            📊 Cadenas de Markov
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
