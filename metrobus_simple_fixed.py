"""
Analizador simple de cadenas de Markov para Metrobús CDMX
Versión de línea de comandos con análisis básico y visualizaciones
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class MetrobusMarkovSimple:
    """Analizador simplificado para cadenas de Markov del Metrobús"""
    
    def __init__(self, data_folder: str = "data"):
        self.data_folder = Path(data_folder)
        self.stops_df = None
        self.routes_df = None
        self.stop_times_df = None
        self.transition_matrix = None
        self.states = None
        self.state_names = {}
        
        # Configurar matplotlib para español
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def cargar_datos(self) -> bool:
        """Carga los datos GTFS del Metrobús"""
        try:
            print("📂 Cargando datos del Metrobús...")
            
            # Usar rutas relativas desde el directorio actual
            stops_file = self.data_folder / "stops.txt"
            routes_file = self.data_folder / "routes.txt"
            stop_times_file = self.data_folder / "stop_times.txt"
            
            # Verificar archivos
            for file_path, name in [(stops_file, "stops.txt"), (routes_file, "routes.txt"), 
                                   (stop_times_file, "stop_times.txt")]:
                if not file_path.exists():
                    print(f"❌ Archivo no encontrado: {file_path}")
                    print(f"📁 Directorio actual: {Path.cwd()}")
                    print(f"🔍 Buscando en: {self.data_folder.absolute()}")
                    return False
            
            self.stops_df = pd.read_csv(stops_file)
            self.routes_df = pd.read_csv(routes_file)
            self.stop_times_df = pd.read_csv(stop_times_file)
            
            print(f"✅ Datos cargados exitosamente:")
            print(f"   🚉 Estaciones: {len(self.stops_df)}")
            print(f"   🚌 Rutas: {len(self.routes_df)}")
            print(f"   ⏰ Horarios: {len(self.stop_times_df)}")
            
            # Crear mapeo de IDs a nombres
            self.state_names = dict(zip(self.stops_df['stop_id'], self.stops_df['stop_name']))
            
            return True
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            print(f"📁 Directorio actual: {Path.cwd()}")
            print(f"🔍 Intentando cargar desde: {self.data_folder}")
            return False
    
    def procesar_secuencias(self, max_trips: int = 200) -> List[List[str]]:
        """Procesa las secuencias de viajes para análisis Markov"""
        print(f"🔄 Procesando secuencias de viajes (máximo {max_trips})...")
        
        secuencias = []
        trip_ids = self.stop_times_df['trip_id'].unique()[:max_trips]
        
        for trip_id in trip_ids:
            # Obtener paradas del viaje ordenadas por secuencia
            trip_stops = self.stop_times_df[
                self.stop_times_df['trip_id'] == trip_id
            ].sort_values('stop_sequence')
            
            if len(trip_stops) > 1:
                secuencia = trip_stops['stop_id'].tolist()
                secuencias.append(secuencia)
        
        print(f"✅ Procesadas {len(secuencias)} secuencias válidas")
        return secuencias
    
    def estimar_matriz_transicion(self, secuencias: List[List[str]]) -> np.ndarray:
        """Estima la matriz de transición entre estaciones"""
        print("🧮 Calculando matriz de transición...")
        
        # Obtener estados únicos
        estados_unicos = set()
        for seq in secuencias:
            estados_unicos.update(seq)
        
        self.states = sorted(list(estados_unicos))
        estado_a_idx = {estado: i for i, estado in enumerate(self.states)}
        n_estados = len(self.states)
        
        # Matriz de conteos
        conteos = np.zeros((n_estados, n_estados))
        
        # Contar transiciones
        for secuencia in secuencias:
            for i in range(len(secuencia) - 1):
                estado_actual = secuencia[i]
                estado_siguiente = secuencia[i + 1]
                
                idx_actual = estado_a_idx[estado_actual]
                idx_siguiente = estado_a_idx[estado_siguiente]
                
                conteos[idx_actual, idx_siguiente] += 1
        
        # Normalizar para obtener probabilidades
        self.transition_matrix = np.zeros_like(conteos)
        for i in range(n_estados):
            total_salidas = conteos[i].sum()
            if total_salidas > 0:
                self.transition_matrix[i] = conteos[i] / total_salidas
        
        print(f"✅ Matriz de transición calculada: {n_estados}x{n_estados}")
        print(f"   📊 Transiciones observadas: {conteos.sum():.0f}")
        print(f"   🔗 Conexiones únicas: {np.count_nonzero(conteos)}")
        
        return self.transition_matrix
    
    def generar_heatmap(self, guardar_archivo: str = "matriz_transicion_metrobus.png"):
        """Genera un heatmap de la matriz de transición"""
        print("🎨 Generando visualización heatmap...")
        
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # Usar nombres de estaciones si hay pocos estados
        if len(self.states) <= 20:
            labels = [self.state_names.get(state, state)[:15] + "..." 
                     if len(self.state_names.get(state, state)) > 15 
                     else self.state_names.get(state, state) for state in self.states]
        else:
            labels = [f"E{i+1}" for i in range(len(self.states))]
        
        # Crear heatmap
        sns.heatmap(self.transition_matrix, 
                   xticklabels=labels,
                   yticklabels=labels,
                   cmap='YlOrRd',
                   annot=len(self.states) <= 10,
                   fmt='.3f',
                   cbar_kws={'label': 'Probabilidad de Transición'},
                   ax=ax)
        
        ax.set_title('Matriz de Transición - Metrobús CDMX', fontsize=16, fontweight='bold')
        ax.set_xlabel('Estación Destino', fontsize=12)
        ax.set_ylabel('Estación Origen', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(guardar_archivo, dpi=300, bbox_inches='tight')
        print(f"✅ Heatmap guardado: {guardar_archivo}")
        
        plt.show()
        return fig
    
    def analizar_estadisticas(self):
        """Analiza estadísticas de la matriz de transición"""
        print("\n📊 ANÁLISIS ESTADÍSTICO")
        print("-" * 40)
        
        n_estados = len(self.states)
        transiciones_observadas = np.count_nonzero(self.transition_matrix)
        transiciones_posibles = n_estados * n_estados
        densidad = (transiciones_observadas / transiciones_posibles) * 100
        
        print(f"🚉 Total de estaciones: {n_estados}")
        print(f"🔗 Transiciones posibles: {transiciones_posibles:,}")
        print(f"📈 Transiciones observadas: {transiciones_observadas}")
        print(f"📊 Densidad de la matriz: {densidad:.1f}%")
        
        # Estaciones más conectadas
        conexiones_salida = self.transition_matrix.sum(axis=1)
        conexiones_entrada = self.transition_matrix.sum(axis=0)
        
        print(f"\n🏆 TOP 5 ESTACIONES MÁS ACTIVAS:")
        for i, idx in enumerate(np.argsort(conexiones_salida)[-5:][::-1]):
            nombre = self.state_names.get(self.states[idx], self.states[idx])
            print(f"   {i+1}. {nombre}: {conexiones_salida[idx]:.3f}")
    
    def simular_ruta(self, estado_inicial: str, pasos: int = 10) -> List[str]:
        """Simula una ruta usando la cadena de Markov"""
        if estado_inicial not in self.states:
            print(f"❌ Estado '{estado_inicial}' no encontrado")
            return []
        
        ruta = [estado_inicial]
        estado_actual = estado_inicial
        
        for _ in range(pasos):
            idx_actual = self.states.index(estado_actual)
            probabilidades = self.transition_matrix[idx_actual]
            
            if probabilidades.sum() == 0:
                break
            
            # Elegir siguiente estado
            idx_siguiente = np.random.choice(len(self.states), p=probabilidades)
            estado_siguiente = self.states[idx_siguiente]
            
            ruta.append(estado_siguiente)
            estado_actual = estado_siguiente
        
        return ruta
    
    def ejecutar_analisis_completo(self):
        """Ejecuta el análisis completo"""
        print("🚌 ANALIZADOR METROBÚS CDMX - CADENAS DE MARKOV")
        print("=" * 55)
        
        # 1. Cargar datos
        if not self.cargar_datos():
            print("❌ Error en carga de datos. Terminando...")
            return False
        
        # 2. Procesar secuencias
        secuencias = self.procesar_secuencias(max_trips=300)
        if not secuencias:
            print("❌ No se pudieron procesar secuencias")
            return False
        
        # 3. Calcular matriz de transición
        self.estimar_matriz_transicion(secuencias)
        
        # 4. Análisis estadístico
        self.analizar_estadisticas()
        
        # 5. Generar visualización
        self.generar_heatmap()
        
        # 6. Ejemplo de simulación
        if self.states:
            print(f"\n🎯 SIMULACIÓN DE RUTA:")
            estado_ejemplo = self.states[0]
            ruta_simulada = self.simular_ruta(estado_ejemplo, pasos=5)
            
            print(f"   Partiendo de: {self.state_names.get(estado_ejemplo, estado_ejemplo)}")
            for i, estado in enumerate(ruta_simulada[1:], 1):
                nombre = self.state_names.get(estado, estado)
                print(f"   Paso {i}: {nombre}")
        
        print(f"\n✅ Análisis completado exitosamente!")
        return True

def main():
    """Función principal"""
    analyzer = MetrobusMarkovSimple()
    analyzer.ejecutar_analisis_completo()

if __name__ == "__main__":
    main()
