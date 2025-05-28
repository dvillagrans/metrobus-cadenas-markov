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
            
            # Usar rutas absolutas
            base_path = Path(__file__).parent / self.data_folder
            
            stops_file = base_path / "stops.txt"
            routes_file = base_path / "routes.txt" 
            stop_times_file = base_path / "stop_times.txt"
            
            # Verificar archivos
            for file_path, name in [(stops_file, "stops.txt"), (routes_file, "routes.txt"), 
                                   (stop_times_file, "stop_times.txt")]:
                if not file_path.exists():
                    print(f"❌ Archivo no encontrado: {file_path}")
                    print(f"📁 Directorio actual: {Path.cwd()}")
                    print(f"🔍 Buscando en: {base_path}")
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
        
        print(f"✅ Se procesaron {len(secuencias)} secuencias válidas")
        return secuencias
    
    def calcular_matriz_transicion(self, secuencias: List[List[str]]) -> np.ndarray:
        """Calcula la matriz de transición"""
        print("🧮 Calculando matriz de transición...")
        
        # Obtener todos los estados únicos
        todos_estados = set()
        for seq in secuencias:
            todos_estados.update(seq)
        
        self.states = sorted(list(todos_estados))
        n_estados = len(self.states)
        
        print(f"📊 Estados únicos encontrados: {n_estados}")
        
        # Crear mapeo estado -> índice
        estado_a_idx = {estado: i for i, estado in enumerate(self.states)}
        
        # Matriz de conteos
        conteos = np.zeros((n_estados, n_estados))
        
        # Contar transiciones
        total_transiciones = 0
        for seq in secuencias:
            for i in range(len(seq) - 1):
                from_idx = estado_a_idx[seq[i]]
                to_idx = estado_a_idx[seq[i + 1]]
                conteos[from_idx, to_idx] += 1
                total_transiciones += 1
        
        # Normalizar para obtener probabilidades
        sumas_filas = conteos.sum(axis=1)
        sumas_filas[sumas_filas == 0] = 1  # Evitar división por cero
        
        self.transition_matrix = conteos / sumas_filas[:, np.newaxis]
        
        print(f"✅ Matriz calculada: {n_estados}x{n_estados}")
        print(f"🔗 Total de transiciones: {total_transiciones}")
        print(f"📈 Densidad de matriz: {(np.count_nonzero(conteos) / (n_estados**2)) * 100:.1f}%")
        
        return self.transition_matrix
    
    def obtener_estadisticas(self) -> Dict:
        """Calcula estadísticas básicas de la matriz"""
        if self.transition_matrix is None:
            return {}
        
        # Conexiones salientes y entrantes
        conexiones_salida = self.transition_matrix.sum(axis=1)
        conexiones_entrada = self.transition_matrix.sum(axis=0)
        
        # Top estaciones
        top_salida_idx = np.argsort(conexiones_salida)[-10:]
        top_entrada_idx = np.argsort(conexiones_entrada)[-10:]
        
        estadisticas = {
            'total_estados': len(self.states),
            'transiciones_activas': np.count_nonzero(self.transition_matrix),
            'densidad': (np.count_nonzero(self.transition_matrix) / len(self.states)**2) * 100,
            'top_salida': [(self.state_names.get(self.states[i], self.states[i]), 
                           conexiones_salida[i]) for i in reversed(top_salida_idx)],
            'top_entrada': [(self.state_names.get(self.states[i], self.states[i]), 
                            conexiones_entrada[i]) for i in reversed(top_entrada_idx)]
        }
        
        return estadisticas
    
    def mostrar_estadisticas(self):
        """Muestra estadísticas en la consola"""
        stats = self.obtener_estadisticas()
        
        print("\n" + "="*60)
        print("📊 ESTADÍSTICAS DE LA MATRIZ DE TRANSICIÓN")
        print("="*60)
        
        print(f"🚉 Total de estaciones: {stats['total_estados']}")
        print(f"🔗 Transiciones activas: {stats['transiciones_activas']:,}")
        print(f"📈 Densidad de matriz: {stats['densidad']:.1f}%")
        
        print(f"\n🔝 TOP 5 ESTACIONES CON MÁS CONEXIONES SALIENTES:")
        for i, (nombre, valor) in enumerate(stats['top_salida'][:5], 1):
            print(f"   {i}. {nombre}: {valor:.3f}")
        
        print(f"\n🎯 TOP 5 ESTACIONES CON MÁS CONEXIONES ENTRANTES:")
        for i, (nombre, valor) in enumerate(stats['top_entrada'][:5], 1):
            print(f"   {i}. {nombre}: {valor:.3f}")
    
    def visualizar_matriz_calor(self, top_n: int = 15, guardar: bool = True):
        """Crea un mapa de calor de la matriz de transición"""
        if self.transition_matrix is None:
            print("❌ Primero debes calcular la matriz de transición")
            return
        
        print(f"🎨 Creando mapa de calor (top {top_n} estaciones)...")
        
        # Seleccionar las estaciones más activas
        actividad = self.transition_matrix.sum(axis=1) + self.transition_matrix.sum(axis=0)
        top_indices = np.argsort(actividad)[-top_n:]
        
        # Submatriz y nombres
        sub_matriz = self.transition_matrix[np.ix_(top_indices, top_indices)]
        nombres = [self.state_names.get(self.states[i], self.states[i])[:20] 
                  for i in top_indices]
        
        # Crear visualización
        plt.figure(figsize=(14, 10))
        sns.heatmap(sub_matriz, 
                   xticklabels=nombres,
                   yticklabels=nombres,
                   cmap='YlOrRd',
                   annot=False,
                   fmt='.3f',
                   cbar_kws={'label': 'Probabilidad de Transición'})
        
        plt.title(f'Matriz de Transición - Top {top_n} Estaciones Más Activas\nMetrobús CDMX', 
                 fontsize=16, pad=20)
        plt.xlabel('Estación Destino', fontsize=12)
        plt.ylabel('Estación Origen', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if guardar:
            plt.savefig('matriz_transicion_metrobus.png', dpi=300, bbox_inches='tight')
            print("💾 Mapa de calor guardado como 'matriz_transicion_metrobus.png'")
        
        plt.show()
    
    def simular_ruta(self, estacion_inicio: str, pasos: int = 10) -> List[str]:
        """Simula una ruta usando la cadena de Markov"""
        if self.transition_matrix is None:
            print("❌ Primero debes calcular la matriz de transición")
            return []
        
        # Buscar la estación por nombre
        estacion_id = None
        for stop_id, nombre in self.state_names.items():
            if estacion_inicio.lower() in nombre.lower() and stop_id in self.states:
                estacion_id = stop_id
                break
        
        if estacion_id is None:
            print(f"❌ No se encontró la estación '{estacion_inicio}'")
            print("💡 Estaciones disponibles (primeras 10):")
            for i, (stop_id, nombre) in enumerate(list(self.state_names.items())[:10]):
                if stop_id in self.states:
                    print(f"   - {nombre}")
            return []
        
        # Simular
        try:
            idx_actual = self.states.index(estacion_id)
        except ValueError:
            print(f"❌ La estación no está en el modelo")
            return []
        
        simulacion = [estacion_id]
        
        for paso in range(pasos):
            probabilidades = self.transition_matrix[idx_actual]
            
            if probabilidades.sum() == 0:
                print(f"🛑 No hay transiciones posibles desde {self.state_names[estacion_id]}")
                break
            
            siguiente_idx = np.random.choice(len(self.states), p=probabilidades)
            siguiente_id = self.states[siguiente_idx]
            simulacion.append(siguiente_id)
            idx_actual = siguiente_idx
        
        return simulacion
    
    def mostrar_ruta_simulada(self, estacion_inicio: str, pasos: int = 10):
        """Muestra una ruta simulada de forma bonita"""
        print(f"\n🚀 Simulando ruta desde '{estacion_inicio}' ({pasos} pasos)...")
        
        ruta = self.simular_ruta(estacion_inicio, pasos)
        
        if not ruta:
            return
        
        print(f"\n🛤️  RUTA SIMULADA:")
        print("="*50)
        
        for i, stop_id in enumerate(ruta):
            nombre = self.state_names.get(stop_id, stop_id)
            if i == 0:
                print(f"🚏 INICIO: {nombre}")
            elif i == len(ruta) - 1:
                print(f"🏁 FIN:    {nombre}")
            else:
                print(f"   {i:2d}:    {nombre}")
        
        print(f"\n📍 Total de paradas: {len(ruta)}")
    
    def analisis_completo(self, max_trips: int = 200, top_n: int = 15):
        """Ejecuta un análisis completo del sistema"""
        print("🚌 ANÁLISIS COMPLETO DEL METROBÚS CDMX")
        print("="*60)
        
        # 1. Cargar datos
        if not self.cargar_datos():
            return False
        
        # 2. Procesar secuencias
        secuencias = self.procesar_secuencias(max_trips)
        if not secuencias:
            print("❌ No se pudieron procesar las secuencias")
            return False
        
        # 3. Calcular matriz
        self.calcular_matriz_transicion(secuencias)
        
        # 4. Mostrar estadísticas
        self.mostrar_estadisticas()
        
        # 5. Crear visualización
        self.visualizar_matriz_calor(top_n)
        
        # 6. Ejemplo de simulación
        print(f"\n🎯 EJEMPLO DE SIMULACIÓN:")
        self.mostrar_ruta_simulada("Buenavista", 8)
        
        print(f"\n✅ Análisis completo finalizado!")
        print(f"💡 Puedes usar los métodos individuales para análisis específicos")
        
        return True


def main():
    """Función principal de demostración"""
    # Crear analizador
    analizador = MetrobusMarkovSimple()
    
    # Ejecutar análisis completo
    exito = analizador.analisis_completo(max_trips=300, top_n=12)
    
    if exito:
        print(f"\n🎉 ¡Análisis completado exitosamente!")
        print(f"📊 Revisa la imagen 'matriz_transicion_metrobus.png'")
        print(f"🔄 Puedes hacer más simulaciones con:")
        print(f"   analizador.mostrar_ruta_simulada('NombreEstacion', pasos)")


if __name__ == "__main__":
    main()
