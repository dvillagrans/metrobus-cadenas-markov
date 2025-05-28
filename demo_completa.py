"""
Demostración completa y bonita del análisis de Markov para Metrobús CDMX
Incluye análisis temporal y visualizaciones mejoradas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def demo_completa():
    """Demostración completa con múltiples análisis"""
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    print("🚌" + "="*60 + "🚌")
    print("    ANÁLISIS AVANZADO DE CADENAS DE MARKOV")
    print("         METROBÚS CIUDAD DE MÉXICO")
    print("🚌" + "="*60 + "🚌")
    
    # 1. CARGAR Y VALIDAR DATOS
    print("\n📊 FASE 1: CARGA Y VALIDACIÓN DE DATOS")
    print("-" * 45)
    
    data_path = Path("data")
    
    try:
        stops_df = pd.read_csv(data_path / "stops.txt")
        routes_df = pd.read_csv(data_path / "routes.txt")
        stop_times_df = pd.read_csv(data_path / "stop_times.txt")
        trips_df = pd.read_csv(data_path / "trips.txt")
        
        print(f"✅ Archivo de estaciones: {len(stops_df):,} registros")
        print(f"✅ Archivo de rutas: {len(routes_df):,} registros")
        print(f"✅ Archivo de horarios: {len(stop_times_df):,} registros")
        print(f"✅ Archivo de viajes: {len(trips_df):,} registros")
        
        # Estadísticas básicas
        total_trips = stop_times_df['trip_id'].nunique()
        total_stops = stop_times_df['stop_id'].nunique()
        
        print(f"\n📈 Estadísticas generales:")
        print(f"   🚉 Estaciones únicas en servicio: {total_stops}")
        print(f"   🚌 Viajes únicos registrados: {total_trips:,}")
        print(f"   📅 Cobertura temporal: Horarios completos")
        
    except Exception as e:
        print(f"❌ Error en carga de datos: {e}")
        return
    
    # 2. ANÁLISIS POR LÍNEAS
    print(f"\n🚇 FASE 2: ANÁLISIS POR LÍNEAS DE METROBÚS")
    print("-" * 50)
    
    # Unir datos para obtener información de líneas
    merged_data = stop_times_df.merge(trips_df, on='trip_id').merge(routes_df, on='route_id')
    
    # Análisis por línea
    lineas_stats = merged_data.groupby('route_short_name').agg({
        'trip_id': 'nunique',
        'stop_id': 'nunique'
    }).rename(columns={'trip_id': 'viajes', 'stop_id': 'estaciones'})
    
    print("📊 Top 5 líneas más activas:")
    top_lines = lineas_stats.sort_values('viajes', ascending=False).head()
    for idx, (linea, data) in enumerate(top_lines.iterrows(), 1):
        print(f"   {idx}. Línea {linea}: {data['viajes']:,} viajes, {data['estaciones']} estaciones")
    
    # 3. PROCESAMIENTO DE SECUENCIAS INTELIGENTE
    print(f"\n🔄 FASE 3: PROCESAMIENTO INTELIGENTE DE SECUENCIAS")
    print("-" * 55)
    
    # Seleccionar muestra representativa
    sample_size = 500
    trip_sample = stop_times_df['trip_id'].unique()[:sample_size]
    
    secuencias = []
    secuencias_por_linea = {}
    
    print(f"🧮 Procesando {sample_size} viajes...")
    
    for trip_id in trip_sample:
        trip_data = stop_times_df[stop_times_df['trip_id'] == trip_id].sort_values('stop_sequence')
        
        if len(trip_data) > 2:  # Al menos 3 paradas
            secuencia = trip_data['stop_id'].tolist()
            secuencias.append(secuencia)
            
            # Clasificar por línea si es posible
            try:
                route_info = merged_data[merged_data['trip_id'] == trip_id]['route_short_name'].iloc[0]
                if route_info not in secuencias_por_linea:
                    secuencias_por_linea[route_info] = []
                secuencias_por_linea[route_info].append(secuencia)
            except:
                pass
    
    print(f"✅ {len(secuencias)} secuencias válidas procesadas")
    print(f"📋 {len(secuencias_por_linea)} líneas identificadas")
    
    # 4. CÁLCULO DE MATRIZ DE TRANSICIÓN AVANZADA
    print(f"\n🧮 FASE 4: CÁLCULO DE MATRIZ DE TRANSICIÓN")
    print("-" * 48)
    
    # Obtener estados únicos
    todos_estados = set()
    for seq in secuencias:
        todos_estados.update(seq)
    
    estados = sorted(list(todos_estados))
    n_estados = len(estados)
    estado_a_idx = {estado: i for i, estado in enumerate(estados)}
    
    print(f"🎯 Estados del sistema: {n_estados}")
    
    # Calcular matriz
    conteos = np.zeros((n_estados, n_estados))
    total_transiciones = 0
    
    for seq in secuencias:
        for i in range(len(seq) - 1):
            from_idx = estado_a_idx[seq[i]]
            to_idx = estado_a_idx[seq[i + 1]]
            conteos[from_idx, to_idx] += 1
            total_transiciones += 1
    
    # Normalizar
    sumas = conteos.sum(axis=1)
    sumas[sumas == 0] = 1
    matriz_prob = conteos / sumas[:, np.newaxis]
    
    # Métricas de la matriz
    transiciones_activas = np.count_nonzero(conteos)
    densidad = (transiciones_activas / (n_estados**2)) * 100
    max_prob = matriz_prob.max()
    
    print(f"✅ Matriz {n_estados}×{n_estados} calculada")
    print(f"🔗 Transiciones observadas: {total_transiciones:,}")
    print(f"💫 Transiciones únicas: {transiciones_activas}")
    print(f"📊 Densidad: {densidad:.1f}%")
    print(f"🔥 Probabilidad máxima: {max_prob:.3f}")
    
    # 5. ANÁLISIS TOPOLÓGICO
    print(f"\n🕸️  FASE 5: ANÁLISIS TOPOLÓGICO DEL SISTEMA")
    print("-" * 48)
    
    # Calcular métricas de centralidad
    conexiones_salida = matriz_prob.sum(axis=1)
    conexiones_entrada = matriz_prob.sum(axis=0)
    actividad_total = conexiones_salida + conexiones_entrada
    
    # Mapeo de nombres
    nombres = dict(zip(stops_df['stop_id'], stops_df['stop_name']))
    
    # Top estaciones por diferentes métricas
    print("🏆 TOP 3 ESTACIONES POR ACTIVIDAD TOTAL:")
    top_actividad = np.argsort(actividad_total)[-3:]
    for i, idx in enumerate(reversed(top_actividad), 1):
        estado_id = estados[idx]
        nombre = nombres.get(estado_id, estado_id)
        print(f"   {i}. {nombre[:35]:<35} (actividad: {actividad_total[idx]:.3f})")
    
    print("\n🚀 TOP 3 ESTACIONES COMO ORIGEN:")
    top_salida = np.argsort(conexiones_salida)[-3:]
    for i, idx in enumerate(reversed(top_salida), 1):
        estado_id = estados[idx]
        nombre = nombres.get(estado_id, estado_id)
        print(f"   {i}. {nombre[:35]:<35} (salidas: {conexiones_salida[idx]:.3f})")
    
    print("\n🎯 TOP 3 ESTACIONES COMO DESTINO:")
    top_entrada = np.argsort(conexiones_entrada)[-3:]
    for i, idx in enumerate(reversed(top_entrada), 1):
        estado_id = estados[idx]
        nombre = nombres.get(estado_id, estado_id)
        print(f"   {i}. {nombre[:35]:<35} (entradas: {conexiones_entrada[idx]:.3f})")
    
    # 6. SIMULACIONES MÚLTIPLES
    print(f"\n🎮 FASE 6: SIMULACIONES MÚLTIPLES")
    print("-" * 40)
    
    def simular_ruta(inicio_idx, pasos=6):
        """Simula una ruta desde un índice de inicio"""
        ruta = [inicio_idx]
        idx_actual = inicio_idx
        
        for _ in range(pasos):
            probs = matriz_prob[idx_actual]
            if probs.sum() == 0:
                break
            siguiente_idx = np.random.choice(len(estados), p=probs)
            ruta.append(siguiente_idx)
            idx_actual = siguiente_idx
        
        return ruta
    
    # Múltiples simulaciones desde estaciones importantes
    print("🛤️  Simulaciones desde estaciones clave:")
    
    estaciones_clave = top_actividad[-3:]  # Top 3 más activas
    
    for i, inicio_idx in enumerate(estaciones_clave, 1):
        estado_id = estados[inicio_idx]
        nombre_inicio = nombres.get(estado_id, estado_id)
        
        print(f"\n   Simulación {i} desde: {nombre_inicio}")
        ruta_indices = simular_ruta(inicio_idx, pasos=5)
        
        for j, idx in enumerate(ruta_indices):
            estado_id = estados[idx]
            nombre = nombres.get(estado_id, estado_id)
            if j == 0:
                print(f"      🚏 INICIO: {nombre}")
            elif j == len(ruta_indices) - 1:
                print(f"      🏁 FIN:    {nombre}")
            else:
                print(f"      ➡️  {j:2d}:     {nombre}")
    
    # 7. VISUALIZACIÓN MEJORADA
    print(f"\n🎨 FASE 7: GENERACIÓN DE VISUALIZACIONES")
    print("-" * 45)
    
    # Crear visualización mejorada
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🚌 Análisis de Cadenas de Markov - Metrobús CDMX', fontsize=16, fontweight='bold')
    
    # 1. Heatmap de matriz de transición (top estaciones)
    top_n = min(12, n_estados)
    top_indices = np.argsort(actividad_total)[-top_n:]
    sub_matriz = matriz_prob[np.ix_(top_indices, top_indices)]
    nombres_cortos = [nombres.get(estados[i], estados[i])[:15] for i in top_indices]
    
    sns.heatmap(sub_matriz, 
                xticklabels=nombres_cortos,
                yticklabels=nombres_cortos,
                cmap='YlOrRd',
                annot=False,
                fmt='.2f',
                ax=ax1,
                cbar_kws={'label': 'Probabilidad'})
    ax1.set_title(f'Matriz de Transición (Top {top_n})')
    ax1.set_xlabel('Destino')
    ax1.set_ylabel('Origen')
    
    # 2. Distribución de conectividad
    ax2.hist(actividad_total, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_title('Distribución de Actividad por Estación')
    ax2.set_xlabel('Actividad Total')
    ax2.set_ylabel('Frecuencia')
    ax2.grid(True, alpha=0.3)
    
    # 3. Comparación entrada vs salida
    ax3.scatter(conexiones_entrada, conexiones_salida, alpha=0.7, s=50, color='coral')
    ax3.set_title('Entradas vs Salidas por Estación')
    ax3.set_xlabel('Conexiones Entrantes')
    ax3.set_ylabel('Conexiones Salientes')
    ax3.grid(True, alpha=0.3)
    
    # Agregar línea diagonal
    max_val = max(conexiones_entrada.max(), conexiones_salida.max())
    ax3.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Balance perfecto')
    ax3.legend()
    
    # 4. Evolución de rutas simuladas
    longitudes_rutas = []
    for _ in range(100):  # 100 simulaciones
        inicio_aleatorio = np.random.choice(range(n_estados))
        ruta = simular_ruta(inicio_aleatorio, pasos=15)
        longitudes_rutas.append(len(ruta))
    
    ax4.hist(longitudes_rutas, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
    ax4.set_title('Distribución de Longitudes de Rutas Simuladas')
    ax4.set_xlabel('Longitud de Ruta (paradas)')
    ax4.set_ylabel('Frecuencia')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analisis_completo_metrobus.png', dpi=300, bbox_inches='tight')
    print("💾 Análisis completo guardado como 'analisis_completo_metrobus.png'")
    plt.show()
    
    # 8. REPORTE FINAL
    print(f"\n📋 REPORTE FINAL DEL ANÁLISIS")
    print("="*50)
    
    print(f"🎯 RESUMEN EJECUTIVO:")
    print(f"   • Sistema analizado: {total_stops} estaciones activas")
    print(f"   • Muestra procesada: {len(secuencias)} secuencias de viaje")
    print(f"   • Densidad de conectividad: {densidad:.1f}%")
    print(f"   • Transiciones observadas: {total_transiciones:,}")
    
    estacion_central = nombres.get(estados[top_actividad[-1]], "N/A")
    print(f"\n🏆 HALLAZGOS PRINCIPALES:")
    print(f"   • Estación más central: {estacion_central}")
    print(f"   • Probabilidad máxima observada: {max_prob:.1%}")
    print(f"   • Promedio de longitud de ruta: {np.mean(longitudes_rutas):.1f} paradas")
    
    print(f"\n💡 APLICACIONES:")
    print(f"   • Optimización de rutas y frecuencias")
    print(f"   • Predicción de patrones de demanda")
    print(f"   • Identificación de cuellos de botella")
    print(f"   • Planificación de expansión del sistema")
    
    print(f"\n🌐 Para análisis interactivo ejecuta:")
    print(f"   streamlit run metrobus_markov_app.py")
    
    print(f"\n🎉 ¡ANÁLISIS COMPLETADO EXITOSAMENTE! 🎉")

if __name__ == "__main__":
    demo_completa()
