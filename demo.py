"""
Demostración rápida del análisis de Markov para Metrobús
Script simple para probar que todo funcione
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def demo_rapida():
    """Demostración rápida del análisis"""
    print("🚌 DEMO RÁPIDA - ANÁLISIS MARKOV METROBÚS")
    print("="*50)
    
    # Verificar archivos
    data_path = Path("data")
    archivos_requeridos = ["stops.txt", "routes.txt", "stop_times.txt"]
    
    print("📂 Verificando archivos...")
    for archivo in archivos_requeridos:
        if (data_path / archivo).exists():
            print(f"   ✅ {archivo}")
        else:
            print(f"   ❌ {archivo} - NO ENCONTRADO")
            return False
    
    # Cargar datos básicos
    try:
        stops_df = pd.read_csv(data_path / "stops.txt")
        stop_times_df = pd.read_csv(data_path / "stop_times.txt")
        
        print(f"✅ Datos cargados:")
        print(f"   🚉 Estaciones: {len(stops_df)}")
        print(f"   ⏰ Registros de tiempo: {len(stop_times_df)}")
        
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return False
    
    # Procesar una muestra pequeña
    print("\n🔄 Procesando muestra de datos...")
    
    # Tomar los primeros 50 viajes únicos
    sample_trips = stop_times_df['trip_id'].unique()[:50]
    secuencias = []
    
    for trip_id in sample_trips:
        trip_data = stop_times_df[stop_times_df['trip_id'] == trip_id].sort_values('stop_sequence')
        if len(trip_data) > 1:
            secuencias.append(trip_data['stop_id'].tolist())
    
    print(f"📊 Secuencias procesadas: {len(secuencias)}")
    
    # Calcular matriz simple
    print("\n🧮 Calculando matriz de transición...")
    
    # Obtener estados únicos
    todos_estados = set()
    for seq in secuencias:
        todos_estados.update(seq)
    
    estados = sorted(list(todos_estados))
    n_estados = len(estados)
    
    print(f"🚉 Estados únicos: {n_estados}")
    
    # Mapeo
    estado_a_idx = {estado: i for i, estado in enumerate(estados)}
    
    # Matriz de conteos
    conteos = np.zeros((n_estados, n_estados))
    
    for seq in secuencias:
        for i in range(len(seq) - 1):
            from_idx = estado_a_idx[seq[i]]
            to_idx = estado_a_idx[seq[i + 1]]
            conteos[from_idx, to_idx] += 1
    
    # Normalizar
    sumas = conteos.sum(axis=1)
    sumas[sumas == 0] = 1
    matriz_prob = conteos / sumas[:, np.newaxis]
    
    # Estadísticas básicas
    transiciones_activas = np.count_nonzero(conteos)
    densidad = (transiciones_activas / (n_estados**2)) * 100
    
    print(f"✅ Matriz calculada: {n_estados}x{n_estados}")
    print(f"🔗 Transiciones activas: {transiciones_activas}")
    print(f"📈 Densidad: {densidad:.1f}%")
    
    # Mostrar top estaciones
    print("\n📊 TOP 5 ESTACIONES MÁS CONECTADAS:")
    actividad = matriz_prob.sum(axis=1) + matriz_prob.sum(axis=0)
    top_indices = np.argsort(actividad)[-5:]
    
    # Mapeo de nombres
    nombres = dict(zip(stops_df['stop_id'], stops_df['stop_name']))
    
    for i, idx in enumerate(reversed(top_indices), 1):
        estado_id = estados[idx]
        nombre = nombres.get(estado_id, estado_id)
        print(f"   {i}. {nombre[:30]} (actividad: {actividad[idx]:.3f})")
    
    # Simulación simple
    print("\n🎯 SIMULACIÓN DE RUTA (ejemplo):")
    if len(estados) > 0:
        # Empezar desde la estación más activa
        inicio_idx = top_indices[-1]
        ruta_sim = [estados[inicio_idx]]
        idx_actual = inicio_idx
        
        for paso in range(5):
            probs = matriz_prob[idx_actual]
            if probs.sum() > 0:
                siguiente_idx = np.random.choice(len(estados), p=probs)
                ruta_sim.append(estados[siguiente_idx])
                idx_actual = siguiente_idx
            else:
                break
        
        print("🛤️  Ruta simulada:")
        for i, stop_id in enumerate(ruta_sim):
            nombre = nombres.get(stop_id, stop_id)
            print(f"   {i+1}. {nombre}")
    
    print(f"\n🎉 ¡Demo completada exitosamente!")
    print(f"💡 Ejecuta 'python metrobus_simple.py' para análisis completo")
    print(f"🌐 O 'python run_app.py' para la aplicación web interactiva")
    
    return True

if __name__ == "__main__":
    demo_rapida()
