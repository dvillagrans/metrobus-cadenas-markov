"""
Script para verificar cuántas estaciones hay en el conjunto de datos del Metrobús
y cuántas se usan realmente en el análisis de Markov
"""

import pandas as pd
from pathlib import Path

# Cargar datos
data_path = Path("data")
stops_df = pd.read_csv(data_path / "stops.txt")
stop_times_df = pd.read_csv(data_path / "stop_times.txt")

print("=== ANÁLISIS DEL CONJUNTO DE DATOS DEL METROBÚS ===\n")

print("📊 DATOS GENERALES:")
print(f"• Total de estaciones en stops.txt: {len(stops_df)}")
print(f"• Estaciones únicas por stop_id: {stops_df['stop_id'].nunique()}")
print(f"• Estaciones únicas por stop_name: {stops_df['stop_name'].nunique()}")

print(f"\n📈 DATOS DE STOP_TIMES:")
print(f"• Total de registros en stop_times.txt: {len(stop_times_df)}")
print(f"• Estaciones únicas en stop_times: {stop_times_df['stop_id'].nunique()}")
print(f"• Viajes únicos totales: {stop_times_df['trip_id'].nunique()}")

print(f"\n🔍 ANÁLISIS RÁPIDO:")
# Manera más eficiente: usar directamente los datos únicos
stops_in_100_trips = stop_times_df[
    stop_times_df['trip_id'].isin(stop_times_df['trip_id'].unique()[:100])
]['stop_id'].nunique()

all_stops_in_data = stop_times_df['stop_id'].nunique()

print(f"• Estaciones únicas en muestra de 100 viajes: {stops_in_100_trips}")
print(f"• Estaciones únicas en TODOS los viajes: {all_stops_in_data}")

print(f"\n💡 CONCLUSIÓN:")
print(f"• El código actual solo analiza {stops_in_100_trips} estaciones de las {all_stops_in_data} disponibles")
print(f"• Esto representa el {stops_in_100_trips/all_stops_in_data*100:.1f}% del total")

# Mostrar algunas estaciones de ejemplo
sample_stops = stop_times_df[
    stop_times_df['trip_id'].isin(stop_times_df['trip_id'].unique()[:100])
]['stop_id'].unique()[:10]

print(f"\n📋 PRIMERAS 10 ESTACIONES EN LA MUESTRA:")
for i, stop_id in enumerate(sample_stops):
    stop_name = stops_df[stops_df['stop_id'] == stop_id]['stop_name'].iloc[0]
    print(f"  {i+1}. {stop_id} - {stop_name}")
