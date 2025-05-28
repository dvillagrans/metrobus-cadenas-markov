# 🚌 Análisis de Cadenas de Markov - Metrobús CDMX

Este proyecto analiza los patrones de transición entre estaciones del Metrobús de la Ciudad de México usando cadenas de Markov. Incluye dos interfaces: una aplicación web interactiva y un analizador de línea de comandos.

## 🌟 Características

- **📊 Matriz de Transición**: Calcula probabilidades de transición entre estaciones
- **🎨 Visualizaciones Interactivas**: Mapas de calor y redes de conexiones
- **🚀 Simulación de Rutas**: Genera rutas probables usando el modelo Markov
- **📈 Estadísticas Detalladas**: Análisis de conectividad y patrones
- **🌐 Interfaz Web**: Dashboard interactivo con Streamlit
- **⚡ Versión Simple**: Analizador de línea de comandos

## 🛠️ Instalación

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Verificar datos

Asegúrate de tener los archivos GTFS del Metrobús en la carpeta `data/`:
- `stops.txt` - Información de estaciones
- `routes.txt` - Información de rutas  
- `stop_times.txt` - Horarios y secuencias
- `trips.txt` - Información de viajes

## 🚀 Uso

### Opción 1: Aplicación Web Interactiva (Recomendada)

```bash
# Ejecutar la aplicación web
python run_app.py

# O directamente con streamlit
streamlit run metrobus_markov_app.py
```

**Características de la aplicación web:**
- 🔥 **Matriz de Calor**: Visualización interactiva de probabilidades
- 🕸️ **Red de Conexiones**: Grafo de transiciones entre estaciones
- 🎯 **Simulación**: Genera rutas probables desde cualquier estación
- 📊 **Estadísticas**: Rankings de estaciones más conectadas

### Opción 2: Analizador Simple (Línea de Comandos)

```bash
# Ejecutar análisis completo
python metrobus_simple.py
```

**Lo que hace el analizador simple:**
- Carga y procesa los datos GTFS
- Calcula la matriz de transición
- Muestra estadísticas en consola
- Genera mapa de calor (imagen PNG)
- Simula rutas de ejemplo

## 📚 Uso Programático

### Analizador Simple

```python
from metrobus_simple import MetrobusMarkovSimple

# Crear analizador
analizador = MetrobusMarkovSimple()

# Cargar datos
analizador.cargar_datos()

# Procesar y calcular matriz
secuencias = analizador.procesar_secuencias(max_trips=500)
analizador.calcular_matriz_transicion(secuencias)

# Ver estadísticas
analizador.mostrar_estadisticas()

# Simular ruta
analizador.mostrar_ruta_simulada("Buenavista", pasos=10)

# Crear visualización
analizador.visualizar_matriz_calor(top_n=20)
```

### Analizador Avanzado (desde app)

```python
from metrobus_markov_app import MetrobusMarkovAnalyzer

analyzer = MetrobusMarkovAnalyzer()
analyzer.load_data()
sequences = analyzer.preprocess_data()
analyzer.estimate_transition_matrix(sequences)

# Simular ruta
route = analyzer.simulate_markov_chain("fa0784", n_steps=8)
print([analyzer.get_stop_name(stop) for stop in route])
```

## 📊 Interpretación de Resultados

### Matriz de Transición
- **Filas**: Estaciones de origen
- **Columnas**: Estaciones de destino  
- **Valores**: Probabilidad de transición (0-1)
- **Colores**: Más brillante = mayor probabilidad

### Estadísticas Clave
- **Conexiones Salientes**: Estaciones que sirven como origen frecuente
- **Conexiones Entrantes**: Estaciones que son destino frecuente
- **Densidad**: Porcentaje de transiciones posibles que ocurren

### Red de Conexiones
- **Nodos**: Estaciones del Metrobús
- **Aristas**: Transiciones frecuentes
- **Umbral**: Solo se muestran conexiones > umbral especificado

## 🎯 Casos de Uso

### 1. Planificación de Rutas
```python
# Encontrar rutas probables desde una estación
analizador.mostrar_ruta_simulada("Revolución", pasos=15)
```

### 2. Análisis de Conectividad
```python
# Ver estaciones más importantes del sistema
analizador.mostrar_estadisticas()
```

### 3. Optimización del Sistema
- Identificar cuellos de botella
- Analizar patrones de demanda
- Optimizar frecuencias de servicio

## 📁 Estructura del Proyecto

```
procesos-estocasticos/
├── data/                          # Datos GTFS del Metrobús
│   ├── stops.txt
│   ├── routes.txt
│   ├── stop_times.txt
│   └── trips.txt
├── metrobus_markov_app.py        # Aplicación web Streamlit
├── metrobus_simple.py            # Analizador de línea de comandos
├── markov_basic.py               # Implementación base de Markov
├── run_app.py                    # Script de ejecución
├── requirements.txt              # Dependencias
└── README.md                     # Esta documentación
```

## 🔧 Parámetros Configurables

### En la Aplicación Web
- **Top N estaciones**: Número de estaciones a mostrar en visualizaciones
- **Umbral de red**: Probabilidad mínima para mostrar conexiones
- **Pasos de simulación**: Longitud de rutas simuladas

### En el Analizador Simple
- **max_trips**: Número máximo de viajes a procesar
- **top_n**: Estaciones a mostrar en visualizaciones
- **pasos**: Longitud de simulaciones

## 📈 Interpretación Matemática

### Matriz de Transición P
```
P[i,j] = P(X_{t+1} = j | X_t = i)
```

Donde:
- `X_t` es el estado (estación) en el tiempo t
- `P[i,j]` es la probabilidad de ir de estación i a estación j

### Propiedades
- **Estocástica**: Cada fila suma 1
- **Dispersa**: Muchas transiciones tienen probabilidad 0
- **Dependiente del tiempo**: Refleja patrones de horarios

## 🛠️ Troubleshooting

### Problema: "No se encontraron datos"
**Solución**: Verifica que los archivos GTFS estén en la carpeta `data/`

### Problema: "Error al instalar dependencias"
**Solución**: 
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Problema: "La aplicación web no abre"
**Solución**:
```bash
python -m streamlit run metrobus_markov_app.py
```

## 📞 Contacto y Contribuciones

Este proyecto es educativo y puede ser extendido para:
- Otros sistemas de transporte público
- Análisis temporales (día/noche, días laborales/fines de semana)
- Predicción de demanda
- Optimización de rutas

¡Las contribuciones y mejoras son bienvenidas! 🎉
