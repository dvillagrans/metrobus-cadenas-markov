"""
Script de ejecución para la aplicación de análisis Markov del Metrobús
Ejecuta la aplicación Streamlit de forma sencilla
"""

import subprocess
import sys
from pathlib import Path

def run_streamlit_app():
    """Ejecuta la aplicación Streamlit"""
    app_file = Path(__file__).parent / "metrobus_markov_app.py"
    
    print("🚌 Iniciando aplicación de análisis Markov del Metrobús CDMX...")
    print(f"📁 Archivo de aplicación: {app_file}")
    print("🌐 La aplicación se abrirá en tu navegador web")
    print("⏹️  Para detener la aplicación, presiona Ctrl+C en la terminal")
    print("-" * 60)
    
    try:
        # Ejecutar streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", str(app_file)]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al ejecutar la aplicación: {e}")
        print("💡 Asegúrate de tener instaladas las dependencias:")
        print("   pip install -r requirements.txt")
    except KeyboardInterrupt:
        print("\n👋 Aplicación detenida por el usuario")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

if __name__ == "__main__":
    run_streamlit_app()
