"""
Script para instalar las dependencias de optimización para el análisis de Markov
"""

import subprocess
import sys
import os

def install_package(package, description):
    """Instala un paquete con manejo de errores"""
    print(f"\n🔄 Instalando {description}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {description} instalado correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando {description}: {e}")
        return False

def main():
    print("🚀 INSTALADOR DE DEPENDENCIAS DE OPTIMIZACIÓN")
    print("=" * 50)
    
    # Dependencias básicas (requeridas)
    basic_packages = [
        ("psutil", "Monitor del sistema"),
        ("multiprocessing-logging", "Logging para multiprocesamiento"),
    ]
    
    # Dependencias de paralelización (opcionales pero recomendadas)
    parallel_packages = [
        ("dask[complete]", "Dask para computación paralela"),
        ("numba", "Numba para compilación JIT"),
    ]
    
    # Dependencias GPU (opcionales, requieren CUDA)
    gpu_packages = [
        ("cupy-cuda12x", "CuPy para aceleración GPU (CUDA 12.x)"),
        # Alternativas para diferentes versiones de CUDA:
        # ("cupy-cuda11x", "CuPy para CUDA 11.x"),
        # ("cupy-cuda10x", "CuPy para CUDA 10.x"),
    ]
    
    print("📦 Instalando dependencias básicas...")
    for package, desc in basic_packages:
        install_package(package, desc)
    
    print("\n⚡ Instalando dependencias de paralelización...")
    for package, desc in parallel_packages:
        install_package(package, desc)
    
    # Preguntar sobre GPU
    install_gpu = input("\n🎮 ¿Quieres instalar soporte para GPU (requiere NVIDIA GPU con CUDA)? (s/n): ").lower()
    
    if install_gpu in ['s', 'si', 'y', 'yes']:
        print("\n🚀 Instalando dependencias GPU...")
        print("⚠️  Nota: Esto requiere una GPU NVIDIA con CUDA instalado")
        
        for package, desc in gpu_packages:
            success = install_package(package, desc)
            if not success:
                print("💡 Si falla, puedes intentar con:")
                print("   pip install cupy-cuda11x  # Para CUDA 11.x")
                print("   pip install cupy-cuda10x  # Para CUDA 10.x")
                break
    
    print("\n" + "=" * 50)
    print("✅ INSTALACIÓN COMPLETADA")
    print("\n📋 Resumen de optimizaciones disponibles:")
    print("   • Paralelización CPU: multiprocessing")
    print("   • Computación distribuida: Dask")
    print("   • Compilación JIT: Numba")
    if install_gpu in ['s', 'si', 'y', 'yes']:
        print("   • Aceleración GPU: CuPy (si se instaló correctamente)")
    
    print("\n🚀 ¡Ya puedes usar la versión optimizada del analizador!")
    print("   python metrobus_markov_optimized.py")

if __name__ == "__main__":
    main()
