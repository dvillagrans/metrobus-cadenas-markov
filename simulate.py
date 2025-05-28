"""
Simulation and Visualization Module
Este módulo integra Markov chains y MDP para simular 10,000 recorridos
y generar comparaciones visuales entre políticas baseline y MDP.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from tqdm import tqdm
import pickle
import os

from markov_basic import MarkovChain, load_gtfs_data, calculate_headways
from mdp_holding import TransportMDP

class TransportSimulator:
    """Simulador completo para análisis de políticas de transporte"""
    
    def __init__(self, gtfs_data_path: str):
        """
        Inicializa el simulador
        
        Args:
            gtfs_data_path: Ruta a los datos GTFS
        """
        self.gtfs_data_path = gtfs_data_path
        self.gtfs_data = None
        self.headways_df = None
        self.markov_chain = None
        self.mdp = None
        
        # Configuración de simulación
        self.n_simulations = 10000
        self.n_steps_per_simulation = 50
        
        # Resultados de simulación
        self.baseline_results = None
        self.mdp_results = None
        self.headway_var_baseline = None
        self.headway_var_mdp = None
        
    def load_and_preprocess_data(self):
        """Carga y preprocesa los datos GTFS"""
        print("Cargando datos GTFS...")
        self.gtfs_data = load_gtfs_data(self.gtfs_data_path)
        
        print("Calculando headways...")
        self.headways_df = calculate_headways(self.gtfs_data['stop_times'])
        
        # Crear archivo parquet con headways para análisis posterior
        os.makedirs('parsed_gtfs_rt', exist_ok=True)
        self.headways_df.to_parquet('parsed_gtfs_rt/headways.parquet')
        
        print(f"Datos procesados: {len(self.headways_df)} headways calculados")    
    def setup_markov_chain(self):
        """Configura y entrena la cadena de Markov baseline"""
        if self.headways_df is None:
            raise ValueError("Debe cargar los datos primero")
            
        print("Configurando cadena de Markov baseline...")
        
        self.markov_chain = MarkovChain()
        
        # Discretizar headways para crear secuencias de estados
        headway_values = np.array(self.headways_df['headway_minutes'].values, dtype=float)
        
        # Crear bins para discretización
        bins = np.percentile(headway_values, [0, 20, 40, 60, 80, 100])
        digitized = np.digitize(headway_values, bins)
        
        # Crear secuencias por parada
        sequences = []
        for stop_id in self.headways_df['stop_id'].unique():
            stop_data = self.headways_df[self.headways_df['stop_id'] == stop_id]
            stop_data = stop_data.sort_values('arrival_time')
            
            if len(stop_data) > 1:
                stop_headways = np.array(stop_data['headway_minutes'].values, dtype=float)
                stop_states = np.digitize(stop_headways, bins)
                sequences.append(stop_states.tolist())
        
        # Entrenar cadena de Markov
        self.markov_chain.estimate_transition_matrix(sequences)
        
        print(f"Cadena de Markov entrenada con {len(sequences)} secuencias")
        
    def setup_mdp(self):
        """Configura y entrena el MDP"""
        print("Configurando MDP...")
        
        # Definir espacios de estados y acciones basados en datos
        headway_percentiles = 
        np.percentile(
            self.headways_df['headway_minutes'].values, 
            [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100]
            )
        holding_actions = [0.0, 0.5, 1.0, 2.0, 3.0]  # minutos de retención

        self.mdp = TransportMDP(headway_percentiles.tolist(), holding_actions)

        # Crear secuencias de estados para el MDP
        sequences = []
        for stop_id in self.headways_df['stop_id'].unique()[:50]:  # Limitar para eficiencia
            stop_data = self.headways_df[self.headways_df['stop_id'] == stop_id]
            stop_data = stop_data.sort_values('arrival_time')
            
            if len(stop_data) > 5:
                stop_headways = stop_data['headway_minutes'].values
                stop_states = self.mdp.discretize_headways(stop_headways)
                # Filtrar estados válidos
                stop_states = np.clip(stop_states, 0, self.mdp.n_states - 1)
                sequences.append(stop_states.tolist())
        
        # Entrenar MDP
        self.mdp.estimate_transition_probabilities(sequences)
        self.mdp.calculate_reward_matrix()
        self.mdp.value_iteration()
        
        print(f"MDP entrenado con {len(sequences)} secuencias")
        
    def run_baseline_simulation(self):
        """Ejecuta simulación con política baseline (Markov chain)"""
        print("Ejecutando simulación baseline...")
        
        # Estados iniciales aleatorios
        initial_states = np.random.choice(
            self.markov_chain.states, 
            size=self.n_simulations
        ).tolist()
        
        # Simular múltiples cadenas
        baseline_sequences = self.markov_chain.simulate_multiple_chains(
            initial_states, 
            self.n_steps_per_simulation, 
            self.n_simulations
        )
        
        # Calcular varianza de headways
        self.headway_var_baseline = []
        
        for sequence in tqdm(baseline_sequences, desc="Procesando secuencias baseline"):
            # Convertir estados a headways (mapeo simple)
            headways = [state * 5 + 2.5 for state in sequence if isinstance(state, (int, float))]
            if len(headways) > 1:
                self.headway_var_baseline.append(np.var(headways))
        
        self.baseline_results = {
            'sequences': baseline_sequences,
            'headway_variance': np.array(self.headway_var_baseline),
            'mean_variance': np.mean(self.headway_var_baseline),
            'std_variance': np.std(self.headway_var_baseline)
        }
        
        print(f"Simulación baseline completada: {len(self.headway_var_baseline)} recorridos")
        
    def run_mdp_simulation(self):
        """Ejecuta simulación con política MDP óptima"""
        print("Ejecutando simulación MDP...")
        
        # Estados iniciales aleatorios
        initial_states = np.random.randint(
            0, self.mdp.n_states, 
            size=self.n_simulations
        ).tolist()
        
        # Simular con política MDP
        mdp_sequences = self.mdp.simulate_mdp_policy(
            initial_states, 
            self.n_steps_per_simulation, 
            self.n_simulations
        )
        
        # Calcular varianza de headways
        self.headway_var_mdp = []
        
        for sequence in tqdm(mdp_sequences, desc="Procesando secuencias MDP"):
            # Convertir estados a headways usando bins del MDP
            headways = []
            for state in sequence:
                if 0 <= state < self.mdp.n_states:
                    headway = (self.mdp.headway_bins[state] + self.mdp.headway_bins[state + 1]) / 2
                    headways.append(headway)
            
            if len(headways) > 1:
                self.headway_var_mdp.append(np.var(headways))
        
        self.mdp_results = {
            'sequences': mdp_sequences,
            'headway_variance': np.array(self.headway_var_mdp),
            'mean_variance': np.mean(self.headway_var_mdp),
            'std_variance': np.std(self.headway_var_mdp)
        }
        
        print(f"Simulación MDP completada: {len(self.headway_var_mdp)} recorridos")
        
    def generate_comparison_plots(self):
        """Genera plots de comparación entre políticas baseline y MDP"""
        print("Generando plots de comparación...")
        
        # Configurar estilo
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Crear figura con múltiples subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comparación de Políticas: Baseline vs MDP', fontsize=16, fontweight='bold')
        
        # 1. Histogramas de varianza de headways
        axes[0, 0].hist(self.headway_var_baseline, bins=50, alpha=0.7, 
                       label='Baseline (Markov)', color='skyblue', density=True)
        axes[0, 0].hist(self.headway_var_mdp, bins=50, alpha=0.7, 
                       label='MDP Óptimo', color='lightcoral', density=True)
        axes[0, 0].set_xlabel('Varianza de Headways')
        axes[0, 0].set_ylabel('Densidad')
        axes[0, 0].set_title('Distribución de Varianza de Headways')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Box plots de comparación
        variance_data = [self.headway_var_baseline, self.headway_var_mdp]
        box_plot = axes[0, 1].boxplot(variance_data, labels=['Baseline', 'MDP'], patch_artist=True)
        box_plot['boxes'][0].set_facecolor('skyblue')
        box_plot['boxes'][1].set_facecolor('lightcoral')
        axes[0, 1].set_ylabel('Varianza de Headways')
        axes[0, 1].set_title('Comparación de Varianza por Método')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Estadísticas de resumen
        stats_text = f"""
        BASELINE (Markov Chain):
        Media: {self.baseline_results['mean_variance']:.2f}
        Std: {self.baseline_results['std_variance']:.2f}
        Mediana: {np.median(self.headway_var_baseline):.2f}
        
        MDP ÓPTIMO:
        Media: {self.mdp_results['mean_variance']:.2f}
        Std: {self.mdp_results['std_variance']:.2f}
        Mediana: {np.median(self.headway_var_mdp):.2f}
        
        MEJORA:
        Reducción media: {((self.baseline_results['mean_variance'] - self.mdp_results['mean_variance']) / self.baseline_results['mean_variance'] * 100):.1f}%
        """
        
        axes[1, 0].text(0.05, 0.95, stats_text, transform=axes[1, 0].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].axis('off')
        axes[1, 0].set_title('Estadísticas de Comparación')
        
        # 4. Política MDP óptima
        if hasattr(self.mdp, 'policy') and self.mdp.policy is not None:
            state_centers = [(self.mdp.headway_bins[i] + self.mdp.headway_bins[i + 1]) / 2 
                           for i in range(self.mdp.n_states)]
            optimal_actions = [self.mdp.holding_actions[self.mdp.policy[s]] for s in range(self.mdp.n_states)]
            
            bars = axes[1, 1].bar(range(self.mdp.n_states), optimal_actions, color='lightgreen', alpha=0.7)
            axes[1, 1].set_xlabel('Estado (Headway promedio)')
            axes[1, 1].set_ylabel('Acción óptima (min retención)')
            axes[1, 1].set_title('Política Óptima de Retención')
            axes[1, 1].set_xticks(range(self.mdp.n_states))
            axes[1, 1].set_xticklabels([f'{sc:.1f}' for sc in state_centers], rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            
            # Añadir valores en las barras
            for i, bar in enumerate(bars):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('comparison_baseline_vs_mdp.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_results(self, filename: str = 'simulation_results.pkl'):
        """Guarda los resultados de la simulación"""
        results = {
            'baseline_results': self.baseline_results,
            'mdp_results': self.mdp_results,
            'headway_var_baseline': self.headway_var_baseline,
            'headway_var_mdp': self.headway_var_mdp,
            'simulation_params': {
                'n_simulations': self.n_simulations,
                'n_steps_per_simulation': self.n_steps_per_simulation
            }
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Resultados guardados en {filename}")
    
    def run_complete_simulation(self):
        """Ejecuta la simulación completa"""
        print("=== INICIANDO SIMULACIÓN COMPLETA ===")
        
        # Paso 1: Cargar y preprocesar datos
        self.load_and_preprocess_data()
        
        # Paso 2: Configurar modelos
        self.setup_markov_chain()
        self.setup_mdp()
        
        # Paso 3: Ejecutar simulaciones
        self.run_baseline_simulation()
        self.run_mdp_simulation()
        
        # Paso 4: Generar visualizaciones
        self.generate_comparison_plots()
        
        # Paso 5: Guardar resultados
        self.save_results()
        
        print("=== SIMULACIÓN COMPLETADA ===")
        
        # Resumen final
        print("\nRESUMEN DE RESULTADOS:")
        print(f"Simulaciones ejecutadas: {self.n_simulations}")
        print(f"Pasos por simulación: {self.n_steps_per_simulation}")
        print(f"Varianza promedio baseline: {self.baseline_results['mean_variance']:.3f}")
        print(f"Varianza promedio MDP: {self.mdp_results['mean_variance']:.3f}")
        
        improvement = ((self.baseline_results['mean_variance'] - self.mdp_results['mean_variance']) 
                      / self.baseline_results['mean_variance'] * 100)
        print(f"Mejora con MDP: {improvement:.1f}%")

if __name__ == "__main__":
    # Ejecutar simulación completa
    simulator = TransportSimulator("data")
    simulator.run_complete_simulation()
