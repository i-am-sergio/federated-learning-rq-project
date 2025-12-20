import flwr as fl
from flwr.server import strategy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ====================================================
# ESTRATEGIA PERSONALIZADA
# ====================================================
class FederatedAverageCustom(strategy.FedAvg):
    def __init__(self, min_fit_clients=2, min_evaluate_clients=2, **kwargs):
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            **kwargs
        )
    
    def aggregate_fit(self, server_round, results, failures):
        """Agregar parámetros usando promedio ponderado"""
        if not results:
            return None, {}
        
        # Convertir resultados (ClientProxy, FitRes) a la estructura que esperas
        # FitRes contiene: parameters, num_examples, metrics, status
        actual_results = [
            (fl.common.parameters_to_ndarrays(res.parameters), res.num_examples)
            for _, res in results
        ]
        
        # Calcular pesos basados en número de muestras
        weights = [num_examples for _, num_examples in actual_results]
        total_samples = sum(weights)
        
        # Normalizar pesos
        weights_normalized = [w / total_samples for w in weights]
        
        # Promedio ponderado de parámetros
        aggregated_parameters = []
        # Tomamos la estructura del primer resultado como base
        for i in range(len(actual_results[0][0])):
            layer_params = []
            for (parameters, _), weight in zip(actual_results, weights_normalized):
                layer_params.append(parameters[i] * weight)
            aggregated_layer = np.sum(layer_params, axis=0)
            aggregated_parameters.append(aggregated_layer)
        
        # Convertir de vuelta a Parameters para Flower
        return fl.common.ndarrays_to_parameters(aggregated_parameters), {}
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Agregar métricas de evaluación"""
        if not results:
            return None, {}
        
        # Results es una lista de (ClientProxy, EvaluateRes)
        # EvaluateRes contiene: loss, num_examples, metrics, status
        total_loss = 0
        total_samples = 0
        total_accuracy = 0
        
        for _, res in results:
            total_loss += res.loss * res.num_examples
            total_samples += res.num_examples
            if "accuracy" in res.metrics:
                total_accuracy += res.metrics["accuracy"] * res.num_examples
        
        aggregated_loss = total_loss / total_samples if total_samples > 0 else 0
        aggregated_accuracy = total_accuracy / total_samples if total_samples > 0 else 0
        
        metrics_aggregated = {
            "loss": aggregated_loss,
            "accuracy": aggregated_accuracy
        }
        
        return aggregated_loss, metrics_aggregated

# ====================================================
# FUNCIÓN DE EVALUACIÓN GLOBAL
# ====================================================
def get_evaluate_fn():
    """Función para evaluación global del modelo"""
    def evaluate(server_round, parameters, config):
        # En un caso real, aquí evaluarías el modelo global en un dataset de prueba
        # Por ahora, retornamos valores dummy
        print(f"\n{'='*50}")
        print(f"Ronda del servidor: {server_round}")
        print(f"Parámetros recibidos para evaluación")
        print(f"{'='*50}")
        
        # Aquí podrías cargar un dataset de prueba y evaluar el modelo
        # Por simplicidad, retornamos valores por defecto
        loss = 0.5  # Valor dummy
        accuracy = 0.5  # Valor dummy
        
        return loss, {"accuracy": accuracy}
    return evaluate

# ====================================================
# CONFIGURACIÓN DEL SERVIDOR
# ====================================================
def main():
    # Configurar estrategia
    strategy = FederatedAverageCustom(
        fraction_fit=1.0,  # Usar todos los clientes disponibles para entrenamiento
        fraction_evaluate=1.0,  # Usar todos los clientes para evaluación
        min_fit_clients=1,  # Mínimo 2 clientes para entrenamiento
        min_evaluate_clients=1,  # Mínimo 2 clientes para evaluación
        min_available_clients=1,  # Esperar al menos 2 clientes
        evaluate_fn=get_evaluate_fn(),  # Función de evaluación global
        on_fit_config_fn=lambda rnd: {"epochs": 1},  # 1 época por ronda
        on_evaluate_config_fn=lambda rnd: {"batch_size": 32},
        initial_parameters=None,  # Inicializar con pesos pre-entrenados
    )
    
    # Configuración del servidor
    config = fl.server.ServerConfig(num_rounds=3)  # 3 rondas de federación
    
    print("\n" + "="*60)
    print("INICIANDO SERVIDOR FEDERADO")
    print("="*60)
    print(f"Rondas de entrenamiento: {config.num_rounds}")
    print(f"Estrategia: FedAvg personalizada")
    print(f"Esperando clientes en: localhost:8080")
    print("="*60 + "\n")
    
    # Iniciar servidor
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
        grpc_max_message_length=1024*1024*1024  # 1GB para modelos grandes
    )

if __name__ == "__main__":
    main()