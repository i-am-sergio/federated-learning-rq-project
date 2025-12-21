import flwr as fl
from flwr.server import strategy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import warnings
import torch
import requests
import time
import os
from google.cloud import storage
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
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            if server_round == 3:
                print("\n" + "*"*30)
                print("GUARDANDO MODELO GLOBAL FINAL...")
                print("*"*30)
                ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
                from transformers import AutoModelForSequenceClassification
                model = AutoModelForSequenceClassification.from_pretrained(
                    "microsoft/mpnet-base", num_labels=2
                )
                params_dict = zip(model.state_dict().keys(), ndarrays)
                state_dict = {k: torch.tensor(v) for k, v in params_dict}
                model.load_state_dict(state_dict, strict=True)
                
                torch.save(model.state_dict(), "mpnet_fed_requirements.pth")
                print("Modelo guardado exitosamente como 'mpnet_fed_requirements.pth'")
                
                bucket_name = os.environ.get("MODEL_BUCKET_NAME")
                if bucket_name:
                    print(f"Subiendo modelo a Bucket: {bucket_name}...")
                    try:
                        storage_client = storage.Client()
                        bucket = storage_client.bucket(bucket_name)
                        blob = bucket.blob("mpnet_fed_requirements.pth")
                        blob.upload_from_filename("mpnet_fed_requirements.pth")
                        print("Modelo subido exitosamente a GCS (Versión guardada).")
                    except Exception as e:
                        print(f"Error subiendo a GCS: {e}")
                
                print("Notificando a la API de Inferencia para recarga...")
                try:
                    response = requests.post("http://127.0.0.1:8000/reload_model")
                    if response.status_code == 200:
                        print(f"ÉXITO: {response.json()}")
                    else:
                        print(f"Error en recarga: {response.text}")
                except Exception as e:
                    print(f"No se pudo contactar a la API: {e}")
        return aggregated_parameters, metrics
    
    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
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
        
        return aggregated_loss, {"loss": aggregated_loss, "accuracy": aggregated_accuracy}

# ====================================================
# FUNCIÓN DE EVALUACIÓN GLOBAL
# ====================================================
def get_evaluate_fn():
    """Función para evaluación global del modelo"""
    def evaluate(server_round, parameters, config):
        print(f"\n{'='*50}")
        print(f"Ronda del servidor: {server_round}")
        print(f"Parámetros recibidos para evaluación")
        print(f"{'='*50}")
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
    
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
        grpc_max_message_length=1024*1024*1024 
    )

if __name__ == "__main__":
    main()