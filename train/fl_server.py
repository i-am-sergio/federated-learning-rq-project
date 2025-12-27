import flwr as fl
from flwr.server import strategy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import warnings
import torch
import os
from google.cloud import storage
warnings.filterwarnings('ignore')

# ====================================================
# FUNCIÓN AUXILIAR PARA SUBIR A GCS
# ====================================================
def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        print(f"EXITO: {source_file_name} subido a gs://{bucket_name}/{destination_blob_name}")
    except Exception as e:
        print(f"ERROR: Fallo la subida a GCS: {e}")

# ====================================================
# ESTRATEGIA PERSONALIZADA
# ====================================================
class FederatedAverageCustom(strategy.FedAvg):
    def __init__(self, num_labels=2, total_rounds=3, min_fit_clients=1, min_evaluate_clients=1, **kwargs):
        self.num_labels = num_labels
        self.total_rounds = total_rounds
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            **kwargs
        )
    
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None and server_round == self.total_rounds:
            print("\n" + "*"*40)
            print(f"GUARDANDO MODELO GLOBAL FINAL (Ronda {server_round})")
            print("*"*40)
            
            ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            from transformers import AutoModelForSequenceClassification
            
            # Reconstruir modelo
            model = AutoModelForSequenceClassification.from_pretrained(
                "microsoft/mpnet-base", num_labels=self.num_labels
            )
            
            params_dict = zip(model.state_dict().keys(), ndarrays)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)
            
            # Nombre del archivo final
            filename = "mpnet_fed_multiclass.pth" if self.num_labels > 2 else "mpnet_fed_requirements.pth"
            
            # Guardar localmente
            torch.save(model.state_dict(), filename)
            print(f"Modelo guardado localmente: {filename}")

            # Subir a Google Cloud Storage
            bucket_name = os.getenv("MODEL_BUCKET_NAME")
            if bucket_name:
                print(f"Subiendo a Bucket: {bucket_name}...")
                upload_to_gcs(bucket_name, filename, filename)
            else:
                print("ADVERTENCIA: Variable 'MODEL_BUCKET_NAME' no encontrada.")
        
        return aggregated_parameters, metrics
    
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
    import sys
    
    usar_multiclase = False
    if len(sys.argv) > 1 and sys.argv[1] == "1":
        usar_multiclase = True
    
    num_labels = 12 if usar_multiclase else 2
    
    if usar_multiclase:
        num_labels = 12
        num_rounds = 5 
        epochs_per_round = 3
    else:
        num_labels = 2
        num_rounds = 3
        epochs_per_round = 1
    
    # Configurar estrategia
    strategy = FederatedAverageCustom(
        num_labels=num_labels,
        total_rounds=num_rounds,
        fraction_fit=1.0,  # Usar todos los clientes disponibles para entrenamiento
        fraction_evaluate=1.0,  # Usar todos los clientes para evaluación
        min_fit_clients=1,  # Mínimo 2 clientes para entrenamiento
        min_evaluate_clients=1,  # Mínimo 2 clientes para evaluación
        min_available_clients=1,  # Esperar al menos 2 clientes
        evaluate_fn=None,  # Función de evaluación global
        on_fit_config_fn=lambda rnd: {"epochs": epochs_per_round}, # época por ronda
        on_evaluate_config_fn=lambda rnd: {"batch_size": 32},
        initial_parameters=None,  # Inicializar con pesos pre-entrenados
    )
    
    # Configuración del servidor
    config = fl.server.ServerConfig(num_rounds=num_rounds) # rondas de federación
    print(f"INICIANDO SERVIDOR - MODO: {'MULTICLASE (12)' if usar_multiclase else 'BINARIO (F/NF)'}")
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
        grpc_max_message_length=1024*1024*1024
    )

if __name__ == "__main__":
    main()