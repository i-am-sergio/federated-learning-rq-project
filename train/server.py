import flwr as fl
from flwr.server import strategy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import warnings
import torch
import os

warnings.filterwarnings('ignore')

# ====================================================
# ESTRATEGIA CLOUD (Coordina Nodos Fog)
# ====================================================
class CloudStrategy(strategy.FedAvg):
    def __init__(self, min_fit_clients=1, min_evaluate_clients=1, **kwargs):
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            **kwargs
        )
    
    def aggregate_fit(self, server_round, results, failures):
        # Agregación estándar (Promedio de pesos de los Nodos Fog)
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Guardado del modelo en la última ronda
            if server_round == 3:
                print("\n" + "*"*40)
                print("CLOUD: GUARDANDO MODELO GLOBAL FINAL...")
                print("*"*40)
                
                ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
                
                from transformers import AutoModelForSequenceClassification
                model = AutoModelForSequenceClassification.from_pretrained(
                    "microsoft/mpnet-base", num_labels=2
                )
                
                params_dict = zip(model.state_dict().keys(), ndarrays)
                state_dict = {k: torch.tensor(v) for k, v in params_dict}
                model.load_state_dict(state_dict, strict=True)
                
                torch.save(model.state_dict(), "mpnet_fed_requirements.pth")
                print("CLOUD: ¡Modelo guardado como 'mpnet_fed_requirements.pth'!")

        return aggregated_parameters, metrics

# ====================================================
# CONFIGURACIÓN DEL SERVIDOR CLOUD
# ====================================================
def main():
    # NOTA: min_fit_clients debe ser igual a tu número de Nodos Fog activos
    # Si tienes 1 VM de Fog, pon 1. Si tienes 2, pon 2.
    strategy = CloudStrategy(
        min_fit_clients=1,  
        min_evaluate_clients=1,
        min_available_clients=1, 
        on_fit_config_fn=lambda rnd: {"epochs": 1}, 
    )
    
    config = fl.server.ServerConfig(num_rounds=3)
    
    print("\n" + "="*60)
    print("INICIANDO CLOUD SERVER (Puerto 8080)")
    print("Esperando conexiones de Nodos FOG...")
    print("="*60 + "\n")
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
        grpc_max_message_length=1024*1024*1024
    )

if __name__ == "__main__":
    main()