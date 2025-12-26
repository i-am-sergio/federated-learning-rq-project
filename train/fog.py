import flwr as fl
import sys
import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg

# ====================================================
# ESTADO COMPARTIDO (Puente Cloud <-> Edge)
# ====================================================
class FogSharedState:
    def __init__(self):
        self.global_parameters = None 
        self.aggregated_parameters = None 

shared_state = FogSharedState()

# ====================================================
# ESTRATEGIA FOG (Coordina Clientes Edge)
# ====================================================
class FogStrategy(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # 1. Promediar resultados de los clientes locales (Edge)
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        # 2. Guardar el resultado en memoria para enviarlo hacia la Nube
        if aggregated_parameters is not None:
            print(f"\n[FOG LOGIC] Agregación de Edge completada. Listo para subir a Cloud.")
            shared_state.aggregated_parameters = aggregated_parameters
            
        return aggregated_parameters, metrics

# ====================================================
# CLIENTE FOG (Se conecta a Cloud)
# ====================================================
class FogClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        if shared_state.global_parameters:
            return parameters_to_ndarrays(shared_state.global_parameters)
        return []

    def fit(self, parameters, config):
        print(f"\n[FOG CLIENT] Recibidos pesos de CLOUD. Iniciando ronda EDGE...")
        
        # 1. Actualizar pesos base con lo que mandó la nube
        shared_state.global_parameters = ndarrays_to_parameters(parameters)
        
        # 2. INICIAR SERVIDOR FOG INTERNO (Bloqueante)
        # Este servidor escucha en el puerto 8081 a los dispositivos finales
        fl.server.start_server(
            server_address="0.0.0.0:8081", 
            config=ServerConfig(num_rounds=1), # 1 ronda local por cada ronda global
            strategy=FogStrategy(
                min_fit_clients=2, # Espera a 2 clientes Edge (Laptop 1, Laptop 2)
                min_available_clients=2,
                initial_parameters=shared_state.global_parameters
            ),
            grpc_max_message_length=1024*1024*1024
        )
        
        # 3. Retornar los pesos ya agregados a la Nube
        if shared_state.aggregated_parameters:
            new_parameters = parameters_to_ndarrays(shared_state.aggregated_parameters)
        else:
            print("[FOG ERROR] No hubo agregación local, devolviendo pesos originales.")
            new_parameters = parameters
            
        # El Fog actúa como 1 "médium" de datos, weight=1 (o proporcional a sus clientes conectados)
        return new_parameters, 1, {} 

    def evaluate(self, parameters, config):
        return 0.5, 1, {"accuracy": 0.0}

# ====================================================
# EJECUCIÓN
# ====================================================
def main():
    # IP PÚBLICA DE LA VM CLOUD (Cambiar por la real de server.py)
    CLOUD_SERVER_IP = "34.134.252.161" 
    
    print(f"--> [FOG] Conectando UPSTREAM a Cloud: {CLOUD_SERVER_IP}:8080")
    print(f"--> [FOG] Escuchando DOWNSTREAM a Edge: 0.0.0.0:8081")

    fl.client.start_numpy_client(
        server_address=f"{CLOUD_SERVER_IP}:8080",
        client=FogClient(),
        grpc_max_message_length=1024*1024*1024
    )

if __name__ == "__main__":
    main()