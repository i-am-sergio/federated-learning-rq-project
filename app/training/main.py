import os
import torch
import flwr as fl
from flwr.server.strategy import FedAvg
from transformers import AutoModelForSequenceClassification
from google.cloud import storage
import shutil

# --- 1. Domain Services (Lógica de Negocio) ---

class ModelFactory:
    """Factory Pattern: Crea instancias del modelo."""
    def __init__(self, model_name: str, num_labels: int):
        self.model_name = model_name
        self.num_labels = num_labels

    def create(self):
        return AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )

class GCSModelPersistor:
    """Repository Pattern: Maneja la persistencia en Google Cloud Storage."""
    def __init__(self, bucket_name: str, factory: ModelFactory):
        self.bucket_name = bucket_name
        self.factory = factory

    def save_checkpoint(self, parameters, destination_blob_prefix="prod_model"):
        print(f"--> Persistiendo modelo en bucket: {self.bucket_name}")
        
        # Reconstruir modelo
        model = self.factory.create()
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        
        # Guardar localmente
        local_path = "/tmp/saved_model"
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        model.save_pretrained(local_path)
        
        # Subir a GCS
        client = storage.Client()
        bucket = client.bucket(self.bucket_name)
        for filename in os.listdir(local_path):
            blob = bucket.blob(f"{destination_blob_prefix}/{filename}")
            blob.upload_from_filename(os.path.join(local_path, filename))
        print("--> Carga completada.")

# --- 2. Application Strategy (Orquestación) ---

class SaveOnCompleteStrategy(FedAvg):
    """Strategy Pattern extendido: Guarda el modelo al finalizar las rondas."""
    def __init__(self, persistor: GCSModelPersistor, total_rounds: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.persistor = persistor
        self.total_rounds = total_rounds

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None and server_round == self.total_rounds:
            print("Entrenamiento finalizado. Ejecutando persistencia...")
            ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            self.persistor.save_checkpoint(ndarrays)
            
        return aggregated_parameters, aggregated_metrics

# --- 3. Entry Point ---

def main():
    # Inyección de Dependencias desde variables de entorno (puestas por Pulumi)
    MODEL_NAME = os.environ.get("MODEL_NAME", "microsoft/mpnet-base")
    BUCKET_NAME = os.environ.get("BUCKET_NAME")
    ROUNDS = int(os.environ.get("ROUNDS", 3))
    
    # Composición de servicios
    factory = ModelFactory(MODEL_NAME, num_labels=2)
    persistor = GCSModelPersistor(BUCKET_NAME, factory)
    
    initial_params = [val.cpu().numpy() for _, val in factory.create().state_dict().items()]
    
    strategy = SaveOnCompleteStrategy(
        persistor=persistor,
        total_rounds=ROUNDS,
        fraction_fit=1.0,
        min_fit_clients=1,
        min_available_clients=1,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_params),
    )

    print(f"Iniciando Server Flower (Modelo: {MODEL_NAME}, Rondas: {ROUNDS})")
    fl.server.start_server(
        server_address="0.0.0.0:8080", 
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()