import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import numpy as np

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN Y DATOS (Tus "Valores Generados")
# -----------------------------------------------------------------------------

# AQUI va tu lógica para cargar los valores que generaste.
# Por ahora simulamos datos falsos para que el código funcione ya mismo.
def load_data():
    print("Cargando datos locales generados...")
    # Simulación: 100 muestras, 10 características cada una
    # Reemplaza 'x_train' con tus valores generados reales
    x_train = torch.randn(100, 10) 
    y_train = torch.randint(0, 2, (100,)) # Etiquetas ficticias (0 o 1)
    
    # Creamos un DataLoader simple
    train_loader = torch.utils.data.DataLoader(
        list(zip(x_train, y_train)), batch_size=32, shuffle=True
    )
    return train_loader

# Definimos un modelo simple que coincida con tus datos (Input: 10 -> Output: 2)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2) # Ajusta '10' al tamaño de tus valores

    def forward(self, x):
        return self.fc(x)

# -----------------------------------------------------------------------------
# 2. DEFINICIÓN DEL CLIENTE FLOWER
# -----------------------------------------------------------------------------

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader
        self.device = torch.device("cpu") 

    def get_parameters(self, config):
        # Extrae los pesos del modelo para enviarlos al servidor
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        # Actualiza el modelo local con los pesos que llegan del servidor
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # Paso 1: Actualizar modelo con pesos globales
        self.set_parameters(parameters)
        
        # Paso 2: Entrenar con "valores generados" locales
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.model.train()
        
        print(f"Entrenando localmente con config: {config}")
        for epoch in range(1):  # Entrenamos 1 época por ronda
            for inputs, labels in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Paso 3: Devolver pesos actualizados al servidor EC2
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        # Evaluación simple (opcional)
        self.set_parameters(parameters)
        loss = 0.0
        return float(loss), len(self.train_loader.dataset), {"accuracy": 0.5}

# -----------------------------------------------------------------------------
# 3. CONEXIÓN AL SERVIDOR (EC2)
# -----------------------------------------------------------------------------

def main():
    # 1. Cargar datos y modelo
    train_loader = load_data()
    model = SimpleModel()

    # 2. Iniciar cliente
    # IMPORTANTE: Cambia la IP por la de tu servidor en Google Cloud
    SERVER_ADDRESS = "35.193.14.101:8080" 
    
    print(f"Conectando al servidor en {SERVER_ADDRESS}...")
    
    fl.client.start_numpy_client(
        server_address=SERVER_ADDRESS,
        client=FlowerClient(model, train_loader),
    )

if __name__ == "__main__":
    main()