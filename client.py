import flwr as fl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from datasets import Dataset
import random
import warnings
warnings.filterwarnings('ignore')
import sys
# ====================================================
# CONFIGURACIÓN DEL CLIENTE
# ====================================================
CLIENT_ID = 0  # Cambiar para cada cliente
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ====================================================
# FUNCIONES AUXILIARES
# ====================================================
def cargar_datos_cliente(client_id, total_clientes=2):
    """Carga PROMISE_extended6, filtra y particiona para el cliente"""
    
    # 1. Cargar el CSV
    try:
        df = pd.read_csv('PROMISE_extended6.csv')
    except FileNotFoundError:
        print("⚠️ No se encontró PROMISE_extended6.csv, creando datos dummy...")
        data = {'RequirementText': ['System must be fast'] * 100, 'class': ['NF'] * 100}
        df = pd.DataFrame(data)

    # --- AGREGAR ESTA LÍNEA (IMPORTANTE) ---
    # Mezclamos los datos aleatoriamente para que ambos clientes tengan de todo
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # ---------------------------------------

    # 2. Tu lógica de limpieza (F vs NF)
    df['class'] = df['class'].apply(lambda x: 'F' if x == 'F' else 'NF')
    
    # 3. Mapear etiquetas
    label2id = {'F': 0, 'NF': 1}
    id2label = {0: 'F', 1: 'NF'}
    df['label'] = df['class'].map(label2id)
    
    # 4. Dividir datos (Particionamiento)
    indices = np.array_split(np.arange(len(df)), total_clientes)
    idx_actual = client_id % total_clientes
    client_indices = indices[idx_actual]
    
    # Imprimir para verificar que ahora sí están mezclados
    subset = df.iloc[client_indices]
    print(f"--> Cliente {client_id}: {len(subset)} datos. Dist: {subset['class'].value_counts().to_dict()}")
    
    return subset, label2id, id2label

def preparar_datasets(df, tokenizer):
    """Prepara datasets para entrenamiento"""
    def tokenize_function(examples):
        return tokenizer(
            examples["RequirementText"],
            padding="max_length",
            truncation=True,
            max_length=32
        )
    
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    columnas_a_mantener = ["input_ids", "attention_mask", "labels"]
    columnas_actuales = tokenized_dataset.column_names
    columnas_a_eliminar = [c for c in columnas_actuales if c not in columnas_a_mantener and c != "label"]
    
    tokenized_dataset = tokenized_dataset.remove_columns(columnas_a_eliminar)
    
    if "label" in tokenized_dataset.column_names:
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    
    tokenized_dataset.set_format("torch")
    
    trainloader = DataLoader(tokenized_dataset, batch_size=32, shuffle=True)
    return trainloader

# ====================================================
# CLASE DEL CLIENTE FLOWER
# ====================================================
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        
        # Cargar datos del cliente
        self.client_df, self.label2id, self.id2label = cargar_datos_cliente(client_id)
        print(f"Cliente {client_id}: {len(self.client_df)} muestras")
        print(f"Distribución: {self.client_df['class'].value_counts().to_dict()}")
        
        # Inicializar modelo y tokenizer
        self.model_name = "microsoft/mpnet-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Preparar datasets
        self.trainloader = preparar_datasets(self.client_df, self.tokenizer)
        
        # Inicializar modelo
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"CLIENTE INICIADO EN DISPOSITIVO: {self.device} ({torch.cuda.get_device_name(0) if self.device.type == 'cuda' else 'CPU'})")
        
        # Optimizador
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        
    def get_parameters(self, config):
        """Obtener parámetros del modelo"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Establecer parámetros del modelo"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """Entrenar el modelo localmente"""
        # Establecer parámetros globales
        self.set_parameters(parameters)
        
        # Configurar entrenamiento
        self.model.train()
        epochs = config.get("epochs", 1)
        
        # Entrenamiento por épocas
        for epoch in range(epochs):
            total_loss = 0
            for batch in self.trainloader:
                # Mover batch al dispositivo
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Cliente {self.client_id}, Época {epoch+1}: Loss = {total_loss/len(self.trainloader):.4f}")
        
        # Devolver nuevos parámetros y métricas
        return self.get_parameters({}), len(self.client_df), {"loss": total_loss/len(self.trainloader)}
    
    def evaluate(self, parameters, config):
        """Evaluar el modelo localmente"""
        self.set_parameters(parameters)
        self.model.eval()
        
        losses = []
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.trainloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                
                # Calcular pérdida
                loss = outputs.loss
                losses.append(loss.item())
                
                # Calcular precisión
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == batch["labels"]).sum().item()
                total_samples += len(batch["labels"])
        
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        loss = np.mean(losses)
        
        print(f"Cliente {self.client_id}: Accuracy = {accuracy:.4f}, Loss = {loss:.4f}")
        
        return float(loss), len(self.client_df), {"accuracy": float(accuracy)}

# ====================================================
# EJECUCIÓN DEL CLIENTE
# ====================================================
def main():
    # Leer argumentos: python client.py <CLIENT_ID>
    if len(sys.argv) > 1:
        client_id = int(sys.argv[1])
    else:
        client_id = 0 # Default
    
    # IMPORTANTE: Aquí pegas la IP que te dio Pulumi (backendIp)
    # Ejemplo: "34.123.45.67"
    SERVER_PUBLIC_IP = "34.171.220.231" 

    print(f"Iniciando Cliente {client_id} conectando a {SERVER_PUBLIC_IP}...")

    client = FederatedClient(client_id)
    
    # Iniciar conexión con Flower
    fl.client.start_numpy_client(
        server_address=f"{SERVER_PUBLIC_IP}:8080",
        client=client
    )

if __name__ == "__main__":
    main()