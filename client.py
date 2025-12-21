import flwr as fl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from datasets import Dataset
import random
import warnings
import sys
warnings.filterwarnings('ignore')

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
    try:
        df = pd.read_csv('PROMISE_extended6.csv')
    except FileNotFoundError:
        print("\nALERTA: No se encontró 'PROMISE_extended6.csv'.")
        print("Usando DATOS DUMMY (Solo para prueba de conexión).\n")
        data = {'RequirementText': ['System must be fast and secure'] * 100, 'class': ['NF'] * 100}
        df = pd.DataFrame(data)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df['class'] = df['class'].apply(lambda x: 'F' if x == 'F' else 'NF')
    label2id = {'F': 0, 'NF': 1}
    id2label = {0: 'F', 1: 'NF'}
    df['labels'] = df['class'].map(label2id)
    indices = np.array_split(np.arange(len(df)), total_clientes)
    idx_actual = client_id % total_clientes
    client_indices = indices[idx_actual]
    
    subset = df.iloc[client_indices]
    print(f"--> Cliente {client_id}: {len(subset)} datos. Dist: {subset['class'].value_counts().to_dict()}")
    return subset, label2id, id2label

def preparar_datasets(df, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["RequirementText"], padding="max_length", truncation=True, max_length=32)
    
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns([c for c in tokenized_dataset.column_names if c not in ["input_ids", "attention_mask", "labels"]])
    tokenized_dataset.set_format("torch")
    return DataLoader(tokenized_dataset, batch_size=32, shuffle=True)

# ====================================================
# CLASE DEL CLIENTE FLOWER
# ====================================================
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.client_df, self.label2id, self.id2label = cargar_datos_cliente(client_id)
        
        self.model_name = "microsoft/mpnet-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.trainloader = preparar_datasets(self.client_df, self.tokenizer)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2, id2label=self.id2label, label2id=self.label2id
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        epochs = config.get("epochs", 1)
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch in self.trainloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"[Cliente {self.client_id}] Entrenamiento - Epoch {epoch+1}/{epochs} | Loss Promedio: {avg_loss:.4f}")

        return self.get_parameters({}), len(self.client_df), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, total = 0, 0
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.trainloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                
                # Calcular loss de validación
                loss = outputs.loss
                total_loss += loss.item()
                num_batches += 1

                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total += len(batch["labels"])
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        print(f"[Cliente {self.client_id}] Evaluación - Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2%}")
        
        return float(avg_loss), len(self.client_df), {"accuracy": accuracy}

# ====================================================
# EJECUCIÓN DEL CLIENTE
# ====================================================
def main():
    if len(sys.argv) > 1:
        client_id = int(sys.argv[1])
    else:
        client_id = 0
    
    SERVER_PUBLIC_IP = "IP EXTERNA VM"

    print(f"Iniciando Cliente {client_id} conectando a {SERVER_PUBLIC_IP}...")

    fl.client.start_numpy_client(
        server_address=f"{SERVER_PUBLIC_IP}:8080",
        client=FederatedClient(client_id),
        grpc_max_message_length=1024*1024*1024 
    )

if __name__ == "__main__":
    main()