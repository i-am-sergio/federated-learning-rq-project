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
# CONFIGURACIÓN
# ====================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ====================================================
# PROCESAMIENTO DE DATOS
# ====================================================
def cargar_datos_cliente(client_id, total_clientes=2):
    # En un escenario real Edge, cada dispositivo tiene su CSV local único
    try:
        df = pd.read_csv('promise_nfr.csv') # Asegúrate de tener este archivo
        df['class'] = df['class'].apply(lambda x: 'F' if x == 'F' else 'NF')
        label2id = {'F': 0, 'NF': 1}
        id2label = {0: 'F', 1: 'NF'}
        df['label'] = df['class'].map(label2id)
        
        # Simulación: Dividimos el dataset para simular datos distintos
        indices = np.array_split(np.arange(len(df)), total_clientes)
        client_indices = indices[client_id % total_clientes]
        return df.iloc[client_indices], label2id, id2label
    except:
        print("Error cargando CSV. Asegúrate de tener 'promise_nfr.csv'")
        sys.exit(1)

def preparar_datasets(df, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["RequirementText"], padding="max_length", truncation=True, max_length=32)
    
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Limpieza estricta de columnas para HuggingFace
    cols_to_keep = ["input_ids", "attention_mask", "labels"]
    cols_to_remove = [c for c in tokenized_dataset.column_names if c not in cols_to_keep and c != "label"]
    tokenized_dataset = tokenized_dataset.remove_columns(cols_to_remove)
    if "label" in tokenized_dataset.column_names:
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        
    tokenized_dataset.set_format("torch")
    return DataLoader(tokenized_dataset, batch_size=32, shuffle=True)

# ====================================================
# LÓGICA DEL CLIENTE
# ====================================================
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        print(f"Inicializando Cliente {client_id}...")
        
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
        epochs = 1 
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in self.trainloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss/len(self.trainloader)
            print(f"Edge {self.client_id} | Epoch {epoch+1} | Loss: {avg_loss:.4f}")
        
        return self.get_parameters({}), len(self.client_df), {"loss": avg_loss}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.trainloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total += len(batch["labels"])
        accuracy = correct / total
        return 0.0, len(self.client_df), {"accuracy": accuracy}

# ====================================================
# MAIN
# ====================================================
def main():
    # ID del cliente local
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    
    FOG_NODE_IP = "136.115.164.74" 
    
    print(f"Conectando Edge Client {client_id} a FOG NODE: {FOG_NODE_IP}:8081")
    
    fl.client.start_numpy_client(
        server_address=f"{FOG_NODE_IP}:8081", # Puerto 8081 para Edge
        client=FederatedClient(client_id),
        grpc_max_message_length=1024*1024*1024
    )

if __name__ == "__main__":
    main()