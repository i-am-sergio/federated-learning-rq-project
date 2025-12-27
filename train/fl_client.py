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
DATASET_FILE = 'promise_nfr.csv'
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ====================================================
# FUNCIONES AUXILIARES
# ====================================================
def cargar_datos_cliente(client_id, total_clientes=5, multiclass=False):
    """
    Carga y divide datos. Se adapta a Binario o Multiclase automáticamente.
    """
    try:
        df = pd.read_csv(DATASET_FILE)
    except FileNotFoundError:
        print(f"Error: No se encuentra {DATASET_FILE}")
        sys.exit(1)


    if multiclass:
        # --- LÓGICA MULTICLASE ---
        # Usamos todas las clases originales
        unique_labels = sorted(df['class'].unique())
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        id2label = {idx: label for label, idx in label2id.items()}
        
        # Mapeamos la columna 'class' a números
        df['label'] = df['class'].map(label2id)
        print(f"Modo Multiclase activado. {len(unique_labels)} clases encontradas: {unique_labels}")
    
    else:
        # --- LÓGICA BINARIA (F vs NF) ---
        df['temp_class'] = df['class'].apply(lambda x: 'F' if x == 'F' else 'NF')
        label2id = {'F': 0, 'NF': 1}
        id2label = {0: 'F', 1: 'NF'}
        df['label'] = df['temp_class'].map(label2id)
        print("Modo Binario activado (F vs NF).")

    indices = np.array_split(np.arange(len(df)), total_clientes)
    client_indices = indices[client_id % total_clientes]
    client_df = df.iloc[client_indices]
    return client_df, label2id, id2label

def preparar_datasets(df, tokenizer):
    def tokenize_function(examples):
        return tokenizer(
            examples["RequirementText"],
            padding="max_length",
            truncation=True,
            max_length=64 # Aumenté un poco a 64 para capturar más contexto
        )
    
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    columnas_a_mantener = ["input_ids", "attention_mask", "labels"]
    # Limpieza de columnas extrañas
    columnas_actuales = tokenized_dataset.column_names
    columnas_a_eliminar = [c for c in columnas_actuales if c not in columnas_a_mantener and c != "label"]
    tokenized_dataset = tokenized_dataset.remove_columns(columnas_a_eliminar)
    
    # Renombrar 'label' a 'labels' (lo que espera HuggingFace)
    if "label" in tokenized_dataset.column_names:
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    
    tokenized_dataset.set_format("torch")
    trainloader = DataLoader(tokenized_dataset, batch_size=32, shuffle=True)
    return trainloader

# ====================================================
# CLASE DEL CLIENTE FLOWER
# ====================================================
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, client_id, multiclass=False):
        self.client_id = client_id
        self.multiclass = multiclass
        self.client_df, self.label2id, self.id2label = cargar_datos_cliente(client_id, multiclass=multiclass)
        self.num_labels = len(self.label2id)
        
        # Cargar datos del cliente
        print(f"Cliente {client_id}: {len(self.client_df)} muestras cargadas.")
        
        # Inicializar modelo y tokenizer
        self.model_name = "microsoft/mpnet-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.trainloader = preparar_datasets(self.client_df, self.tokenizer)
        
        # Inicializar modelo
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        # Configurar dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
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
        self.set_parameters(parameters)
        self.model.train()
        epochs = config.get("epochs", 1)
        
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
            avg_loss = total_loss / len(self.trainloader)
            
            mode_str = 'Multi' if self.multiclass else 'Bin'
            print(f"Cliente {self.client_id} [Mode: {mode_str}], Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        return self.get_parameters({}), len(self.client_df), {"loss": avg_loss}
    
    def evaluate(self, parameters, config):
        """Evaluar el modelo localmente"""
        self.set_parameters(parameters)
        self.model.eval()
        
        losses = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.trainloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                
                # Calcular pérdida
                loss = outputs.loss
                losses.append(loss.item())
                
                # Calcular precisión
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total += len(batch["labels"])
        
        accuracy = correct / total if total > 0 else 0
        loss_avg = np.mean(losses)
        
        print(f"Cliente {self.client_id}: Accuracy = {accuracy:.4f}, Loss = {loss_avg:.4f}")
        return float(loss_avg), len(self.client_df), {"accuracy": float(accuracy)}

# ====================================================
# EJECUCIÓN DEL CLIENTE
# ====================================================
def main():
    # Obtener ID del cliente desde argumentos o variable de entorno
    # Argumento 1: Client ID (Por defecto 0)
    if len(sys.argv) > 1:
        client_id = int(sys.argv[1])
    else:
        client_id = 0
        
    # Argumento 2: Modo (1=Multiclase, 0=Binario)
    usar_multiclase = False
    if len(sys.argv) > 2 and sys.argv[2] == "1":
        usar_multiclase = True
    
    SERVER_PUBLIC_IP = "34.121.92.169"
    print(f"Iniciando Cliente {client_id} en modo {'MULTICLASE' if usar_multiclase else 'BINARIO'}")

    # Iniciar cliente
    client = FederatedClient(client_id, multiclass=usar_multiclase)
    
    fl.client.start_client(
        server_address=f"{SERVER_PUBLIC_IP}:8080",
        client=client
    )

if __name__ == "__main__":
    main()