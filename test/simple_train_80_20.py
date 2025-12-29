import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    AutoConfig
)
from datasets import Dataset
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    precision_recall_fscore_support,
    balanced_accuracy_score
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
os.environ["WANDB_DISABLED"] = "true"

# Fijar semilla global
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ====================================================
# 1. Cargar dataset y mapear a binario (F vs NF)
# ====================================================
df = pd.read_csv('PROMISE_extended6.csv')
df['class'] = df['class'].apply(lambda x: 'F' if x == 'F' else 'NF')

print("Class distribution:")
print(df['class'].value_counts())

# Mapear clases a IDs
label2id = {label: idx for idx, label in enumerate(df['class'].unique())}
id2label = {idx: label for label, idx in label2id.items()}
df['label'] = df['class'].map(label2id)

# ====================================================
# 2. Tokenizer
# ====================================================
model_name = "microsoft/mpnet-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples["RequirementText"],
        padding="max_length",
        truncation=True,
        max_length=64
    )

# ====================================================
# 3. Métricas
# ====================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average='binary'),
        "precision": precision_score(labels, predictions, average='binary'),
        "recall": recall_score(labels, predictions, average='binary')
    }

# ====================================================
# 4. Split Train/Test (80/20)
# ====================================================
# Usamos stratify para mantener la misma proporción de clases que en el original
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df['label'],
    random_state=42
)

print(f"\nTraining samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# Crear Datasets de HuggingFace
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenizar
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# ====================================================
# 5. Configuración del Modelo
# ====================================================
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# Congelar capas iniciales (opcional)
layers = model.base_model.encoder.layer
for layer in layers[:6]:  # congela primeras 6
    for param in layer.parameters():
        param.requires_grad = False

# ====================================================
# 6. Args de entrenamiento
# ====================================================
training_args = TrainingArguments(
    output_dir="./results2class/single_run", # Carpeta única
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs/single_run",
    report_to="none",
    save_total_limit=1
)

# ====================================================
# 7. Trainer
# ====================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test, # Usamos el set de test como validación
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Entrenar
print("\nIniciando entrenamiento...")
trainer.train()

# ====================================================
# 8. Evaluación y Resultados Finales
# ====================================================
print("\nEvaluando modelo final...")
eval_results = trainer.evaluate()
print("Eval results (dict):", eval_results)

# Predicciones sobre el set de test
preds = trainer.predict(tokenized_test)
pred_labels = np.argmax(preds.predictions, axis=1)
true_labels = test_df['label'].values

# Reporte de clasificación
print("\n========== Classification Report ==========")
print(classification_report(
    true_labels,
    pred_labels,
    target_names=[id2label[i] for i in range(len(id2label))]
))

# Métricas globales
acc = accuracy_score(true_labels, pred_labels)
prec = precision_score(true_labels, pred_labels, average='binary')
rec = recall_score(true_labels, pred_labels, average='binary')
f1 = f1_score(true_labels, pred_labels, average='binary')

print("\n========== Métricas Globales ==========")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")

# Matriz de Confusión
print("\n========== Matriz de Confusión ==========")
cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(label2id))))
df_cm = pd.DataFrame(
    cm,
    index=[f"True_{id2label[i]}" for i in range(len(id2label))],
    columns=[f"Pred_{id2label[i]}" for i in range(len(id2label))]
)
print(df_cm)