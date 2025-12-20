import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
import torch
from sklearn.metrics import classification_report, confusion_matrix
import random
import os
os.environ["WANDB_DISABLED"] = "true"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def cargar_datos():
    # df = pd.read_csv('../PROMISE_extended6.csv')
    df = pd.read_csv('promise_nfr.csv')
    df['class'] = df['class'].apply(lambda x: 'F' if x == 'F' else 'NF')
    return df

def preparar_etiquetas(df):
    label2id = {label: idx for idx, label in enumerate(df['class'].unique())}
    id2label = {idx: label for label, idx in label2id.items()}
    df['label'] = df['class'].map(label2id)
    return df, label2id, id2label

def preparar_tokenizer(model_name="microsoft/mpnet-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def tokenizar_datos(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(
            examples["RequirementText"],
            padding="max_length",
            truncation=True,
            max_length=32
        )
    return dataset.map(tokenize_function, batched=True)

def crear_modelo(model_name, label2id, id2label):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    return model

def congelar_capas(model, num_capas=6):
    layers = model.base_model.encoder.layer
    for layer in layers[:num_capas]:
        for param in layer.parameters():
            param.requires_grad = False
    return model

def calcular_metricas(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average='binary'),
        "precision": precision_score(labels, predictions, average='binary'),
        "recall": recall_score(labels, predictions, average='binary')
    }

def entrenar_modelo(trainer):
    print("Iniciando entrenamiento...")
    trainer.train()
    return trainer

def evaluar_modelo(trainer, tokenized_test, test_df, id2label):
    print("\nEvaluando modelo final...")
    eval_results = trainer.evaluate()
    print("Resultados evaluación:", eval_results)

    preds = trainer.predict(tokenized_test)
    pred_labels = np.argmax(preds.predictions, axis=1)
    true_labels = test_df['label'].values

    print("\nReporte de clasificación:")
    print(classification_report(
        true_labels,
        pred_labels,
        target_names=[id2label[i] for i in range(len(id2label))]
    ))

    cm = confusion_matrix(true_labels, pred_labels)
    print("\nMatriz de confusión:")
    print(cm)

def main():
    df = cargar_datos()
    df, label2id, id2label = preparar_etiquetas(df)
    
    tokenizer = preparar_tokenizer()
    
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df['label'],
        random_state=42
    )
    
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    tokenized_train = tokenizar_datos(train_dataset, tokenizer)
    tokenized_test = tokenizar_datos(test_dataset, tokenizer)
    
    model = crear_modelo("microsoft/mpnet-base", label2id, id2label)
    model = congelar_capas(model, num_capas=6)
    
    training_args = TrainingArguments(
        output_dir="./results2class/single_run",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="./logs/single_run",
        report_to="none",
        save_total_limit=1
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=calcular_metricas,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    entrenar_modelo(trainer)
    evaluar_modelo(trainer, tokenized_test, test_df, id2label)

if __name__ == "__main__":
    main()