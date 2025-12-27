import pandas as pd
import pickle
import os
import sys
from google.cloud import storage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Configuración
BUCKET_NAME = os.environ.get("MODEL_BUCKET_NAME", "confide-models-bucket-ca9a711")

def upload_to_gcs(bucket_name, source_file, destination_blob):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob)
        blob.upload_from_filename(source_file)
        print(f"EXITO: {source_file} subido a gs://{bucket_name}/{destination_blob}")
    except Exception as e:
        print(f"ERROR: No se pudo subir a GCS: {e}")

def main():
    # 1. Configuración de Argumentos (1 = Multiclase, 0 = Binario)
    usar_multiclase = False
    if len(sys.argv) > 1 and sys.argv[1] == "1":
        usar_multiclase = True

    if usar_multiclase:
        LOCAL_FILENAME = "multiclass.pkl"
        print("MODO: MULTICLASE (Guardando en multiclass.pkl)")
    else:
        LOCAL_FILENAME = "fast_model.pkl"
        print("MODO: BINARIO (Guardando en fast_model.pkl)")

    # 2. Cargando datos
    print("Cargando datos...")
    try:
        df = pd.read_csv('promise_nfr.csv')
    except:
        # Fallback por si acaso usas el otro nombre de archivo
        try:
            df = pd.read_csv('PROMISE_extended6.csv')
        except:
            print("Error: No encuentro 'promise_nfr.csv' ni 'PROMISE_extended6.csv'")
            return
    df = df.dropna(subset=['RequirementText', 'class'])
    X = df['RequirementText']

    # 3. Preparando Etiquetas
    if usar_multiclase:
        # Usamos las etiquetas originales de texto
        y = df['class']
        print(f"Clases encontradas: {y.unique()}")
    else:
        # Lógica Binaria: 0 = F, 1 = NF
        y = df['class'].apply(lambda x: 0 if x == 'F' else 1)

    # 4. Entrenando FastModel
    print("Entrenando Modelo (TF-IDF + LR)...")
    # Aumentamos max_iter para asegurar convergencia en multiclase
    # model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
    # class_weight='balanced': Equilibra automáticamente las clases
    # ngram_range=(1, 2): Aprende palabras sueltas Y pares de palabras (ej: "response time")
    model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)), 
    LogisticRegression(max_iter=1000, class_weight='balanced')
)
    model.fit(X, y)

    # 5. Guardando localmente
    print(f"Guardando {LOCAL_FILENAME}...")
    with open(LOCAL_FILENAME, 'wb') as f:
        pickle.dump(model, f)

    # 6. Subiendo al Bucket
    print("Subiendo al Bucket...")
    upload_to_gcs(BUCKET_NAME, LOCAL_FILENAME, LOCAL_FILENAME)

if __name__ == "__main__":
    main()