# federated-learning-rq-project

```cmd
my-fl-project/
├── app/
│   ├── training/
│   │   ├── main.py             # Lógica Flower (Python puro)
│   │   └── startup.sh          # Script Bash (Solo instalación de dependencias)
│   └── inference/
│       ├── app.py              # API FastAPI
│       ├── Dockerfile
│       └── requirements.txt
├── infra/
│   ├── network.py              # Contexto: Red
│   ├── storage.py              # Contexto: Storage
│   ├── compute.py              # Contexto: VM (Lee startup.sh + main.py)
│   ├── serverless.py           # Contexto: Cloud Run
│   └── __main__.py             # Orquestador
├── Pulumi.yaml
```

## IaC

- **Mandar Docker a GCloud**

```bash
gcloud config set project mi-app-tofu-123456
gcloud services enable run.googleapis.com cloudbuild.googleapis.com containerregistry.googleapis.com
gcloud builds submit --tag gcr.io/mi-app-tofu-123456/fl-inference:latest app/inference
```

- **Iniciar Pulumi**

```bash
pulumi login --local
pulumi stack init dev

pulumi config set gcp:project mi-app-tofu-123456
pulumi config set gcp:region us-central1
pulumi config set app:dockerImage "gcr.io/mi-app-tofu-123456/fl-inference:latest"
pulumi config set app:modelName "microsoft/mpnet-base"
pulumi config set app:rounds "3"

python3 -m venv venv
source venv/bin/activate
pip install pulumi pulumi-gcp

pulumi up
```

IP Externa de Instancia fl-training-node-6927d8a: 35.193.14.101

IP Externa de Serveless fl-inference-api-2f04e1f: https://fl-inference-api-2f04e1f-601659189313.us-central1.run.app

## Cliente

```bash
pip install flwr torch numpy
python client.py
```
