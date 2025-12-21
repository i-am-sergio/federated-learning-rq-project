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
```

- **Iniciar Pulumi**

```bash
gcloud config set project mi-app-tofu-123456
pulumi login --local
pulumi stack init dev

pulumi config set gcp:project mi-app-tofu-123456
pulumi config set gcp:region us-central1
npm install
pulumi install
gcloud auth configure-docker
gcloud services enable run.googleapis.com cloudbuild.googleapis.com containerregistry.googleapis.com
pulumi up
```

IP Externa de Instancia fl-training-node-6927d8a: 35.193.14.101

IP Externa de Serveless fl-inference-api-2f04e1f: https://fl-inference-api-2f04e1f-601659189313.us-central1.run.app

## Cliente

```bash
pip install flwr torch numpy
python client.py
```

## Terminal 1: Server (VM on Cloud)

- Deploy Infrastructure with Pulumi

```sh
cd infrastructure
pulumi up
```

`Note:` Copy Public IP of the created VM

- Connect to the VM with ssh

```sh
gcloud compute ssh smogollon@ml-server-b010e71 --zone us-central1-a
```

- Install dependencies

```sh
sudo apt update
sudo apt install python3-venv -y
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install flwr==1.5.0
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets pandas numpy scikit-learn
```

- Start the server

```sh
cd train
python server.py
```

## Terminal 2: Client (Edge Device)

- Install dependencies

```sh
python3 -m venv venv
source venv/bin/activate
pip install accelerate
pip install flwr==1.5.0 transformers datasets torch pandas numpy scikit-learn
```

- Change the server IP address in `client.py`

```python
SERVER_IP = "<PASTE_SERVER_PUBLIC_IP_HERE>"
```

- Start the client

```sh
cd train
python client.py
```

## Terminal 3: Download Model Trained

- After training is complete

```sh
gcloud compute scp smogollon@ml-server-b010e71:~/mpnet_fed_requirements.pth ./ --zone us-central1-a
```

```bash
# 1. Crear el entorno llamado 'fl-gpu' con Python 3.10
conda create -n fl-gpu python=3.10 -y

# 2. Activar el entorno
conda activate fl-gpu

# 3. Instalar PyTorch con soporte para CUDA 12.4
# (Tu driver 12.8 es compatible hacia atrás con 12.4, que es la versión estable de PyTorch)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Instalar Flower (versión compatible con tu servidor), Transformers y Data
pip install flwr==1.5.0 transformers datasets pandas numpy scikit-learn accelerate

python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

```sh
cat /app/server.log
cat /app/api.log
```
