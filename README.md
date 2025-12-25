# Federated Learning and Computation Offloading for Requirements Classification

## Terminal 1: Deploy Infrastructure
- Deploy Infrastructure with Pulumi
```sh
cd infrastructure
pulumi up
```
`Note:` Copy Public IP of the created VM

- List VMs to get their names
```sh
gcloud compute instances list
```

## Terminal 2: Cloud Server (VM on Cloud)

- Connect to the VM with ssh
```sh
gcloud compute ssh <CLOUD_VM_NAME> --zone us-central1-a
```

## Terminal 3: Fog Server (VM on Edge)

- Connect to the VM with ssh
```sh
gcloud compute ssh <FOG_VM_NAME> --zone us-central1-b
```



<!-- # federated-learning-rq-project

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

- Start prediction server in background process
```sh
nohup python predict_server.py > predict_server.log 2>&1 & # Iniciar
ps aux | grep predict_server.py # Verificar que esté corriendo
pgrep -f predict_server.py # Obtener el PID
kill <PID> # Detener el proceso
```

- Test API
```sh
curl -X POST http://34.9.5.148:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "The system shall be modular to facilitate easy updates and maintenance."}'
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
``` -->