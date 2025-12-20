# federated-learning-rq-project

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


