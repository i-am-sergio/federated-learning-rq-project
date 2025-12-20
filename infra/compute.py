import pulumi
from pulumi_gcp import compute
import os

class TrainingServer:
    """Bounded Context: Compute."""

    def __init__(self, name: str, ip_address: compute.Address, bucket_name: pulumi.Output, config: pulumi.Config):
        self.name = name
        self.ip = ip_address
        self.bucket_name = bucket_name # Esto es un Output, no un string
        self.model_name = config.get("modelName") or "microsoft/mpnet-base"
        self.rounds = config.get("rounds") or "3"

    def _read_file(self, relative_path: str) -> str:
        path = os.path.join(os.path.dirname(__file__), relative_path)
        with open(path, 'r') as f:
            return f.read()

    def _construct_metadata_script(self):
        # 1. Leemos los archivos de texto (esto es seguro porque son strings locales)
        startup_script_content = self._read_file('../app/training/startup.sh')
        python_app_content = self._read_file('../app/training/main.py')
        
        # 2. LA SOLUCIÓN: Usamos .apply() sobre el bucket_name
        # Pulumi pasará el valor real del nombre del bucket a la variable 'b_name' dentro del lambda
        return self.bucket_name.apply(lambda b_name: f"""
{startup_script_content}

# --- Inyección de Código de Aplicación ---
cd /app
source venv/bin/activate

echo "Escribiendo main.py..."
cat << 'EOF' > main.py
{python_app_content}
EOF

# --- Ejecución ---
echo "Iniciando Servidor FL..."
# Aquí usamos b_name (el valor resuelto) en lugar de self.bucket_name
export BUCKET_NAME="{b_name}"
export MODEL_NAME="{self.model_name}"
export ROUNDS={self.rounds}

# Ejecutar en segundo plano
nohup python main.py > server.log 2>&1 &
        """)

    def deploy(self) -> compute.Instance:
        return compute.Instance(self.name,
            machine_type="e2-standard-4",
            zone="us-central1-a",
            tags=["fl-server"],
            boot_disk=compute.InstanceBootDiskArgs(
                initialize_params=compute.InstanceBootDiskInitializeParamsArgs(
                    image="ubuntu-os-cloud/ubuntu-2204-lts",
                    size=50,
                ),
            ),
            network_interfaces=[compute.InstanceNetworkInterfaceArgs(
                network="default",
                access_configs=[compute.InstanceNetworkInterfaceAccessConfigArgs(
                    nat_ip=self.ip.address
                )],
            )],
            # Pulumi acepta Outputs aquí automáticamente
            metadata_startup_script=self._construct_metadata_script(),
            service_account=compute.InstanceServiceAccountArgs(
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            ),
        )