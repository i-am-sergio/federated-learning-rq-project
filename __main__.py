import pulumi
from infra.network import NetworkManager
from infra.storage import ModelRepository
from infra.compute import TrainingServer
from infra.serverless import InferenceService

# 1. Configuración
config = pulumi.Config()
app_config = pulumi.Config("app") # Namespace 'app'
gcp_project = pulumi.Config("gcp").require("project")
docker_image = app_config.require("dockerImage")

# 2. Inyección de Dependencias (Instanciación)
network_mgr = NetworkManager("fl-infra")
storage_repo = ModelRepository("fl-models-prod")

# 3. Recursos Compartidos
static_ip = network_mgr.create_static_ip()
firewall = network_mgr.create_firewall()
bucket = storage_repo.create_bucket()

# 4. Training (Instancia)
# Le pasamos la config para que extraiga el modelo y rondas
trainer = TrainingServer(
    name="fl-training-node", 
    ip_address=static_ip, 
    bucket_name=bucket.name, 
    config=app_config
)
vm = trainer.deploy()

# 5. Inference (Serverless)
inference = InferenceService(
    name="fl-inference-api", 
    image_url=docker_image, 
    bucket_name=bucket.name
)
api_service = inference.deploy()

# 6. Exportar Resultados
pulumi.export("FL_Server_IP", static_ip.address)
pulumi.export("Inference_Endpoint", api_service.statuses[0].url)
pulumi.export("Bucket_Name", bucket.name)