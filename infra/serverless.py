import pulumi
from pulumi_gcp import cloudrun, storage, organizations

class InferenceService:
    def __init__(self, name: str, image_url: str, bucket_name: pulumi.Output):
        self.name = name
        self.image_url = image_url
        self.bucket_name = bucket_name 

    def deploy(self) -> cloudrun.Service:
        config = pulumi.Config("gcp")
        project_id = config.require("project")
        
        # 1. Obtenemos metadata del proyecto
        project_data = organizations.get_project(project_id=project_id)
        default_sa = pulumi.Output.concat(project_data.number, "-compute@developer.gserviceaccount.com")

        # 2. Definición del Servicio Cloud Run
        service = cloudrun.Service(self.name,
            location="us-central1",
            template=cloudrun.ServiceTemplateArgs(
                spec=cloudrun.ServiceTemplateSpecArgs(
                    timeout_seconds=300, # Timeout general de la petición del usuario
                    service_account_name=default_sa, 
                    containers=[cloudrun.ServiceTemplateSpecContainerArgs(
                        image=self.image_url,
                        
                        # Variables de entorno
                        envs=[
                            cloudrun.ServiceTemplateSpecContainerEnvArgs(
                                name="BUCKET_NAME",
                                value=self.bucket_name
                            ),
                            cloudrun.ServiceTemplateSpecContainerEnvArgs(
                                name="HF_HOME",
                                value="/tmp"
                            ),
                            cloudrun.ServiceTemplateSpecContainerEnvArgs(
                                name="PYTHONUNBUFFERED",
                                value="1"
                            )
                        ],
                        
                        # Recursos
                        resources=cloudrun.ServiceTemplateSpecContainerResourcesArgs(
                            limits={
                                "memory": "4Gi",
                                "cpu": "2000m"
                            }
                        ),
                        
                        # Puertos
                        ports=[cloudrun.ServiceTemplateSpecContainerPortArgs(
                            container_port=8080
                        )],

                        # --- CORRECCIÓN STARTUP PROBE ---
                        startup_probe=cloudrun.ServiceTemplateSpecContainerStartupProbeArgs(
                            initial_delay_seconds=10, 
                            timeout_seconds=5,        # <--- CORREGIDO: Debe ser menor que period_seconds (10)
                            period_seconds=10,        # Chequear cada 10 segundos
                            failure_threshold=24,     # 24 intentos * 10s = 240s (4 minutos total)
                            http_get=cloudrun.ServiceTemplateSpecContainerStartupProbeHttpGetArgs(
                                path="/health",       
                                port=8080
                            )
                        )
                        # --------------------------------
                    )]
                )
            ),
            traffics=[cloudrun.ServiceTrafficArgs(
                percent=100,
                latest_revision=True
            )]
        )
        
        # 3. Permisos
        storage.BucketIAMMember(f"{self.name}-storage-access",
            bucket=self.bucket_name,
            role="roles/storage.objectViewer",
            member=pulumi.Output.concat("serviceAccount:", default_sa)
        )
        
        cloudrun.IamMember(f"{self.name}-public",
            service=service.name,
            location=service.location,
            role="roles/run.invoker",
            member="allUsers"
        )
        
        return service