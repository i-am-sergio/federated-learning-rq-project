
import * as pulumi from "@pulumi/pulumi";
import * as gcp from "@pulumi/gcp";
import * as docker from "@pulumi/docker"; // Necesitas @pulumi/docker

// Configuración
const config = new pulumi.Config();
const bucketName = "confide-models-bucket-b1a9b2e"; 

// 1. Artifact Registry (Para guardar la imagen Docker del Cloud Run)
const repository = new gcp.artifactregistry.Repository("repo", {
    location: "us-central1",
    repositoryId: "fl-inference-repo",
    format: "DOCKER",
});

// 2. Construir y Subir Imagen Docker (Modelo Pesado)
const image = new docker.Image("cloud-run-image", {
    imageName: pulumi.interpolate`${repository.location}-docker.pkg.dev/${gcp.config.project}/${repository.repositoryId}/deep-model:v1`,
    build: {
        context: "./cloud_run_app",
        platform: "linux/amd64", // Importante para Cloud Run
    },
});

// 3. Desplegar Cloud Run (Servicio Pesado)
const cloudRunService = new gcp.cloudrunv2.Service("deep-model-service", {
    location: "us-central1",
    template: {
        containers: [{
            image: image.imageName,
            resources: {
                limits: {
                    cpu: "2",      // 1 vCPU
                    memory: "4Gi", // 2GB RAM para MPNet
                },
                cpuIdle: false, // esto es para no facturar CPU inactiva
            },
            envs: [
                { name: "MODEL_BUCKET_NAME", value: bucketName },
            ],
        }],
        // Aumentar timeout de arranque por si la descarga es lenta
        timeout: "600s", 
        maxInstanceRequestConcurrency: 10, // Puede atender 10 peticiones a la vez
    },
});

// Hacer público el Cloud Run (o restríngelo solo a la Cloud Function si prefieres)
const runIam = new gcp.cloudrunv2.ServiceIamMember("run-invoker", {
    location: cloudRunService.location,
    name: cloudRunService.name,
    role: "roles/run.invoker",
    member: "allUsers",
});

// 4. Desplegar Cloud Function (Fog - Modelo Ligero)
// Bucket para subir el código fuente de la función
const sourceBucket = new gcp.storage.Bucket("fn-source-bucket", {
    location: "US",
    uniformBucketLevelAccess: true,
});

const fogArchive = new gcp.storage.BucketObject("fog-zip", {
    bucket: sourceBucket.name,
    source: new pulumi.asset.AssetArchive({
        ".": new pulumi.asset.FileArchive("./fog_function"),
    }),
});

const fogFunction = new gcp.cloudfunctionsv2.Function("fog-node-fn", {
    location: "us-central1",
    buildConfig: {
        runtime: "python310",
        entryPoint: "fog_predict",
        source: {
            storageSource: {
                bucket: sourceBucket.name,
                object: fogArchive.name,
            },
        },
    },
    serviceConfig: {
        maxInstanceCount: 10,
        availableMemory: "256Mi", // Muy poca memoria (barato)
        environmentVariables: {
            MODEL_BUCKET_NAME: bucketName,
            // Inyectamos la URL del Cloud Run dinámicamente
            CLOUD_RUN_URL: cloudRunService.uri,
        },
    },
});

// Hacer pública la función Fog (Tu punto de entrada)
const fogIam = new gcp.cloudfunctionsv2.FunctionIamMember("fog-invoker", {
    location: fogFunction.location,
    cloudFunction: fogFunction.name,
    role: "roles/cloudfunctions.invoker",
    member: "allUsers",
});

// Permisos para leer el Bucket de Modelos
const project = pulumi.output(gcp.organizations.getProject({}));
const defaultComputeSa = project.apply(p => `${p.number}-compute@developer.gserviceaccount.com`);
const bucketReader = new gcp.storage.BucketIAMMember("sa-model-reader", {
    bucket: bucketName,
    role: "roles/storage.objectViewer",
    member: defaultComputeSa.apply(email => `serviceAccount:${email}`),
});

// Exportar la URL que usarás en tu React App
export const entryPointUrl = fogFunction.serviceConfig.apply(c => c?.uri);