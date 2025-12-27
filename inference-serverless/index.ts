import * as pulumi from "@pulumi/pulumi";
import * as gcp from "@pulumi/gcp";
import * as docker from "@pulumi/docker";
import * as fs from "fs";
import * as path from "path";
import * as mime from "mime";

// Configuración
const config = new pulumi.Config();
const bucketName = "confide-models-bucket-ca9a711";
const baseModelDir = "./model_base_template";

const files = fs.readdirSync(baseModelDir);

files.forEach((fileName) => {
  const filePath = path.join(baseModelDir, fileName);
  new gcp.storage.BucketObject(`template-${fileName}`, {
    bucket: bucketName,
    name: `templates/mpnet-base/${fileName}`,
    source: new pulumi.asset.FileAsset(filePath),
    contentType: mime.getType(filePath) || "application/octet-stream",
  });
});

const repository = new gcp.artifactregistry.Repository("repo", {
  location: "us-central1",
  repositoryId: "fl-inference-repo",
  format: "DOCKER",
});

// Construir Imagen Docker (Se usa para ambos servicios)
const image = new docker.Image("cloud-run-image", {
  // SUBIMOS VERSIÓN para forzar el rebuild del app.py
  imageName: pulumi.interpolate`${repository.location}-docker.pkg.dev/${gcp.config.project}/${repository.repositoryId}/deep-model:v7`,
  build: {
    context: "./cloud_run_app",
    platform: "linux/amd64",
  },
});

// --- SERVICIO 1: BINARY (Ligero) ---
const binaryService = new gcp.cloudrunv2.Service("deep-model-binary", {
  location: "us-central1",
  template: {
    containers: [
      {
        image: image.imageName,
        resources: {
          limits: {
            cpu: "1", // Binario requiere menos CPU
            memory: "2Gi", // Solo carga 1 modelo pequeño
          },
          cpuIdle: false,
        },
        envs: [
          { name: "MODEL_BUCKET_NAME", value: bucketName },
          { name: "SERVING_MODE", value: "binary" }, // CONFIGURACIÓN CLAVE
        ],
      },
    ],
    timeout: "300s",
    maxInstanceRequestConcurrency: 20,
  },
});

new gcp.cloudrunv2.ServiceIamMember("bin-invoker", {
  location: binaryService.location,
  name: binaryService.name,
  role: "roles/run.invoker",
  member: "allUsers",
});

// --- SERVICIO 2: MULTICLASS (Pesado) ---
const multiclassService = new gcp.cloudrunv2.Service("deep-model-multiclass", {
  location: "us-central1",
  template: {
    containers: [
      {
        image: image.imageName,
        resources: {
          limits: {
            cpu: "2", // Multiclase requiere más CPU
            memory: "4Gi", // Requiere más memoria para 12 etiquetas
          },
          cpuIdle: false,
        },
        envs: [
          { name: "MODEL_BUCKET_NAME", value: bucketName },
          { name: "SERVING_MODE", value: "multiclass" }, // CONFIGURACIÓN CLAVE
        ],
      },
    ],
    timeout: "600s",
    maxInstanceRequestConcurrency: 10,
  },
});

new gcp.cloudrunv2.ServiceIamMember("multi-invoker", {
  location: multiclassService.location,
  name: multiclassService.name,
  role: "roles/run.invoker",
  member: "allUsers",
});

// --- FOG NODE (Orquestador) ---
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
    availableMemory: "512Mi",
    environmentVariables: {
      MODEL_BUCKET_NAME: bucketName,
      CLOUD_RUN_BINARY_URL: binaryService.uri,
      CLOUD_RUN_MULTI_URL: multiclassService.uri,
      FORCE_UPDATE: "v5",
    },
  },
});

new gcp.cloudrunv2.ServiceIamMember("fog-invoker", {
  location: fogFunction.location,
  name: fogFunction.name,
  role: "roles/run.invoker",
  member: "allUsers",
});

export const entryPointUrl = fogFunction.serviceConfig.apply((c) => c?.uri);
