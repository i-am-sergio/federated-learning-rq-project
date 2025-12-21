import * as gcp from "@pulumi/gcp";
import * as docker from "@pulumi/docker"; // <--- NUEVO IMPORT
import * as pulumi from "@pulumi/pulumi";
import * as fs from "fs";
import * as path from "path";

// 1. LEER ARCHIVOS DE CÓDIGO
const serverCodePath = path.resolve(__dirname, "../train/server.py");
const inferenceCodePath = path.resolve(__dirname, "../train/inference_api.py");
const startupScriptPath = path.resolve(__dirname, "startup.sh");

const serverCode = fs.readFileSync(serverCodePath, "utf-8");
const inferenceCode = fs.readFileSync(inferenceCodePath, "utf-8");
const rawStartupScript = fs.readFileSync(startupScriptPath, "utf-8");

// 2. PREPARAR STARTUP SCRIPT
let finalStartupScript = rawStartupScript.replace(
  "{{SERVER_CODE}}",
  serverCode
);
finalStartupScript = finalStartupScript.replace(
  "{{INFERENCE_CODE}}",
  inferenceCode
);

// --- INFRAESTRUCTURA DE RED ---

const vpc = new gcp.compute.Network("vpc-network", {
  autoCreateSubnetworks: false,
});

const subnet = new gcp.compute.Subnetwork("vpc-subnet", {
  ipCidrRange: "10.0.1.0/24",
  region: "us-central1",
  network: vpc.id,
});

const firewall = new gcp.compute.Firewall("allow-ports", {
  network: vpc.id,
  allows: [
    {
      protocol: "tcp",
      ports: ["22", "8080", "8000"],
    },
  ],
  sourceRanges: ["0.0.0.0/0"],
  targetTags: ["server-node"],
});

// --- MAQUINA VIRTUAL (BACKEND) ---

const vmInstance = new gcp.compute.Instance("ml-server", {
  machineType: "e2-standard-4",
  zone: "us-central1-a",
  tags: ["server-node"],
  bootDisk: {
    initializeParams: {
      image: "debian-cloud/debian-11",
      size: 50,
    },
  },
  networkInterfaces: [
    {
      network: vpc.id,
      subnetwork: subnet.id,
      accessConfigs: [{}],
    },
  ],
  metadataStartupScript: finalStartupScript,
  serviceAccount: {
    scopes: ["https://www.googleapis.com/auth/cloud-platform"], // Permisos para logs y storage
  },
});

const vmPublicIp = vmInstance.networkInterfaces.apply(
  (ni) => ni[0].accessConfigs![0].natIp
);
export const backendIp = vmPublicIp;

// --- CLOUD RUN (FRONTEND) ---

// 1. Construir y Subir la Imagen Docker (Usando @pulumi/docker)
// Esto requiere que tengas Docker corriendo en tu PC local
const frontendImage = new docker.Image("frontend-image", {
  imageName: pulumi.interpolate`gcr.io/${gcp.config.project}/frontend-chat:v1`,
  build: {
    context: "../frontend", // Ruta a tu carpeta frontend
  },
});

// 2. Servicio de Cloud Run
const runService = new gcp.cloudrun.Service("frontend-chat", {
  location: "us-central1",
  template: {
    spec: {
      containers: [
        {
          image: frontendImage.imageName, // Usamos la imagen construida arriba
          envs: [
            {
              name: "VM_API_URL",
              value: pulumi.interpolate`http://${vmPublicIp}:8000`,
            },
          ],
          ports: [
            {
              containerPort: 8501,
            },
          ],
        },
      ],
    },
  },
  traffics: [
    {
      percent: 100,
      latestRevision: true,
    },
  ],
});

// 3. Hacerlo Público
const iam = new gcp.cloudrun.IamMember("frontend-public", {
  service: runService.name,
  location: runService.location,
  role: "roles/run.invoker",
  member: "allUsers",
});

export const frontendUrl = runService.statuses[0].url;
