import * as pulumi from "@pulumi/pulumi";
import * as gcp from "@pulumi/gcp";

// ====================================================
// 1. RED Y SEGURIDAD
// ====================================================
const vpc = new gcp.compute.Network("vpc-network", {
    autoCreateSubnetworks: false,
});

const subnet = new gcp.compute.Subnetwork("vpc-subnet", {
    ipCidrRange: "10.0.1.0/24",
    region: "us-central1",
    network: vpc.id,
});

const firewall = new gcp.compute.Firewall("allow-flower-ecosystem", {
    network: vpc.id,
    allows: [{
        protocol: "tcp",
        ports: ["22", "8080", "8081", "5000"], 
    }],
    sourceRanges: ["0.0.0.0/0"],
    targetTags: ["cloud-node", "fog-node"], 
});

// ====================================================
// 2. ALMACENAMIENTO (MODEL REGISTRY)
// ====================================================
// Bucket para guardar los modelos (FastModel y DeepModel)
const modelsBucket = new gcp.storage.Bucket("confide-models-bucket", {
    location: "US",
    forceDestroy: true, // Permite borrar el bucket aunque tenga archivos (útil en dev)
    uniformBucketLevelAccess: true,
});

// ====================================================
// 3. PERMISOS (IAM)
// ====================================================
// Service Account para que las VMs puedan leer/escribir en el Bucket
const vmServiceAccount = new gcp.serviceaccount.Account("vm-sa", {
    accountId: "confide-req-sa",
    displayName: "Service Account for Cloud and Fog Nodes",
});

// Dar permisos de Admin de Storage a la cuenta de servicio
const bucketIam = new gcp.storage.BucketIAMMember("sa-storage-admin", {
    bucket: modelsBucket.name,
    role: "roles/storage.objectAdmin",
    member: pulumi.interpolate`serviceAccount:${vmServiceAccount.email}`,
});

// ====================================================
// 4. SCRIPT DE INICIO (INSTALACIÓN AUTOMÁTICA)
// ====================================================
// Este script se ejecutará al crear la VM para instalar Python, PyTorch, etc.
const startupScript = `#!/bin/bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git

# Crear entorno virtual
mkdir -p /app
chmod 777 /app
cd /app
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias base (CPU version para ahorrar espacio/tiempo en demo)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install flwr transformers datasets pandas numpy scikit-learn google-cloud-storage requests flask flask-cors

# --- CONFIGURACIÓN DE ENTORNO ---
echo "export MODEL_BUCKET_NAME=${modelsBucket.name}" >> /app/.env
echo "source /app/.env" >> /app/venv/bin/activate

chmod -R 777 /app
`;
// flwr==1.5.0

// ====================================================
// 5. INSTANCIAS DE COMPUTO
// ====================================================

// --- CLOUD SERVER (Orquestador Global + DeepModel) ---
const cloudServerInstance = new gcp.compute.Instance("cloud-server", {
    machineType: "e2-standard-4", 
    zone: "us-central1-a",
    tags: ["cloud-node"],
    bootDisk: {
        initializeParams: {
            image: "debian-cloud/debian-11",
            size: 50,
        },
    },
    networkInterfaces: [{
        network: vpc.id,
        subnetwork: subnet.id,
        accessConfigs: [{}], 
    }],
    serviceAccount: {
        email: vmServiceAccount.email,
        scopes: ["https://www.googleapis.com/auth/cloud-platform"],
    },
    metadataStartupScript: startupScript,
});

// --- FOG NODE (Orquestador Regional + FastModel) ---
const fogInstance = new gcp.compute.Instance("fog-server", {
    machineType: "e2-standard-4", 
    zone: "us-central1-b", // Simulación de distancia física
    tags: ["fog-node"],
    bootDisk: {
        initializeParams: {
            image: "debian-cloud/debian-11",
            size: 50,
        },
    },
    networkInterfaces: [{
        network: vpc.id,
        subnetwork: subnet.id,
        accessConfigs: [{}], 
    }],
    serviceAccount: {
        email: vmServiceAccount.email,
        scopes: ["https://www.googleapis.com/auth/cloud-platform"],
    },
    metadataStartupScript: startupScript,
});

// ====================================================
// 6. EXPORTS
// ====================================================
export const cloudPublicIp = cloudServerInstance.networkInterfaces.apply(ni => ni[0].accessConfigs![0].natIp);
export const fogPublicIp = fogInstance.networkInterfaces.apply(ni => ni[0].accessConfigs![0].natIp);
export const bucketName = modelsBucket.name;
export const cloudName = cloudServerInstance.name;
export const fogName = fogInstance.name;