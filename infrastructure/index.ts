import * as gcp from "@pulumi/gcp";

// 1. Crear una Red VPC personalizada
const vpc = new gcp.compute.Network("vpc-network", {
    autoCreateSubnetworks: false,
});

// 2. Crear una Subred en una región específica
const subnet = new gcp.compute.Subnetwork("vpc-subnet", {
    ipCidrRange: "10.0.1.0/24",
    region: "us-central1",
    network: vpc.id,
});

// 3. Configurar Firewall para permitir SSH (puerto 22)
const firewall = new gcp.compute.Firewall("allow-ssh", {
    network: vpc.id,
    allows: [{
        protocol: "tcp",
        ports: ["22"],
    }],
    sourceRanges: ["0.0.0.0/0"],
    targetTags: ["ssh-enabled"], 
});

// 4. Crear la Instancia de VM
const vmInstance = new gcp.compute.Instance("web-server", {
    machineType: "e2-medium", // Memoria: 4 GB, vCPUs: 2
    zone: "us-central1-a",
    tags: ["ssh-enabled"], // Aplicar el tag para el firewall
    
    bootDisk: {
        initializeParams: {
            image: "debian-cloud/debian-11",
            size: 20, 
        },
    },

    networkInterfaces: [{
        network: vpc.id,
        subnetwork: subnet.id,
        accessConfigs: [{}], 
    }],

    // Script de inicio opcional
    metadataStartupScript: "echo 'Hola desde Pulumi' > /var/www/html/index.html",
});

// Exportar la IP pública de la instancia
export const publicIp = vmInstance.networkInterfaces.apply(ni => ni[0].accessConfigs![0].natIp);
export const instanceName = vmInstance.name;
export const firewallName = firewall.name;