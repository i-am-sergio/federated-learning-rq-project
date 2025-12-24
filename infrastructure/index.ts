import * as gcp from "@pulumi/gcp";

// 1. Red y 2. Subred (Sin cambios)
const vpc = new gcp.compute.Network("vpc-network", {
    autoCreateSubnetworks: false,
});

const subnet = new gcp.compute.Subnetwork("vpc-subnet", {
    ipCidrRange: "10.0.1.0/24",
    region: "us-central1",
    network: vpc.id,
});

// 3. Firewall (Puerto 22 y 8080 para Flower)
const firewall = new gcp.compute.Firewall("allow-ssh-flower", {
    network: vpc.id,
    allows: [{
        protocol: "tcp",
        ports: ["22", "8080", "5000"],
    }],
    sourceRanges: ["0.0.0.0/0"],
    targetTags: ["server-node"], 
});

// 4. Instancia de VM 
const vmInstance = new gcp.compute.Instance("ml-server", {
    machineType: "e2-standard-4", // 4 vCPUs y 16GB RAM (Sin GPU)
    zone: "us-central1-a",
    tags: ["server-node"],
    bootDisk: {
        initializeParams: {
            image: "debian-cloud/debian-11",
            size: 50, // Disco de 50GB 
        },
    },
    networkInterfaces: [{
        network: vpc.id,
        subnetwork: subnet.id,
        accessConfigs: [{}], 
    }],
});


export const publicIp = vmInstance.networkInterfaces.apply(ni => ni[0].accessConfigs![0].natIp);
export const instanceName = vmInstance.name;
export const firewallName = firewall.name;