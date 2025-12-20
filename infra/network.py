import pulumi
from pulumi_gcp import compute

class NetworkManager:
    """Bounded Context: Networking. 
    Responsable de proveer conectividad y seguridad."""
    
    def __init__(self, name_prefix: str):
        self.prefix = name_prefix

    def create_static_ip(self) -> compute.Address:
        return compute.Address(f"{self.prefix}-static-ip")

    def create_firewall(self, network="default") -> compute.Firewall:
        return compute.Firewall(f"{self.prefix}-firewall",
            network=network,
            allows=[
                compute.FirewallAllowArgs(protocol="tcp", ports=["8080", "22"]), # Flower gRPC & SSH
            ],
            source_ranges=["0.0.0.0/0"],
            target_tags=["fl-server"]
        )