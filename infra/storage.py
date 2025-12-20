import pulumi
from pulumi_gcp import storage

class ModelRepository:
    """Bounded Context: Storage.
    Actúa como repositorio de artefactos (modelos)."""

    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name

    def create_bucket(self) -> storage.Bucket:
        return storage.Bucket(self.bucket_name,
            location="US",
            force_destroy=True,
            uniform_bucket_level_access=True
        )