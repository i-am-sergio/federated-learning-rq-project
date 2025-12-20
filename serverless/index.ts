import * as pulumi from "@pulumi/pulumi";
import * as gcp from "@pulumi/gcp";

// 1. Crear Bucket para el código
const bucket = new gcp.storage.Bucket("serverless-models-bucket", {
    location: "US",
    forceDestroy: true,
});

// 2. Empaquetar: Combinamos el código de /serverless y el modelo de /models
const archive = new gcp.storage.BucketObject("function-archive", {
    bucket: bucket.name,
    source: new pulumi.asset.AssetArchive({
        "main.py": new pulumi.asset.FileAsset("./lambda_function/main.py"),
        "requirements.txt": new pulumi.asset.FileAsset("./lambda_function/requirements.txt"),
        "mpnet_fed_requirements.pth": new pulumi.asset.FileAsset("./lambda_function/mpnet_fed_requirements.pth"),
    }),
});

// 3. Desplegar la Cloud Function (2nd Gen)
const predictFunction = new gcp.cloudfunctionsv2.Function("requirement-classifier", {
    location: "us-central1",
    buildConfig: {
        runtime: "python311",
        entryPoint: "predict_requirement",
        source: {
            storageSource: {
                bucket: bucket.name,
                object: archive.name,
            },
        },
    },
    serviceConfig: {
        maxInstanceCount: 3,
        minInstanceCount: 0,
        availableMemory: "2Gi", // Necesario para cargar MPNet en RAM
        timeoutSeconds: 60,
    },
});

// 4. Hacer la función pública (opcional, solo para pruebas)
const invoker = new gcp.cloudfunctionsv2.FunctionIamMember("invoker", {
    location: predictFunction.location,
    cloudFunction: predictFunction.name,
    role: "roles/cloudfunctions.invoker",
    member: "allUsers",
});

export const endpoint = predictFunction.serviceConfig.apply(sc => sc?.uri || "");