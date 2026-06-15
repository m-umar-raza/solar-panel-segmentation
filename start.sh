#!/bin/bash
mkdir -p models
if [ ! -f models/best_model.pth ]; then
    echo "Downloading model from GCS..."
    python -c "
from google.cloud import storage
client = storage.Client()
bucket = client.bucket('solar-panel-seg-models')
blob = bucket.blob('best_model.pth')
blob.download_to_filename('models/best_model.pth')
print('Model downloaded successfully')
"
fi
uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8080}
