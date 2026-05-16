# Solar Panel Segmentation

End-to-end deep learning pipeline for detecting and segmenting solar panels in aerial imagery.

## Status

Work in progress. Currently building the data pipeline.

## Tech Stack

- PyTorch for model training
- segmentation-models-pytorch for U-Net architecture
- Albumentations for data augmentation
- FastAPI for model serving (planned)
- Docker and GitHub Actions for deployment (planned)

## Setup

```bash
python -m venv venv
source venv/Scripts/activate  # Git Bash on Windows
pip install -r requirements.txt
```

## Project Structure

See `docs/architecture.md` for details.s