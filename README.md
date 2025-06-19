# LungConVT-Net: A Visual Transformer Network with Blended Features for Pneumonia Detection

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.10%2B-orange)](https://www.tensorflow.org/)

## 🔬 Abstract

Respiratory ailments, especially pneumonia, demand advanced diagnostic tools for timely and more accurate detection. Addressing this critical need, we introduce LungConVT-Net, an innovative architecture that blend the strengths of Vision Transformer (ViT) and Convolutional Neural Networks (CNN) to delineate among three crucial lung conditions viz., Viral Pneumonia, Bacterial Pneumonia, and COVID-19, as well as normal lung manifestations. The proposed model leverages depthwise separable convolutions, optimizing computational efficiency without sacrificing spatial filtering. Additionally, we integrate the proposed Dynamic Hierarchical Multi-Head Attention Convolution (DH-MHAC) and Adaptive Multi-Granular Multi-Head Attention (AMG-MHA) modules. These modules bridge the self-attention mechanisms with convolutions and utilize non-overlapping patches, culminating in enhanced feature extraction, respectively. A strategically incorporated Multi-Layer Perceptron (MLP) block within the AMG-MHA refines the model's prowess in understanding intricate data patterns. The Gradient Connection Enhancers (GCE) capture both long-range and short-range feature dependencies, addressing potential challenges in gradient descent and promoting training stability. Experimental evaluations, spanning from bi-class to complex quad-class combinations, reveal the model's performance compared to the state-of-the-art models.  The results unveil AUC scores consistently surpassing 99\% in most bi-class scenarios and show strong performance in complex multi-class settings, with AUC scores exceeding 99\% for Pneumonia, COVID-19, and Normal categories. Moreover, in the quad-class combination, our model achieves an AUC score of 98.19\%, highlighting LungConVT-Net's effectiveness in advancing respiratory disease diagnostics.


### Key Features

- **Hybrid Architecture**: Seamlessly integrates CNN and Transformer blocks
- **High Accuracy**: 2% improve accuracy on multi-class lung disease classification
- **Explainable AI**: Integrated Grad-CAM for visual explanations
- **Comprehensive Evaluation**: ROC curves, confusion matrices, and statistical tests
- **Reproducible**: Fixed seeds and detailed environment specifications
- **Production Ready**: Modular code structure with CLI interface


## Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 16GB RAM minimum

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/LaskerAsifuzzaman/lungconvtnet.git
cd lungconvtnet
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python validate_setup.py
```

### Training the Model

```bash
# For headless environments (servers, docker)
./run_headless.sh --mode train --config config.yaml

# Or directly
python main.py --mode train --config config.yaml
```

### Evaluating the Model

```bash
python main.py --mode evaluate --model_path experiments/lungconvt_baseline/models/lungconvt_final.h5 --config config.yaml
```

### Generating Explanations

```bash
python main.py --mode explain --model_path experiments/lungconvt_baseline/models/lungconvt_final.h5 --image_path path/to/image.jpg
```

## 📁 Project Structure

```
lungconvtnet/
├──  data_loader.py       # Data loading and preprocessing
├──  model.py            # LungConVT-NET architecture
├──  train.py            # Training pipeline
├──  evaluate.py         # Model evaluation
├──  explain.py          # Grad-CAM explanations
├──  utils.py            # Helper functions
├──  main.py             # CLI interface
├──  demo.ipynb          # Interactive demo
├──  config.yaml         # Configuration
└──  requirements.txt    # Dependencies
```

## Model Architecture

LungConVT-NET features a hierarchical architecture with four main components:

1. **Initial Feature Extraction**: Convolutional blocks for low-level features
2. **Depthwise Separable Convolutions**: Efficient feature extraction
3. **Dual-Head Convolutional Multi-Head Attention (DHC-MHA)**: Local-global feature integration
4. **Adaptive Multi-Grained Multi-Head Attention (AMG-MHA)**: Multi-scale feature analysis


## Dataset

The model is trained on a curated dataset of chest X-ray images:

- **Total Images**: 9,208
- **Classes**: 4 (COVID-19, Normal, Bacterial Pneumonia, Viral Pneumonia)
- **Split**: 80% training, 20% testing
- **Augmentation**: Rotation, zoom, shift, shear

### Dataset Structure
```
data/
├── data_mapping.csv
└── sample_data/
    ├── COVID-19/
    ├── Normal/
    ├── Pneumonia-Bacterial/
    └── Pneumonia-Viral/
```

## Reproducibility

We ensure full reproducibility through:

- Fixed random seeds (NumPy, TensorFlow, Python)
- Exact package versions in `requirements.txt`
- Hardware specifications documented
- Detailed training logs saved automatically


## Explainability

LungConVT-NET includes Grad-CAM visualizations to understand model decisions:

## Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 16GB
- GPU: 4GB 

**Recommended:**
- CPU: 8+ cores
- RAM: 16GB+
- GPU: NVIDIA RTX 2070+ (8GB VRAM)

## Configuration

Edit `config.yaml` to customize:

```yaml
model:
  input_size: [256, 256]
  
training:
  batch_size: 16
  epochs: 50
  learning_rate: 0.001
  
data:
  test_size: 0.2
  augmentation:
    rotation_range: 0.25
    zoom_range: 0.15
    shear_range: 0.10
    shift: 0.05
```

## Troubleshooting

<details>
<summary>Common Issues & Solutions</summary>

### Display/Qt Errors
```bash
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen
```

### Out of Memory
- Reduce `batch_size` in config.yaml
- Enable mixed precision training

### Installation Issues
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

</details>


## 🙏 Acknowledgments

- Medical imaging community for datasets
- TensorFlow and Keras teams
- Vision Transformer and PulmoNetX papers for inspiration
