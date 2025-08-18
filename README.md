# PockNet Data Generation

A comprehensive protein feature extraction and analysis pipeline for ligand binding site prediction using machine learning. This repository processes protein structures to extract P2Rank-compatible molecular features and compare different ML approaches for binding site prediction.

## 🧬 Overview

This project extracts molecular features from protein structures that can be used for machine learning-based ligand binding site prediction, following methodologies similar to [P2Rank](https://github.com/rdk/p2rank). The pipeline supports multiple datasets, feature extraction, and comparative analysis of various machine learning models.

## 📊 Dataset

The protein structures used in this project are sourced from the [p2rank-datasets repository](https://github.com/rdk/p2rank-datasets), which provides standardized benchmarks for binding site prediction research.

### Available Datasets
- **`all_train.ds`** - Complete training dataset
- **`chen11.ds`** - Chen11 benchmark dataset for evaluation
- **`holo4k.ds`** - Holo4K dataset (holo proteins with known binding sites)
- **`bu48.ds`** - BU48 dataset
- **`joind.ds`** - Combined/joined dataset
- **`test_single.ds`** - Single protein test case
- **`test_small.ds`** - Small test dataset

## 🔧 Installation & Setup

### Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd pocknet_datagenereation

# Create conda environment from file
conda env create -f environment.yml
conda activate PockNet
```

### Key Dependencies
- **Python 3.10+**
- **PyMOL** - Protein structure analysis
- **OpenBabel** - Chemical informatics
- **NumPy, Pandas** - Data processing
- **Scikit-learn** - Machine learning models
- **Matplotlib, Seaborn** - Visualization
- **Jupyter** - Interactive analysis

## 🚀 Usage

### Feature Extraction

The main feature extraction script processes protein PDB files and generates feature vectors:

```bash
python extract_protein_features.py <dataset_file> <output_directory>
```

**Example:**
```bash
python extract_protein_features.py chen11.ds ./out_test
```

### Extracted Features

The script generates P2Rank-compatible features including:

- **Spatial Features**: SAS (Solvent Accessible Surface) point coordinates (x, y, z)
- **Structural Features**: Chain ID, residue numbers, secondary structure
- **Chemical Properties**: Hydrophobicity, charge, polarity, atom types
- **Geometric Features**: Atom density, protrusion indices, cavity measurements
- **Volumetric Features**: Site descriptors and volume-based properties
- **Dynamic Features**: B-factor information and flexibility measures

## 📁 Directory Structure

```
├── 📄 extract_protein_features.py    # Main feature extraction script
├── 📊 comparison.ipynb               # ML model comparison analysis
├── 📊 insights.ipynb                 # Data exploration and insights
├── 🔧 environment.yml                # Conda environment specification
│
├── 📂 out_train/                     # Training feature vectors
│   └── vectorsTrain.csv
├── 📂 out_test/                      # Test feature vectors  
│   └── vectorsTest.csv
├── 📂 out_finetune/                  # Fine-tuning datasets
│   └── vectorsTrain.csv
├── 📂 chen11_test_prep/              # Preprocessed Chen11 proteins
│   ├── a.001.001.001_1s69a.pdb
│   ├── a.001.001.002_1do1a.pdb
│   └── a.001.001.003_1qgwc.pdb
│
├── 📂 RF/                            # Random Forest model
│   ├── config.yaml                   # RF configuration
│   ├── model_utils.py               # Model utilities
│   └── train_rf.py                  # RF training script
│
├── 📂 plots/                         # Generated visualizations
│   ├── comprehensive_model_comparison.csv
│   ├── iou_performance_by_model.png
│   ├── model_performance_heatmap.pdf
│   └── ... (additional plots)
│
└── 📂 Configuration Files
    ├── atomic-properties.csv         # Atomic property lookup
    ├── volsite-atomic-properties.csv # Volume site properties
    └── *.ds                          # Dataset definition files
```

## 🔬 Analysis & Modeling

### Jupyter Notebooks

1. **`comparison.ipynb`** - Comprehensive comparison of different ML models:
   - Random Forest vs. Neural Networks
   - Performance metrics (Precision, Recall, F1, IoU)
   - Feature importance analysis
   - Cross-validation results

2. **`insights.ipynb`** - Data exploration and feature analysis:
   - Feature distribution analysis
   - Correlation studies
   - Data quality assessment
   - Visualization of binding site properties

### Machine Learning Models

The repository includes implementations and comparisons of:
- **Random Forest** (primary model)
- **Neural Networks** (TabNet, MLP)
- **Gradient Boosting** (XGBoost)
- **Support Vector Machines**

### Performance Metrics

Models are evaluated using:
- **IoU (Intersection over Union)** - Primary metric for spatial prediction
- **Precision/Recall** - Classification performance
- **F1-Score** - Balanced accuracy measure
- **AUC-ROC** - Ranking quality assessment

## ⚙️ Configuration

### Random Forest Configuration (`RF/config.yaml`)
```yaml
rf_trees: 200           # Number of decision trees
rf_depth: 0            # Unlimited tree depth
rf_threads: 0          # Use all available CPU cores
class_weight: "balanced" # Handle class imbalance
validation_size: 0.2   # Validation set fraction
```

### Feature Extraction Settings
- **SAS Probe Radius**: 1.4 Å (water molecule size)
- **Grid Resolution**: Configurable for different precision levels
- **Parallel Processing**: Multi-threaded extraction for large datasets

## 📈 Results & Visualizations

The `plots/` directory contains comprehensive analysis results:

- **Performance Comparisons**: Model accuracy across different datasets
- **Feature Importance**: Which molecular properties are most predictive
- **IoU Analysis**: Spatial prediction quality assessment  
- **Efficiency Studies**: Training time vs. performance trade-offs

## 🏃‍♂️ Quick Start Example

```bash
# 1. Setup environment
conda activate PockNet

# 2. Extract features from Chen11 dataset
python extract_protein_features.py chen11.ds ./output_chen11

# 3. Train Random Forest model
cd RF
python train_rf.py

# 4. Run comparative analysis
jupyter notebook comparison.ipynb
```

## 📝 Log Files

- **`feature_extraction.log`** - Detailed extraction progress and debugging info
- **`tabnet_summary_report.txt`** - TabNet model performance summary


## 📚 References

- **P2Rank Datasets**: [https://github.com/rdk/p2rank-datasets](https://github.com/rdk/p2rank-datasets)
- **P2Rank Method**: Krivák, R. & Hoksza, D. P2Rank: machine learning based tool for rapid and accurate prediction of ligand binding sites from protein structure. *J Cheminform* **10**, 39 (2018).

---

*This repository contains the complete pipeline for protein feature extraction and binding site prediction analysis as part of advanced computational biology research.*
