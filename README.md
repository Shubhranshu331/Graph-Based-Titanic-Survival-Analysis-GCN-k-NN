# Titanic Survival Prediction using Graph Neural Networks

This project applies Graph Neural Networks (GCN) to the classic Titanic dataset by converting tabular data into a graph structure using k-Nearest Neighbors (k-NN). Instead of treating passengers independently, relationships between similar passengers are leveraged for improved learning.

## Overview

- Dataset: Titanic (891 passengers)
- Approach:
  - Feature preprocessing
  - k-NN graph construction
  - Graph-based learning using GCN
- Task: Binary classification (Survived / Not Survived)

## Key Components

### 1. Graph Construction
- k-NN (k=5) used to connect similar passengers
- Graph built using edge list representation
- Nodes represent passengers

### 2. Model
- Graph Convolutional Network (GCN)
- Learns node embeddings using neighborhood information
- Optimized using negative log-likelihood loss

### 3. Training & Evaluation
- Training and validation loss tracked over epochs
- ROC Curve and AUC used for performance evaluation

### 4. Visualizations
- Loss curves (training vs validation)
- ROC curve
- NetworkX static graph visualization
- Subgraph visualization (first 100 nodes)
- Interactive graph using PyVis
- Node degree distribution

## Results

- Loss curves help identify underfitting/overfitting
- ROC-AUC evaluates classification performance across thresholds
- Graph visualization shows structural relationships between passengers

## Outputs

Saved in `/kaggle/working/`:
- `loss_curves.png`
- `roc_curve.png`
- `titanic_knn_graph.png`
- `titanic_knn_subgraph.png`
- `titanic_knn_graph.html`
- `degree_distribution.png`

## Tech Stack

- Python
- PyTorch / PyTorch Geometric (for GCN)
- NetworkX
- Matplotlib
- Scikit-learn
- PyVis

## Notes

- Graph-based learning provides a different perspective compared to traditional tabular ML models.
- Subgraph visualization is recommended for clarity due to graph density.
- Interactive visualization helps explore node-level relationships.

## Future Improvements (optional)

- Hyperparameter tuning (k value, layers, hidden units)
- Comparison with traditional ML models
- Feature engineering for better graph connectivity
