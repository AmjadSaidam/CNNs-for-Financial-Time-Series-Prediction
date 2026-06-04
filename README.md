# CNNs-for-Financial-Time-Series-Prediction

This repository contains the assoctited code and results fro the paper *Convolutional Neural Networks (CNNs) for Asset Price Prediction*. 

The paper investigates the use of Gramian Anguler Fields (GAFs) as a feature enginearing technique that transforms time series (asset price) into a matrix of with high hierarchical structure suitbale for training using a CNN.

## Repository Structure 

```
CNNs-for-Financial-Time-Series-Prediction/
├── code/
│   ├── cnn.py                    # CNN model, training, and evaluation logic
│   └── data_pre_process.py       # GAF preprocessing pipeline
├── notebooks/
│   └── model_application.ipynb   # End-to-end: data → train → backtest
├── paper/
│   └── Convolutional_Neural_Networks_for_Asset_Price_Prediction.pdf  # Reference paper
├── requirements.txt              # Python dependencies
├── CLAUDE.md                     # Codebase instructions for Claude Code
└── README.md                     # Project overview
```

