# ECG Arrhythmia Classification using Deep Spatiotemporal Learning

## Overview

This project investigates automated cardiac arrhythmia classification using Electrocardiogram (ECG) time-series signals from the MIT-BIH Arrhythmia Database.

The study compares classical machine learning baselines against deep sequence architectures to evaluate the importance of preserving both:

* waveform morphology (spatial features)
* temporal rhythm dependencies (sequential features)

A hybrid 1D-CNN + Bidirectional LSTM architecture was developed to classify heartbeat segments into normal and arrhythmic categories while following a strict leakage-free patient-wise evaluation methodology.

---

# Key Results

| Model               | Accuracy   | Precision | Recall   | F1-Score |
| ------------------- | ---------- | --------- | -------- | -------- |
| Logistic Regression | 75.25%     | 0.28      | 0.88     | 0.42     |
| MLP Baseline        | 90.28%     | 0.51      | 0.98     | 0.68     |
| Tuned CNN-BiLSTM    | **93.31%** | **0.61**  | **0.96** | **0.75** |

## Best Performing Model

Hybrid 1D-CNN + BiLSTM

## Clinical Outcome

The final model successfully identified 96% of arrhythmias in completely unseen patients.

---

# Dataset

Dataset: MIT-BIH Arrhythmia Database

Official Source:
[https://physionet.org/content/mitdb/1.0.0/](https://physionet.org/content/mitdb/1.0.0/)

## Dataset Information

* 48 half-hour ECG recordings
* Two-channel ambulatory ECG signals
* 47 clinical subjects
* Sampling frequency: 360 Hz

## Classification Objective

| Class | Description                                   |
| ----- | --------------------------------------------- |
| 0     | Normal Sinus Rhythm                           |
| 1     | Arrhythmia (PVCs, Bundle Branch Blocks, etc.) |

The dataset is highly imbalanced, making recall-sensitive evaluation essential.

---

# Project Objectives

* Compare classical ML baselines against deep sequence architectures
* Evaluate the importance of temporal modeling in ECG signals
* Prevent patient identity leakage during evaluation
* Analyze physiological causes of model misclassification
* Develop a clinically meaningful arrhythmia detection pipeline

---

# ECG Preprocessing Pipeline

Raw ECG signals contain:

* baseline wander
* muscle noise
* sensor artifacts
* patient-specific electrical signatures

A preprocessing pipeline was implemented to stabilize learning and improve generalization.

---

## 1. Temporal Windowing

Each heartbeat was extracted using a fixed-length 300-sample window centered on the annotated R-peak.

At a sampling frequency of 360 Hz, this corresponds to approximately:

* 0.83 seconds per heartbeat window

This captures:

* P-wave
* QRS complex
* T-wave

while minimizing overlap with neighboring beats.

---

## 2. Per-Window Normalization

Independent Z-score normalization was applied to every heartbeat window:

```text
z = (x - μ) / σ
```

This:

* removes baseline drift
* standardizes amplitudes
* forces the model to learn waveform morphology rather than raw voltage magnitude

---

## 3. Leakage-Free Patient Split

To prevent patient identity leakage:

* GroupShuffleSplit was implemented
* all heartbeat windows were grouped by Patient ID
* test patients remained completely unseen during training

This ensures realistic clinical generalization.

---

# Model Architectures

## 1. Logistic Regression (Linear Baseline)

A flattened heartbeat vector was used as input to a logistic regression classifier.

Purpose:

* establish a linear performance baseline
* demonstrate the non-linear nature of arrhythmia classification

### Limitation

Flattening destroys temporal structure.

---

## 2. Multi-Layer Perceptron (MLP)

A fully connected neural network introduced non-linearity while still operating on flattened heartbeat vectors.

### Limitation

The model cannot explicitly preserve:

* waveform topology
* sequential rhythm dependencies

---

## 3. Hybrid 1D-CNN + BiLSTM

The final architecture combines spatial and temporal sequence modeling.

---

## 1D-CNN: Morphological Feature Extraction

Convolutional filters slide across the ECG waveform to detect:

* QRS widening
* abnormal spikes
* waveform distortions
* local cardiac morphology

The CNN acts as an automated feature extractor.

---

## BiLSTM: Temporal Sequence Modeling

The Bidirectional LSTM processes the sequence:

* forward in time
* backward in time

This preserves:

* rhythm context
* sequential dependencies
* surrounding cardiac behavior

---

# Handling Class Imbalance

Because arrhythmias represent the minority class, standard optimization methods may collapse toward predicting healthy beats.

To address this:

* BCEWithLogitsLoss was implemented
* dynamic `pos_weight` balancing was applied

This penalizes false negatives more heavily.

Clinical motivation:
Missing an arrhythmia is more dangerous than over-flagging a healthy beat.

---

# Hyperparameter Optimization

A randomized search strategy evaluated:

* hidden dimensions
* dropout rates
* dense layer sizes
* learning configurations

Early stopping was implemented to reduce overfitting.

---

# Results and Performance Analysis

The progression from linear baselines to deep sequence architectures produced a significant improvement in classification capability.

## Logistic Regression

* weak class separation
* high false positive rate
* poor precision

## MLP Baseline

* strong improvement through non-linearity
* limited by flattened temporal representation

## CNN-BiLSTM

* preserved waveform morphology
* retained temporal rhythm context
* achieved the best balance between sensitivity and precision

---

# ROC-AUC Analysis

The CNN-BiLSTM achieved an AUC approaching 1.0, significantly outperforming the classical baselines across all classification thresholds.

This indicates strong generalized separation between:

* healthy cardiac morphology
* pathological arrhythmias

---

# Misclassification Analysis

## False Negatives (4% Miss Rate)

Most missed arrhythmias involved:

* unusually narrow Premature Ventricular Contractions (PVCs)

These abnormal beats visually resembled healthy sinus rhythms, making them difficult for CNN filters to distinguish.

---

## False Positives

Healthy beats were occasionally flagged due to:

* electromyographic muscle noise
* severe baseline wander
* naturally widened QRS boundaries

This conservative bias is partially intentional due to weighted loss optimization prioritizing patient safety.

---

# Final Recommendation

The hybrid CNN-BiLSTM architecture demonstrated the strongest overall performance for ECG sequence classification.

The results show that:

* ECG classification benefits from preserving both spatial morphology and temporal dynamics
* deep sequence architectures outperform flat machine learning baselines
* leakage-free evaluation is critical for clinically meaningful metrics

Final model performance:

* 93.31% Accuracy
* 0.96 Recall
* 0.75 F1-Score

---

# Tech Stack

## Languages & Libraries

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* WFDB
* Matplotlib
* Seaborn

---

# Installation

```bash
git clone <your-repository-url>

cd AI_ML_SEM5_FINALPROJECT

pip install -r requirements.txt
```

---

# Dataset Setup

The notebook automatically downloads the MIT-BIH dataset using either:

* direct ZIP extraction
* or the official WFDB API fallback method

Downloaded files are stored inside:

```text
data/
```

---

# Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebooks/ECG-Arrhythmia-Detection.ipynb
```

---

# Future Improvements

Potential future extensions include:

* Transformer-based architectures
* Attention mechanisms
* Multi-class arrhythmia classification
* Real-time ECG inference
* Explainable AI using Grad-CAM or attention visualization
* Edge deployment for embedded medical systems

---

# References

1. Moody, G. B., & Mark, R. G. (2001). *The impact of the MIT-BIH Arrhythmia Database*. IEEE Engineering in Medicine and Biology Magazine, 20(3), 45–50.

2. Goldberger, A. L., et al. (2000). *PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals*. Circulation, 101(23), e215–e220.

3. MIT-BIH Arrhythmia Database — PhysioNet
   [https://physionet.org/content/mitdb/1.0.0/](https://physionet.org/content/mitdb/1.0.0/)
