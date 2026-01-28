# SE-ResNet: Channel-wise Attention for Low-Resolution Image Classification

This project implements a Squeeze-and-Excitation Residual Network (SE-ResNet) to solve a fine-grained classification problem on $32 \times 32$ images. It addresses signal-to-noise challenges in low-resolution data.

---

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [The Challenge](#-the-challenge)
- [Methodology & Architecture](#-methodology--architecture)
- [Getting Started](#-getting-started)
- [Training Strategy](#-training-strategy)
- [Results](#-results)
- [References](#-references)

---

## 🔍 Project Overview

The objective of this project is to approximate the conditional probability distribution $P(Y|\mathbf{X})$ for a dataset of 16,000 observations. The inputs are single-channel grayscale images ($\mathbb{R}^{32 \times 32}$) categorized into 4 mutually exclusive classes.

Standard CNNs often struggle with low-resolution data due to limited spatial information. This project proposes a **SE-ResNet** architecture that combines two powerful inductive biases:
1.  **Residual Learning:** To ease gradient flow and allow for deeper networks.
2.  **Channel Attention (SE Blocks):** To adaptively recalibrate channel-wise feature responses, explicitly modeling interdependencies between channels.

---

## 📉 The Challenge

* **Input Space:** High-dimensional raw feature vectors ($\mathbb{R}^{1024}$) with single-channel intensity.
* **Data Constraints:** No predefined canonical test set; requires rigorous stratified shuffling to ensure valid generalization error estimation ($80/10/10$ split).
* **Signal-to-Noise:** Significant intra-class variance in orientation and lighting, necessitating robust feature extraction.

---

## 🧠 Methodology & Architecture

### 1. Data Preprocessing
* **Normalization:** Min-max scaling projects input space to $[0, 1]$ to ensure an isotropic loss landscape.
* **Pseudo-RGB Expansion:** Inputs are transformed via $\mathbf{x}_{new} = \text{Repeat}(\text{Unsqueeze}(\mathbf{x})) \in \mathbb{R}^{32 \times 32 \times 3}$. This allows the network to utilize standard architectural backbones designed for 3-channel data.

### 2. The SE-ResNet Architecture
The model (`SE_ResNet_Advanced`) is constructed using the Keras Functional API.

#### The Building Block
Each residual block typically follows this flow:
1.  **Convolutional Path:** `Conv2D` $\rightarrow$ `BatchNormalization` $\rightarrow$ `ReLU`.
2.  **Squeeze-and-Excitation (SE) Mechanism:**
    * *Squeeze:* Global Average Pooling aggregates spatial information into a channel descriptor.
    * *Excitation:* A gating mechanism (Sigmoid) with a reduction ratio ($r=16$) learns non-linear interactions between channels.
    * *Scale:* The input feature map is reweighted by the learned channel activations.
3.  **Residual Connection:** The scaled output is added element-wise to the input tensor ($\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$).

---

## 🚀 Getting Started

### Prerequisites
* Google Colab (recommended for GPU access) or a local Python environment.
* **Dataset:** `4class_32x32.npz` (Ensure this is placed in the correct directory).