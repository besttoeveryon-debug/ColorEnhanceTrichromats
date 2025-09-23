# Enhancing Color Images for Anomalous Trichromats with Detail Preservation

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Framework](https://img.shields.io/badge/PyTorch-1.7+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains the official PyTorch implementation of the paper:  
**"Enhancing Color Image for Anomalous Trichromats: A Deep Learning Approach with Detail Preservation"**

> **Abstract:** This study introduces a deep-learning-based method that simultaneously achieves contrast enhancement, naturalness preservation, and fine-grained detail preservation for anomalous trichromats. By constructing a dataset validated by color-deficient observers and employing a modified Pix2Pix GAN model with Swin Transformer and NonLocalBlock modules, our approach demonstrates superior performance.
## ✨ Key Features

- **Dual-Attention Generator:** Integrates **Swin Transformer** for local detail refinement and **NonLocalBlock** for global semantic contrast enhancement within a U-Net architecture.
- **Perceptually-Validated Dataset:** Utilizes a dataset where the enhanced images are optimized based on subjective experiments with color-deficient observers.
- **Comprehensive Loss Function:** Combines adversarial loss, perceptual loss (VGG-based), color perception loss, and edge loss to guide the model towards high-quality, natural results.
- **State-of-the-Art Performance:** Outperforms traditional methods in terms of chromatic difference (CD), perceptual color ratio (PCR), and structural similarity (SSIM).
