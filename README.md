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
## 🎯 Qualitative Results

The following figure shows the enhancement results for deuteranomalous trichromats. Our method better preserves fine-grained details (e.g., feather textures, thread separations) compared to the conventional method [12].

![Teaser](docs/display.png)
**Figure:** Enhanced images for deuteranomalous trichromats. From left to right: Original image, our result, result from Yang et al. [12]. The second row shows the simulated perceptions for deuteranomaly.
## 🛠 Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/your_username/your_repo_name.git
   cd your_repo_name
## 📁 Dataset

The dataset used for training and validation can be downloaded from 10.5281/zenodo.17187079


The dataset contains 5,910 original RGB images and their corresponding enhanced versions for three conditions:
- `deutan`: Enhanced for deuteranomaly and deuteranopia.
- `protan_mild`: Mildly enhanced for protanomaly.
- `protan_severe`: Severely enhanced for protanopia and some protanomaly.

Please refer to our paper for detailed construction process.
## 🏋️‍♂️ Training

### Preparing the Dataset
1. Download the dataset from the link provided above and extract it.
2. Organize the dataset into the following structure:
## 📜 Citation

If you use this code or our dataset in your research, please cite our paper:

```bibtex
@article{he2025enhancing,
  title={Enhancing Color Image for Anomalous Trichromats: A Deep Learning Approach with Detail Preservation},
  author={He, Zihao and Ma, Zejun and Ma, Hai and Zhang, Bingkun and Ma, Ruiqing},
  journal={Journal Name},
  volume={},
  pages={},
  year={2025},
  publisher={Springer}
}
## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
