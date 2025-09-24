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

We provide two methods to set up the environment. Please follow one of them.

### Prerequisites
- Ensure you have **Python 3.8** or later installed.
- For GPU training, ensure you have a compatible **NVIDIA GPU and CUDA** toolkit installed.

### Method 1: Using Conda (Recommended for Full Dependency Management)

This method is highly recommended as it automatically handles both Python and non-Python dependencies (like CUDA toolkits) in an isolated environment.

1.  **Create and activate the conda environment** from the provided `environment.yml` file. This will install all necessary packages, including PyTorch with GPU support.
    ```bash
    conda env create -f environment.yml
    conda activate color_enhancement
    ```

### Method 2: Using Pip (Alternative)

If you prefer using `pip`, you can install the dependencies from the `requirements.txt` file. We recommend doing this within a virtual environment.

1.  **Create a Python virtual environment** (optional but recommended).
    ```bash
    python -m venv color_env
    # On Linux/macOS:
    source color_env/bin/activate
    # On Windows:
    .\color_env\Scripts\activate
    ```

2.  **Install the required Python packages**.
    ```bash
    pip install -r requirements.txt
    ```

### Dataset Preparation

The training and evaluation code expects the dataset to be placed in a specific directory structure.

1.  **Download the dataset** from the official source (link to be provided) and extract it.
2.  **Place the data** in the project's `datasets` directory. The expected structure is as follows:

    ```
    anomalous-trichromat-enhancement/  # Project root
    ├── datasets/
    │   └── color/          # Your dataset folder
    │       ├── train/      # Training data
    │       │   ├── A/      # Contains original images (e.g., 0001.png)
    │       │   └── B/      # Contains corresponding enhanced images (e.g., 0001.png)
    │       └── val/        # Validation data (same A/B structure)
    ├── train.py
    └── ...
    ```

### Verification

To verify that the installation was successful, you can run a simple check:

```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## 📁 Dataset

The dataset used for training and validation can be downloaded from 10.5281/zenodo.17187079


The dataset contains 5,910 original RGB images and their corresponding enhanced versions for three conditions:
- `deutan`: Enhanced for deuteranomaly and deuteranopia.
- `protan_mild`: Mildly enhanced for protanomaly.
- `protan_severe`: Severely enhanced for protanopia and some protanomaly.

Please refer to our paper for detailed construction process.
## 🏋️‍♂️ Training

### Visdom Setup (Optional)
For real-time training visualization, we recommend using Visdom. Start the Visdom server before training:

```bash
# Start Visdom server in a separate terminal
python -m visdom.server
```
Then open http://localhost:8097 in your browser to view training progress.
Basic Training Commands
Train models for different color vision conditions:

```bash
# For green deficiency (deuteranomaly/deuteranopia)
python train.py --dataroot ./datasets/color --name deutan_model --model pix2pix --direction AtoB

# For mild red deficiency (protanomaly)
python train.py --dataroot ./datasets/color --name protan_mild_model --model pix2pix --direction AtoB

# For severe red deficiency (protanopia)
python train.py --dataroot ./datasets/color --name protan_severe_model --model pix2pix --direction AtoB
```
To resume training from a checkpoint, add the --continue_train flag:
```bash
python train.py --dataroot ./datasets/color --name deutan_model --model pix2pix --direction AtoB --continue_train
```
Before starting training, you need to manually set the simulation matrix value in `models/pix2pix_model.py`. The simulation matrix is already provided in the code and should be modified according to your specific requirements.

And to evaluate a trained model, use the test script:

```bash
python test.py --dataroot ./datasets/color --name deutan_model --model pix2pix --direction BtoA
```

## 📜 Citation

If you use this code or our work in your research, please cite:

### Our Paper
```bibtex
@article{he2025enhancing,
  title={Enhancing Color Image for Anomalous Trichromats: A Deep Learning Approach with Detail Preservation},
  author={He, Zihao and Ma, Zejun and Ma, Hai and Zhang, Bingkun and Ma, Ruiqing},
  journal={Journal of Computer Science and Technology},
  year={2025},
  volume={40},
  number={3},
  pages={1--15}
}
```
This project is built upon the pytorch-CycleGAN-and-pix2pix codebase. Please also cite their work:
```bibtex
@article{he2025enhancing,
  title={Enhancing Color Image for Anomalous Trichromats: A Deep Learning Approach with Detail Preservation},
  author={He, Zihao and Ma, Zejun and Ma, Hai and Zhang, Bingkun and Ma, Ruiqing},
  journal={the visual computer},
  note={Submitted},
  year={2024}
}
```
While our Swin Transformer implementation is original, the architectural concept is based on:
```bibtex
@article{liu2021swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```
## 📄 License

This project is licensed under the BSD 2-Clause License - see the [LICENSE](LICENSE) file for full details.
## 🙏 Acknowledgments

This research was supported by the Humanities and Social Science Foundation of the Ministry of Education of China (22YJCZH125).
