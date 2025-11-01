# SIKGDR

## 📦 Setup

### Environment
The conda environment can be constructed with the configuration file `env.yml`:
```bash
conda env create -f env.yml

The codes are tested with CUDA and PyTorch.

Before running the codes, don’t forget to activate the environment:

conda activate SIKGDR

Data

We have provided preprocessed data in raw_data.zip.
Please unzip it before running the training scripts.

Run
1. Reproduce Results
python train_script.py

2. Cold Start Setting
python cold_start.py

Notes

Ensure that CUDA and PyTorch are correctly installed for GPU acceleration.

You can modify hyperparameters or data paths in the scripts if you want to use your own dataset.
