# KLA-PS01-Fourmidables-453
Microscopic images used to detect semiconductor fabrication defects suffer from Gaussian noise, speckle, and degraded spatial resolution. This project implements an AI-driven image restoration pipeline using NRSRGAN. The approach jointly eliminates speckle noise and recovers lost high-resolution details from images. 
# i4C x KLA Hackathon 2026 (SEMICON India 2026) — AI-Based Restoration of Degraded Images for Semiconductor Inspection
# Team
|Name|Contact|
|||
|||
|||
|||
College : National Institute of Technology Puducherry
# Problem 
# Approach
# Results

# Real-ESRGAN x2 Image Restoration
## Repository Structure

```text
team_name/
│
├── run.py
├── requirements.txt
├── README.md
│
└── models/
    └── RealESRGAN_x2_trained_generator.pth

# Requirements
Python 3.x
PyTorch
NVIDIA GPU with CUDA support recommended
NumPy
BasicSR

The model weights are included in the models/ directory. No model download or Internet connection is required during inference.

# Input

The program accepts a directory containing .npy files.

Each input is expected to represent a grayscale low-resolution image.

Example:

input/
├── 000001.npy
├── 000002.npy
├── 000003.npy
└── ...
# Output

The program creates the output directory automatically and generates one .npy file for every input file.

The output:

Preserves the input filename.
Is grayscale.
Has shape (H, W).
Contains float32 values.
Contains values in the range [0, 1].
Contains no NaN or Inf values.
Has the required 2× target resolution.

Example:

output/
├── 000001.npy
├── 000002.npy
├── 000003.npy
└── ...

# Model

The restoration network is based on Real-ESRGAN/RRDBNet.

Model configuration:

Input channels : 3
Output channels: 3
Features       : 64
RRDB blocks    : 10
Growth channels: 32
Scale          : ×2

The discriminator is not required during inference.
