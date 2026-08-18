# KLA-PS01-Fourmidables-453
Microscopic images used to detect semiconductor fabrication defects suffer from Gaussian noise, speckle, and degraded spatial resolution. This project implements an AI-driven image restoration pipeline using NRSRGAN. The approach jointly eliminates speckle noise and recovers lost high-resolution details from images. 
# i4C x KLA Hackathon 2026 (SEMICON India 2026) — AI-Based Restoration of Degraded Images for Semiconductor Inspection
# Team
|Name|Contact| <br>
|Jayanthi B|jayanthib1995@yahoo.com| <br>
|Anitha S|ssanithasvce@gmail.com| <br>    
|Selva Sundary M|selvasundary26@gmail.com|<br>
|Giridharan M |giridharan7482@gmail.com| <br>

College : National Institute of Technology Puducherry
# Problem 
Challenge : Semiconductor inspection relies on microscopic images to identify critical defects. Speckle/Gaussian noise and 2×/4× spatial resolution loss can obscure fine structures, edges and defect-level details, reducing inspection reliability. Hence, restoring degraded semiconductor inspection images with high structural fidelity — reducing noise and recovering resolution without compromising critical defect details is essential.

# Real-World Impact 
A small loss of structural information or a single corrupted region can make critical defects difficult to distinguish, potentially affecting inspection accuracy, quality control and manufacturing yield. This AI based solution for restoration helps increase the yield and quality of the IC. 

## Project Structure

```text
Fourmidables-453/
├── run.py
├── requirements.txt
├── README.md
└── models/
    └── RealESRGAN_x2_trained_generator.pth
```
## Setup 
```text
%cd /content

!git clone https://github.com/xinntao/Real-ESRGAN.git

%cd /content/Real-ESRGAN
```
## Installation Files
```text
!pip install -q basicsr facexlib gfpgan
!pip install -q -r requirements.txt
!python setup.py develop
```
## run
```text
python run.py <input-dir> <output-dir>
```
## Performance Comparison of RRDB Blocks

| **RRDB BLOCKS** | **TOTAL PARAMETERS** | **EPOCH** | **TOTAL ITERATIONS** | **PSNR (dB)** | **SSIM** |
|:---------------:|:--------------------:|:---------:|:--------------------:|:-------------:|:--------:|
| 23 | 16,703,171 | 10 | 6400 | 35.11 | 0.9454 |
| 20 | 14,376,897 | 10 | 6400 | **35.34** | 0.9343 |
| **10** | **7,350,654** | 10 | 6400 | **35.03** | **0.9379** |
| 5 | 3,753,539 | 10 | 6400 | 34.83 | 0.9169 |
All the models were trained using 2560 (80%) images for training and 640 (20%) images for testing. The PSNR and SSIM are calculated for the testing images. The RRDB with 23 blocks is considered as the baseline model. The NRSRGAN model with 20, 10, and 5 RRDB blocks achieves 13.93%, 55.99%, and 77.53% reductions in the number of parameters, respectively. The corresponding PSNR changes are a 0.66% increase, 0.23% decrease, and 1.60% decrease, while SSIM decreases by 1.17%, 0.79%, and 1.59%, respectively, compared with the 23-RRDB model. From the above results the model with 10 RRBD blocks performs better in terms of PSNR, SSIM and less number of parameters
