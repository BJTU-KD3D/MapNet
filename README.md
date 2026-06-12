# MapNet: Mamba Prompt Network for Weakly-supervised Point Cloud Semantic Segmentation

MapNet is a point cloud semantic segmentation framework that couples a state-space (Mamba) backbone with geometry-aware space-filling curve serialization. It is designed for weak supervision, where only a small fraction of points carry annotations, and it targets both indoor (S3DIS, ScanNetv2) and large-scale real-world (STPLS3D) scenes.


## Architecture Overview

MapNet follows an encoder-decoder design built from three core modules that are each stacked 5 times:

**Input point cloud → SSD (5×) → MPE (5×) → ⊕ → GCD (5×) → Output**


## Highlights
1. A Mamba-based encoder (**MPE**) that combines linear-complexity state-space modeling with geometry-aware prompts to capture long-range context in point clouds.
2. Multiple enhanced **space-filling curve** serializations (density / geometry / hierarchical / circular) that are ensembled to handle uneven density and diverse scene structures.
3. A geometry-constrained encoder-decoder (**SSD** + **GCD**) tailored for weak supervision, achieving strong performance on indoor and real-world benchmarks.

# Get Started
## Dependencies
- Ubuntu: 20.04
- Python: 3.7
- PyTorch: 1.10.1 
- CUDA: 11.3
- Hardware: NVIDIA RTX 3080 (or higher)

## Environment

1. Install dependencies

```
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html

pip install -r requirements.txt
```

2. Compile pointops

Make sure you have installed `gcc` and `cuda`, and `nvcc` can work (Note that if you install cuda by conda, it won't provide nvcc and you should install cuda manually.). Then, compile and install pointops as follows.
```
cd lib/pointops
python3 setup.py install
```

## Datasets Preparation

### S3DIS
Please refer to https://github.com/yanx27/Pointnet_Pointnet2_pytorch for S3DIS preprocessing. Then modify the `data_root` entry in the .yaml configuration file.

### ScanNetv2
Please refer to https://github.com/dvlab-research/PointGroup for the ScanNetv2 preprocessing. Then change the `data_root` entry in the .yaml configuration file accordingly.

### STPLS3D
Please refer to https://github.com/meidachen/STPLS3D/tree/main/point-transformer for STPLS3D preprocessing. Then modify the `data_root` entry in the .yaml configuration file.

## Training
First check the `save_path`, `resume` and `labeled_point` (if applicable) accordingly. Then, run the following command.

[CONFIG] = config/s3dis.yaml, config/scannet.yaml, config/stpls3d.yaml
```
python3 train.py --config config/s3dis.yaml
```

Note: It is normal to see the results on S3DIS fluctuate between -0.5\% and +0.5\% mIoU maybe because the size of S3DIS is relatively small, while the results on ScanNetv2 and STPLS3D are relatively stable.


## Acknowledgement
This codebase builds upon [PointCT](https://github.com/POSTECH-CVLab/point-transformer), [Point Transformer](https://github.com/POSTECH-CVLab/point-transformer), [Stratified Transformer](https://github.com/dvlab-research/Stratified-Transformer), and the [Mamba](https://github.com/state-spaces/mamba) state-space model.
