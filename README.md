# Monocular Depth Estimation using a U-Net with MobileNetV2 Backbone

This project was developed as part of **CSE 575: Statistical Machine Learning (2025 Spring C)** at **Arizona State University**.

---

## Project Overview

This project focuses on monocular depth estimation using a custom U-Net architecture. The encoder is based on **MobileNetV2**, chosen for its efficiency and performance, and has been integrated into the U-Net framework. The decoder was modified accordingly to ensure proper upsampling of feature maps.

To enhance depth prediction accuracy, I experimented with a composite loss function combining **L1 loss**, **Structural Similarity Index (SSIM)**, and **edge-aware loss**. However, after evaluating the results, the edge-aware component was excluded due to suboptimal performance. The final model uses a **weighted combination of L1 and SSIM losses** to balance pixel-level accuracy with perceptual quality.

---

## Sample Results

*A sample of the model's output is provided below to illustrate the results.*

<!-- Row 1 -->
<p float="left">
  <img src="test_vid/video1.gif" width="45%" />
  <img src="test_vid/depth1.gif" width="45%" />
</p>

<!-- Row 2 -->
<p float="left">
  <img src="test_vid/video2.gif" width="45%" />
  <img src="test_vid/depth2.gif" width="45%" />
</p>

## Repository Structure
```bash
├── data/             # Folder to store dataset
  ├── nyu2_test/
  ├── nyu2_train/
  ├── nyu2_test.csv
  ├── nyu2_train.csv
├── test_vid/         # Stores the frames, videos and gifs generated from test_video.ipynb
  ├── frames/
  ├── depth_frames/
├── train.py          # Code for training the model
├── test.py           # Code for generating an output and storing an image of the output, image and expected depth map
├── validation.py     # Code for testing with the h5 files in the validation set of NYU Depth V2 dataset
└── README.md         # Project overview
```

## Training
```bash
python train.py --checkpath ./checkpoints/ --run $run_num --epoch $epoch_num
```

## Validation
```bash
python validation.py --checkpath ./checkpoints/ --val_path $validation_data_path
```

## Generating Visual Outputs for Comparison
```bash
python test.py --checkpath ./checkpoints/ --checkpoint $last_run_epoch_num --batch $num_of_images
```
