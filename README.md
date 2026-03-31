# Bird's-Eye-View (BEV) 2D Occupancy Transformer

This repository contains the Round 1 Proof of Concept (PoC) codebase for **Problem Statement 3: Bird's-Eye-View (BEV) 2D Occupancy**.

> model link:
``` https://drive.google.com/file/d/1ZYaHg2k7ys3Y5Jzk7s_JNuwgKzlniJWV/view?usp=drive_link```

## Project Overview
Autonomous L4 vehicles cannot rely solely on perspective-distorted front-facing cameras for path planning; they require an accurate top-down 2D "Occupancy Grid." Standard geometric warping (like Inverse Perspective Mapping) stretches 3D obstacles (like poles or cars) across the ground plane. 

This project solves this by implementing a custom View-Transformer neural network. It mathematically maps 2D visual features from a front-facing camera into a precise 3D top-down perspective without spatial stretching, outputting a binary grid of "Free Space" vs. "Occupied Space."

## Model Architecture
Our PyTorch architecture follows a three-stage "Lift and Splat" inspired pipeline:
1. **Feature Extraction (Backbone):** A lightweight `ResNet18` CNN processes the raw `224x480` front camera images to extract deep spatial features efficiently.
2. **Perspective Transformation (View Transformer):** The 2D spatial features are flattened and passed through a Learned Spatial Mapper (Linear layer mapping 105 image features to 6400 grid features). This module implicitly learns camera depth and projects the visual plane directly into a top-down perspective.
3. **Occupancy Decoder:** A series of Convolutional and BatchNorm layers refine the transformed features into a final `80x80` binary occupancy grid representing a 40m x 40m physical area.

## Dataset Used
* **Dataset:** Official `nuScenes v1.0-mini` database.
* **Ground Truth Generation:** We engineered a custom PyTorch `Dataset` class that ingests 3D LiDAR point clouds and camera intrinsic matrices, discretizing the physical points into a 2D 80x80 binary tensor to serve as the training target.

## Setup & Installation Instructions
Kaggle notebook `https://www.kaggle.com/code/shashankssvision/hackathon-mit`
accelerator - GPU T4 x2

## How to Run the Code
1. Open attached `BEV_Occupancy_PoC.ipynb` in Kaggle.
2. Ensure the `DATAROOT` variable in the notebook points to your Kaggle dataset path (e.g., `/kaggle/input/nuscenes-mini/v1.0-mini`).
3. **Run All Cells:** The notebook will automatically initialize the dataset, extract the LiDAR ground truth, instantiate the PyTorch model, run the training loop, and output the evaluation metrics.

*(Note: You can load the included `bev_occupancy_model.pth` file to bypass training and jump straight to inference).*

## Example Outputs / Results
During initial Proof of Concept testing, the model successfully learned the perspective transformation mapping, achieving a strong **Batch Mean Occupancy IoU**.
![input](https://github.com/shashankssvi/Bit_and_Bytes_BEV/blob/main/img1?raw=true)

![output](https://github.com/shashankssvi/Bit_and_Bytes_BEV/blob/main/img2?raw=true)

![output](https://github.com/shashankssvi/Bit_and_Bytes_BEV/blob/main/img3?raw=true)

![output](https://github.com/shashankssvi/Bit_and_Bytes_BEV/blob/main/img4?raw=true)

![Confusion Materx](https://github.com/shashankssvi/Bit_and_Bytes_BEV/blob/main/confusionMatrix?raw=true)
*(Note to evaluator: Please see the repository images for side-by-side comparisons of the Raw Camera Input, LiDAR Ground Truth, and the Model's predicted Occupancy Heatmap, alongside the Pixel-wise Confusion Matrix).*
