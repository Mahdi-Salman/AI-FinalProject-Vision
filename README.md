# Autonomous Driving Object Detection: YOLOv8 End-to-End Pipeline

## Table of Contents
* [📌 Project Overview](#-project-overview)
* [🚀 Key Features & Achievements](#-key-features--achievements)
* [📂 Project Phases & Model Evolution](#-project-phases--model-evolution)
* [⬇️ Download Pre-trained Weights](#️-download-pre-trained-weights)
* [🛠️ Prerequisites & Tech Stack](#️-prerequisites--tech-stack)
* [⚙️ Installation & Setup](#️-installation--setup)
* [🐳 Docker Usage Guide (Step-by-Step)](#-docker-usage-guide-step-by-step)
* [📁 Repository Structure](#-repository-structure)

---

## 📌 Project Overview
This repository contains a comprehensive, production-ready Deep Learning pipeline for Object Detection in autonomous driving environments. Developed as a final university project, it leverages the state-of-the-art **YOLOv8** architecture to detect and classify vehicles, pedestrians, and cyclists using the highly complex **KITTI Vision Benchmark Suite**.

The project is structured into two main phases, evolving from raw data preparation to a fully containerized, interactive web application.

---

## 🚀 Key Features & Achievements
* **Complete MLOps Pipeline**: End-to-end workflow covering data preprocessing, model training, evaluation, and deployment, fully orchestrated with Docker Compose.
* **Tackling Class Imbalance**: Successfully addressed the scarcity of minority classes (Pedestrians and Cyclists) by upgrading to a deeper architecture (YOLOv8s) and applying advanced augmentation techniques (Mosaic 1.0, Mixup 0.1).
* **Exceptional Accuracy**: Achieved a final **mAP50 of 89.5 percent** on a strictly separated and unseen Test Set.
* **Interactive Web Dashboard**: Features a real-time Streamlit user interface that allows users to upload custom images/videos and dynamically adjust confidence thresholds.
* **Hardware Optimized**: Fully configured for NVIDIA GPUs (CUDA) within isolated Linux Docker containers.

---

## 📂 Project Phases & Model Evolution

### Phase 1: Data Engineering & Preparation (KITTI Dataset)
The KITTI dataset initially comes in a format incompatible with YOLO architectures. In this phase:
* Raw images and labels were downloaded, cleaned, and normalized.
* Bounding box coordinates were converted from KITTI format (pixel values) to YOLO format (normalized center coordinates).
* The dataset was strictly split into Training, Validation, and Test sets to prevent data leakage.

### Phase 2: Model Evolution, Evaluation, and Deployment
In this phase, we adopted a scientific, iterative approach to model training. Rather than deploying a single model, we trained three separate iterations to thoroughly understand the network behavior, diagnose bottlenecks, and engineer specific solutions for autonomous driving challenges.

**1. Baseline Model (YOLOv8 Nano - Trained from Scratch)**
* **Approach**: Initialized with random weights without any prior knowledge of object shapes.
* **Analysis**: The model struggled to extract meaningful spatial features from the limited size of the KITTI dataset. 
* **Result**: Resulted in severe underfitting, plateauing at a mere **34 percent mAP50**. It failed to detect smaller objects like pedestrians and frequently confused background elements with vehicles.

**2. Primary Optimized Model (YOLOv8 Nano - Transfer Learning)**
* **Approach**: Integrated pre-trained weights to provide a foundational understanding of shapes, edges, and objects.
* **Analysis**: Performance saw a massive initial leap, reaching **85 percent mAP50**. However, rigorous real-world testing revealed a critical engineering flaw: **Domain Shift and Overfitting**. The network memorized the specific backgrounds, street textures, and camera angles of the German streets in the KITTI dataset rather than learning the abstract concept of the vehicles themselves.
* **Minority Class Drop**: The model aggressively filtered out pedestrians and cyclists in dense environments to artificially boost its confidence, leading to high False Negatives.

**3. Advanced Final Model (YOLOv8 Small - Deep Architecture + Augmentation)**
* **Approach**: Upgraded the architecture to a deeper YOLOv8s network (over 11 million parameters) to increase learning capacity.
* **Engineering Data**: We applied aggressive data manipulation using **Mosaic (1.0)** and **Mixup (0.1)**. By stitching four images together with varying transparencies, the model was forced into a visual "torture test" where it could no longer rely on street backgrounds. It was forced to learn the exact geometric shapes of the objects. We also implemented Early Stopping (patience=10) to halt training at the exact peak of generalization.
* **Final Result**: A complete engineering victory. The model hit **89.5 percent mAP50** on unseen test data. The minority class detection skyrocketed, bringing Pedestrian accuracy to 85.9 percent and Cyclist accuracy to 87.9 percent, creating a perfectly balanced detection pipeline ready for deployment.

---

## ⬇️ Download Pre-trained Weights
To skip the training phase and directly test the Advanced Model or run the UI, you can download the best weights generated from our final training cycle:

* 📦 [Download best.pt (Advanced YOLOv8s)](https://github.com/Mahdi-Salman/AI-FinalProject-Vision/releases/download/v2.2-advanced-model-eval/best.pt)

*(Place this downloaded file inside the designated weights directory before running the Evaluator or UI services).*

---

## 🛠️ Prerequisites & Tech Stack
Before running the project, ensure your host machine (preferably Ubuntu Linux) has the following installed:
* **Python** 3.10+
* **Docker** and **Docker Compose**
* **NVIDIA GPU** (Tested on RTX 4060)
* **NVIDIA Container Toolkit** (Crucial for GPU pass-through to Docker)

---

## ⚙️ Installation & Setup

Clone the repository to your local machine:

```bash
git clone https://github.com/Mahdi-Salman/AI-FinalProject-Vision.git
cd AI-FinalProject-Vision
```

Verify that your formatted KITTI dataset is placed in the `datasets/` directory at the root of the project.

---

## 🐳 Docker Usage Guide (Step-by-Step)

The entire architecture is decoupled into distinct services using `docker-compose.yml` to prevent resource conflicts. You do not need to install heavy AI libraries on your host machine; Docker handles everything.

### 1. Training the Model (Trainer Service)
To train the YOLOv8s model from scratch using the defined hyperparameters, data augmentation, and GPU acceleration, run:

```bash
docker compose up trainer --build
```

> **Note:** This process will download the pre-trained weights automatically, allocate the GPU, and save the best weights inside the `runs/detect/train/` directory upon completion.

### 2. Evaluating the Model (Evaluator Service)
To test the trained model against the unseen Test Set and generate evaluation metrics (Confusion Matrix, PR-Curves, F1-Scores), execute:

```bash
docker compose up evaluator --build
```

> **Note:** The output graphs and predictions will be saved automatically, demonstrating the model accuracy on edge cases.

### 3. Running the Interactive Dashboard (UI Service)
To deploy the model into a user-friendly web interface for real-time inference, run:

```bash
docker compose up ui --build
```

* Wait for the container to initialize.
* Open your web browser and navigate to: `http://localhost:8501`
* **Features:** You can upload any traffic image or video. Use the slider on the sidebar to adjust the Confidence Threshold dynamically and see how the model reacts to dense environments in real-time.

---

## 📁 Repository Structure

```text
├── assets/                  # Visualized metrics, F1/PR curves, and confusion matrices
├── datasets/                # The preprocessed KITTI dataset 
├── runs/                    # Auto-generated directory containing training weights and logs
├── src/
│   ├── train.py             # Logic for model training and augmentation
│   ├── evaluate.py          # Logic for model testing and metric 
│   └── ui/                  # Streamlit frontend architecture and 
├── docker-compose.yml       # Orchestration file for Trainer,          
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---
