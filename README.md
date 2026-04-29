# Waste Segregation Using Deep Learning: Sample Efficiency of Object Detectors

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-brightgreen)](https://github.com/ultralytics/ultralytics)
[![Detectron2](https://img.shields.io/badge/Detectron2-Faster%20R--CNN-blueviolet)](https://github.com/facebookresearch/detectron2)

**This repository contains the official implementation and benchmark results for the paper:**  
*"Waste Segregation Using Deep Learning: Sample Efficiency of Object Detectors on a Small Waste Dataset"*

> **Key Finding:** Fine‑tuning a pre‑trained two‑stage detector (Faster R‑CNN) is significantly more sample‑efficient than training YOLOv8 variants from scratch when data and training epochs are extremely limited (10 epochs, 1,200 images). YOLOv8n achieves the lowest CPU latency (~100 ms/image) but requires ≥50 epochs or COCO pre‑training to become useful.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Models Evaluated](#models-evaluated)
- [Results Summary](#results-summary)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Model Selection Guide](#model-selection-guide)
- [Ethical Considerations](#ethical-considerations)
- [Limitations & Future Work](#limitations--future-work)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 📖 Overview

Automated waste segregation can reduce pollution and improve recycling efficiency. This project benchmarks **four object detection architectures** on a subset of the **TACO dataset** (1,500 images, 60 waste categories) under **identical, constrained training conditions** (10 epochs, no pre‑training for YOLO variants). The goal is to measure **sample efficiency** — how well each model learns from very limited data — rather than to claim general architectural superiority.

**Models compared:**
- Faster R‑CNN (fine‑tuned from ImageNet) – two‑stage detector
- YOLOv8s (small) – trained from scratch
- YOLOv8m (medium) – trained from scratch
- YOLOv8n (nano) – trained from scratch

**Key contributions:**
- Pareto analysis of speed (CPU latency) vs. accuracy (mAP50)
- Deployment decision tree for edge vs. server applications
- Explicit ethical discussion (bias, energy cost, safety)
- Reproducible training/evaluation code

---

## 📂 Dataset

We use a subset of the **TACO** (Trash Annotations in Context) dataset. The subset contains **1,500 images** with bounding boxes for 60 waste categories (plastic, paper, glass, metal, organic, etc.).

| Split        | Images | Percentage |
|--------------|--------|------------|
| Train        | 1,200  | 80%        |
| Validation   | 150    | 10%        |
| Test         | 150    | 10%        |

**Download links:**
- Official website: [http://tacodataset.org/](http://tacodataset.org/)
- GitHub toolkit: [https://github.com/pedropro/TACO](https://github.com/pedropro/TACO)
- Kaggle mirror: [https://www.kaggle.com/kneroma/tacotrashdataset](https://www.kaggle.com/kneroma/tacotrashdataset)
- Zenodo archive (citable): [https://zenodo.org/records/3354286](https://zenodo.org/records/3354286)

After downloading, convert annotations to YOLO format using the provided script `convert_taco_to_yolo.py` (see `scripts/` folder).

---

## 🧠 Models Evaluated

| Model          | Paradigm       | Parameters | Pre‑training      | Training epochs |
|----------------|----------------|------------|-------------------|-----------------|
| Faster R‑CNN   | Two‑stage      | ~25M       | ImageNet (fine‑tuned) | 10              |
| YOLOv8s        | Single‑stage   | 11.2M      | None (from scratch) | 10              |
| YOLOv8m        | Single‑stage   | 22.6M      | None (from scratch) | 10              |
| YOLOv8n        | Single‑stage   | 3.2M       | None (from scratch) | 10              |

> **Important:** YOLO models were deliberately trained **from scratch** to test sample efficiency. In practice, use COCO pre‑training + fine‑tuning for YOLO to achieve competitive accuracy.

---

## 📊 Results Summary

**Test set (150 images)**

| Model          | mAP50 | mAP50-95 | Precision | Recall | CPU Speed (ms/img) | FPS (CPU) |
|----------------|-------|----------|-----------|--------|--------------------|------------|
| Faster R‑CNN   | 0.450 | 0.280    | 0.660     | 0.600  | ~1450              | ~0.7       |
| YOLOv8s        | 0.083 | 0.056    | 0.677     | 0.069  | ~175               | ~5.7       |
| YOLOv8m        | 0.007 | 0.005    | 0.008     | 0.061  | ~300               | ~3.3       |
| YOLOv8n        | 0.007 | 0.005    | 0.005     | 0.032  | ~100               | ~9.0       |

