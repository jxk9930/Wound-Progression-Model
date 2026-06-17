# 🩹 Wound-Progression-Model

🏆 **3rd Prize Winner at the Texas Healthcare Challenge Hackathon (TXHCC)**

This repository contains a biomechanics-based model designed to estimate wound progression, specifically targeting **Diabetic Foot Ulcers (DFU)**. 

## 📌 Overview
The core of this model relies on biomechanical equations to simulate and predict the healing or worsening of foot ulcers. To automate and enhance the precision of the initial inputs, the system integrates robust pre-trained computer vision models for classification and segmentation.

## 🧩 Integrated Pre-trained Models
The workflow utilizes the following models to process the input data:
* **Image Classification:** `dqnguyen/Diabetic_Foot_Ulcer_Image_Classification`
* **Foot Segmentation:** `roboflow/Foot Segmentation Computer Vision Model`
* **Ulcer Segmentation:** `masih4/Foot_Ulcer_Segmentation`

## 📊 Progression Simulation Results
*(The progression of wound healing and worsening based on our biomechanical model)*
<img width="1990" height="468" alt="Healing_case" src="https://github.com/user-attachments/assets/e6c1848f-c612-42f7-a732-395937048128" />
<img width="1990" height="450" alt="Worsening_case" src="https://github.com/user-attachments/assets/9b57703e-07c8-49f1-95b0-808e9637dadd" />
