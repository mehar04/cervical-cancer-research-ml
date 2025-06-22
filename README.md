# cervical-cancer-research-ml
🧠 Prediction of Cervical Cancer with Machine Learning Approaches

📄 Overview

This repository contains the source code, methodology, and dataset details for our research paper:

*"Prediction of Cervical Cancer with Machine Learning Approaches"*  
Published in: Journal of the Institution of Engineers (India): Series B, 2025

🎯 Abstract

Cervical cancer remains one of the leading causes of mortality among women worldwide. This study explores the application of machine learning (ML) and deep learning (DL) techniques, including InceptionV3 and Support Vector Machines (SVM), for the early detection and classification of cervical cancer using Pap smear images. Our proposed hybrid model achieves *99.1% precision*, significantly outperforming traditional models.

 🧪 Methods Used

- *Transfer Learning* with InceptionV3 for feature extraction  
- *Support Vector Machine (SVM)* for classification  
- Other models for comparison:
  - Logistic Regression (LR)
  - K-Nearest Neighbors (KNN)
  - Convolutional Neural Networks (CNN)
  - Bagging Decision Trees (Bagging DT)
  - Ensemble Models

🧬 Datasets Used

Three publicly available cervical cancer datasets were used in this work, including:
- A large-scale Pap smear image dataset
- A 224×224 cervical screening dataset
- A medical dataset sourced from Mendeley

These datasets enabled robust model training, validation, and testing.

 🧰 Requirements

- Python 3.7+
- TensorFlow / Keras
- scikit-learn
- NumPy
- OpenCV
- Matplotlib

  
📊 RESULTS
| Model               | Precision | Recall | F1-score |
| ------------------- | --------- | ------ | -------- |
| InceptionV3 + SVM   | 1.00      | 1.00   | 1.00     |
| CNN                 | 0.94      | 0.77   | 0.83     |
| Logistic Regression | 0.77      | 0.72   | 0.74     |
| KNN                 | 0.89      | 0.65   | 0.69     |



🧠Highlights
->Direct use of Pap smear images without segmentation
->Hybrid DL-ML model with optimized feature extraction
->Use of data augmentation for better generalization
->Results show high robustness and reliability

👨‍💻 Authors
Mehar Khurana
Aditya Pandey
Jatin Arora
Gursamrath Singh
Neha Sharma
Neeru Jindal

📌 Citation
(If you use this work, please cite:)

Pandey, A., Arora, J., Kohli, G. S. R., Khurana, M., Sharma, N., & Jindal, N. (2025).  
**Prediction of Cervical Cancer with Machine Learning Approaches.**  
*Journal of The Institution of Engineers (India): Series B*. Springer.  
https://doi.org/10.1007/s40031-025-01248-7
