# Brain-tumor-detection-mri-cnn
MRI brain tumor classification (glioma, meningioma, pituitary) using CNN with 97% accuracy — published in IOP Journal

# Overview
A deep learning system that automatically detects and classifies brain tumors (glioma, meningioma, pituitary adenoma) from MRi scans using convolutional Neural Networks (CNNs). Deployed as an interactive Streamlit web application. 
Results: 
<img width="421" height="484" alt="image" src="https://github.com/user-attachments/assets/39d787fd-1c3e-476d-a1f0-a8dc98b121f6" />

#Tech Stack
- Python, TensorFlow/Keras, OpenCV
- CNN Architecture (4 Conv layers, ReLU, Global Average Pooling)
- Streamlit (web deployment)
- Adam Optimizer | Dropout 0.4 | Categorical Cross-Entropy

## 📁 Project Structure
brain-tumor-detection-mri-cnn/
├── model/cnn_model.py
├── preprocessing/preprocess.py
├── app/app.py
├── results/confusion_matrix.png
└── requirements.txt

## 📄 Publication
Co-authored paper submitted to IOP Journal (2025)
Authors: Venkata Rao Y., Ganja V., Biradar R., Mallreddygari A., Shifa Naaz
