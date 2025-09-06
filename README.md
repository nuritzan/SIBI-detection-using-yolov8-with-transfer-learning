# Indonesian Sign Language Alphabet Detection using YOLOv8 with Transfer Learning

## 📖 Overview
This project focuses on real-time detection of the Indonesian Sign Language (SIBI) alphabet using YOLOv8 with transfer learning. The system is designed to recognize hand gestures corresponding to the SIBI alphabet and provide accurate detection results in real time.  

## 🛠️ Tech Stack
- Python  
- Google Colab (training)  
- PyCharm (real-time testing)  
- Roboflow (dataset annotation & preprocessing)  
- YOLOv8 (Ultralytics)  

## 📊 Dataset
- Source: [SIBI Dataset](https://www.kaggle.com/datasets/alvinbintang/sibi-dataset)
- Total images: 5,280
- Classes: 24 alphabet classes (Excluding J and Z)  
- Preprocessing steps:  
  - Image resizing to 640×640  
  - Grayscale conversion  
  - Normalization  
- Dataset split: 80% training, 10% validation, 10% testing

## 🚀 Model Training
- Base model: YOLOv8s
- Optimizer: AdamW
- Epochs: 100
- Loss function: box loss, classification loss, and DFL
- Training environment: Google Colab (GPU)  

## 📈 Results
- mAP@0.5: 99.5%
- mAP@[0.5:0.95]: 98.02%
- Precision & Recall: 100% across all classes
- Average Inference Speed: ~300ms per frame  
- Real-time detection confidence: > 0.8

## 🎥 Demo
The model was tested using a webcam for real-time recognition. Detection proved effective under proper lighting and uncluttered backgrounds, confirming the system’s potential for assistive communication technologies.
