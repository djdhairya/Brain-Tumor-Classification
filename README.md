
# 🧠 Brain Tumor Classification 

## Overview
This project focuses on classifying brain tumors using MRI images. It categorizes MRI scans into four distinct classes:
- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

We utilize the **EfficientNetB1** model (pre-trained on ImageNet) and fine-tune it for this classification task. The dataset is sourced from Kaggle and contains labeled MRI scans for training and testing.

---

## Key Features
- **Data Preprocessing**: Includes cropping MRI images to focus on the tumor area, resizing to a uniform size, and applying data augmentation techniques.
- **Model Architecture**: Utilizes `EfficientNetB1` as a feature extractor followed by custom layers for classification.
- **Training Enhancements**: 
  - Early Stopping
  - ReduceLROnPlateau for learning rate adjustment
  - Model Checkpointing to save the best model
- **Visualization**: Displays training/validation accuracy and loss graphs, confusion matrix, and classification reports for evaluation.


---

## Installation & Setup
1. **Clone Repository & Install Dependencies:**
   ```bash
   !pip install opendatasets
   !pip install tensorflow
   !pip install imutils
   !pip install opencv-python
   !pip install matplotlib seaborn scikit-learn
   ```

2. **Download Dataset:**
   ```python
   import opendatasets as od
   od.download("https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data")
   ```

---

## Workflow Summary
1. **Data Exploration & Visualization**
2. **Image Cropping Function**: Focus on brain area by removing extra background.
3. **Preprocessing & Saving Cropped Images** for each class.
4. **Image Augmentation** using `ImageDataGenerator` (Rotation, Shift, Flip).
5. **Model Building**: Fine-tuning EfficientNetB1 with additional layers.
6. **Training** with Early Stopping and LR Scheduler.
7. **Evaluation**: Accuracy, Loss plots, Confusion Matrix, and Classification Report.

---

## Model Performance
- **Evaluation Metrics:**
  - Training Accuracy & Loss
  - Validation Accuracy & Loss
  - Confusion Matrix
  - Classification Report

Graphs and metrics help in understanding the model's learning and generalization capabilities.

---

## Google Colab 🚀
> **Highly Recommended:**  
For **better GPU performance** and faster training, it's advised to run the notebook on **Google Colab**.

You can upload your notebook to [Google Colab](https://colab.research.google.com/) and select **GPU Runtime**:



---

## Folder Structure:
```
├── brain-tumor-mri-dataset/
│   ├── Training/
│   └── Testing/
├── Cropped_img/
│   ├── Train_Data/
│   └── Test_Data/
├── model.keras
├── README.md
└── brain_tumor_classification.ipynb
```

---

## Dependencies
- TensorFlow
- OpenCV
- Matplotlib
- Seaborn
- Imutils
- Scikit-learn
- NumPy
- Pandas

---

## Results
The model achieves excellent classification accuracy on both training and testing datasets, providing a robust solution for brain tumor detection from MRI images.

---

## Future Improvements
- Implement **transfer learning** with deeper EfficientNet variants.
- Integrate attention mechanisms for better focus on tumor regions.
- Hyperparameter tuning for improved performance.
- Deployment of the model as a web application.



---

