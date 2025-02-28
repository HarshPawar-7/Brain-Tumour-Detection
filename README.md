# Brain-Tumor-Detection-Using-Deep-Learning

## Achieved Accuracy upto 98.7% ##
### for 4 types of tumours ###

## Overview
This project implements a deep learning model for detecting brain tumors using MRI images. It leverages transfer learning with the VGG16 model to classify MRI scans into four categories: glioma, meningioma, pituitary, and no tumor. The model is trained using TensorFlow and Keras on a labeled dataset and achieves high accuracy in detecting brain tumors.

## Dataset
The dataset consists of MRI images organized into training and testing sets, stored in Google Drive. It includes four classes:
- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

## Installation
Follow these steps to set up the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Brain-Tumor-Detection.git
   cd Brain-Tumor-Detection
   ```
2. Install dependencies:
   ```bash
   pip install tensorflow numpy matplotlib seaborn scikit-learn pillow
   ```
3. Mount Google Drive (for Colab users):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

## Model
The model is built using **transfer learning** with **VGG16**. The key steps are:
- Load the **VGG16** model (pre-trained on ImageNet) with `include_top=False`
- Freeze all layers except the last few to fine-tune the model
- Add custom **fully connected layers** and **dropout layers** to reduce overfitting
- Compile the model using **Adam optimizer** and **sparse categorical cross-entropy** loss

## Training
The model is trained using **data generators** with augmentation. Training parameters:
- **Batch size:** 20
- **Epochs:** 5
- **Optimizer:** Adam
- **Loss function:** Sparse categorical cross-entropy

Run the training process:
```python
history = model.fit(datagen(train_paths, train_labels, batch_size=20, epochs=5),
                    epochs=5, steps_per_epoch=len(train_paths) // 20)
```

## Evaluation
After training, the model is evaluated on the test dataset:
```python
test_predictions = model.predict(test_images)
print(classification_report(test_labels_encoded, np.argmax(test_predictions, axis=1)))
```
It generates metrics such as **accuracy, precision, recall, and F1-score**.

## Visualization
### Training Performance
```python
plt.plot(history.history['sparse_categorical_accuracy'], label='Accuracy')
plt.plot(history.history['loss'], label='Loss')
plt.legend()
plt.show()
```

### Confusion Matrix
```python
sns.heatmap(confusion_matrix(test_labels_encoded, np.argmax(test_predictions, axis=1)), annot=True, cmap="Blues")
plt.show()
```

### ROC Curve
```python
for i in range(len(class_labels)):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
plt.legend()
plt.show()
```

## Testing the Model
You can test the model by providing an MRI image path:
```python
image_path = 'path/to/mri/image.jpg'
detect_and_display(image_path, model)
```
This function loads an image, preprocesses it, and displays the predicted tumor type with confidence score.

## Saving and Loading the Model
To save the trained model:
```python
model.save('model.h5')
```
To load the saved model:
```python
from tensorflow.keras.models import load_model
model = load_model('model.h5')
```

## Acknowledgments
- The dataset is sourced from **Kaggle** and publicly available MRI scans.
- The **VGG16** model is used for transfer learning.
- The project is implemented using **TensorFlow, Keras, and Python**.
  

## License
This project is licensed under the **MIT License**.

---
For more details, check the **code files** in the repository!


