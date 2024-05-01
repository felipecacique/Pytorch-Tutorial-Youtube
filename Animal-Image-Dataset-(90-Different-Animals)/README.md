# Animal Image Classification
## Overview
This project focuses on classifying images of animals into 90 different classes using machine learning techniques. It utilizes a dataset of 5400 labeled animal images sourced from [Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals).

## Technologies Used
- Python
- PyTorch
- GPU

## Dataset
The dataset consists of images of various animal species, with each image labeled with the corresponding animal class. This allows for supervised learning.

## Model Architecture
The model architecture is based on a convolutional neural network (CNN) with the EfficientNet_B3 and ConvNeXt models pretrained on the ImageNet dataset as a backbone. The final fully connected layer is replaced with a custom classifier for the specific animal classification task.

## Training Process

- **Data Loading and Preprocessing:** Images are loaded and preprocessed using data augmentation techniques such as random cropping, flipping, and rotation.

- **Model Definition:** The CNN model architecture is defined, consisting of the EfficientNet_B3 or ConvNeXt backbone followed by custom fully connected layers for classification.

- **Loss Function and Optimizer:** Cross-entropy loss function and the Adam optimizer are used for training.

- **Training Loop:** The model is trained over multiple epochs, with batches of images fed into the model for forward pass, loss calculation, and backward pass for gradient descent.

- **Evaluation:** The trained model is evaluated on a separate test set to assess its generalization performance.

## Results
The trained model achieves an accuracy of 90.9% on the test set, indicating its effectiveness in classifying animal images.

## Conclusion
This project demonstrates the use of transfer learning and deep learning techniques for animal image classification, with potential applications in wildlife monitoring, species identification, and conservation efforts.
