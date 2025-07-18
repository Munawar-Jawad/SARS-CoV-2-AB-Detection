import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Function to load and preprocess an image
def load_and_preprocess_image(file_path):
    try:
        image = cv2.imread(file_path)
        if image is not None:
            resized_image = cv2.resize(image, (256, 256))
            resized_image = resized_image / 255.0  # Normalize pixel values
            return resized_image
        else:
            print(f"Failed to load image: {file_path}")
            return None
    except Exception as e:
        print(f"Failed to load image: {file_path}\n{e}")
        return None

# Function to preprocess a dataset
def preprocess_dataset(file_paths, labels):
    resized_images = []
    resized_labels = []
    for path, label in zip(file_paths, labels):
        img = load_and_preprocess_image(path)
        if img is not None:
            resized_images.append(img)
            resized_labels.append(label)
    return resized_images, resized_labels

# Load dataset
data = pd.read_csv('C:\AGGLUTINATION\Image\Multilevel_classification_dataset.csv') #Insert dataset filepath here

# Encode labels
label_encoder = LabelEncoder()
data['labels'] = label_encoder.fit_transform(data['labels'])

# Split into train/test
file_paths = data['Filepath'].tolist()
labels = data['labels'].tolist()
X_train, X_test, y_train, y_test = train_test_split(
    file_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

# Preprocess training and testing datasets
x_train_resized_images, x_train_resized_labels = preprocess_dataset(X_train, y_train)
x_test_resized_images, x_test_resized_labels = preprocess_dataset(X_test, y_test)

# Flatten images
X_train_flat = [img.flatten() for img in x_train_resized_images]
X_test_flat = [img.flatten() for img in x_test_resized_images]

# Train SVM
svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(X_train_flat, x_train_resized_labels)

# Evaluate on training set
y_train_pred = svm_classifier.predict(X_train_flat)
train_accuracy = accuracy_score(x_train_resized_labels, y_train_pred)
print(f"Train Accuracy: {train_accuracy:.2f}")

# Evaluate on test set
y_test_pred = svm_classifier.predict(X_test_flat)
test_accuracy = accuracy_score(x_test_resized_labels, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Cross-validation
print("Performing cross-validation...")
cv_scores = cross_val_score(svm_classifier, X_train_flat, x_train_resized_labels, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# Classification report
print("Classification Report:\n", classification_report(x_test_resized_labels, y_test_pred))

# Confusion matrix
conf_matrix = confusion_matrix(x_test_resized_labels, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x=label_encoder.inverse_transform(data['labels']))
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.xticks(rotation=45)
plt.show()

# Display incorrect predictions
incorrect_predictions = [
    (X_test[i], x_test_resized_labels[i], y_test_pred[i])
    for i in range(len(x_test_resized_labels))
    if y_test_pred[i] != x_test_resized_labels[i]
]

for i, (image_path, actual_label, predicted_label) in enumerate(incorrect_predictions[:5]):
    new_image = cv2.imread(image_path)
    if new_image is not None:
        resized_image = cv2.resize(new_image, (256, 256))

        # #  Debug print
        # print("BGR pixel [0,0]:", resized_image[0, 0])
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        # print("RGB pixel [0,0]:", rgb_image[0, 0])

        #  Label names
        actual_name = label_encoder.inverse_transform([actual_label])[0]
        predicted_name = label_encoder.inverse_transform([predicted_label])[0]

        #  Display RGB image
        plt.figure(figsize=(4, 4))
        plt.imshow(rgb_image)
        plt.title(f'Actual: {actual_name}, Predicted: {predicted_name}')
        plt.axis('off')
        plt.show()

    else:
        print(f"Failed to load the image: {image_path}")

# Predict on new images
new_image_paths = [
    r'H:\AGGLUTINATION\Image\crop Weak\W21.jpg',
    r'H:\AGGLUTINATION\Image\crop Weak\W22.jpg',
    r'H:\AGGLUTINATION\Image\crop Strong\S61.jpg',
    r'H:\AGGLUTINATION\Image\crop Strong\S62.jpg',
    r'H:\AGGLUTINATION\Image\crop Moderate\M26.jpg',
    r'H:\AGGLUTINATION\Image\crop Moderate\M27.jpg',
    r'H:\AGGLUTINATION\Image\crop Moderate\M28.jpg',
    r'H:\AGGLUTINATION\Image\crop Moderate\M29.jpg',
    r'H:\AGGLUTINATION\Image\crop Moderate\M30.jpg',
    r'H:\AGGLUTINATION\Image\crop Moderate\M31.jpg',
    r'H:\AGGLUTINATION\Image\crop Moderate\M32.jpg',
    r'H:\AGGLUTINATION\Image\crop Negative\N34.jpg',
    r'H:\AGGLUTINATION\Image\crop Negative\N35.jpg',
    r'H:\AGGLUTINATION\Image\crop Negative\N36.jpg',
    r'H:\AGGLUTINATION\Image\crop Negative\N78.jpg',
    r'H:\AGGLUTINATION\Image\crop Negative\N79.jpg',
    r'H:\AGGLUTINATION\Image\crop Negative\N80.jpg',
    r'H:\AGGLUTINATION\Image\crop Negative\N81.jpg'
]

for image_path in new_image_paths:
    new_image = load_and_preprocess_image(image_path)
    if new_image is not None:
        flattened_image = new_image.flatten().reshape(1, -1)
        prediction = svm_classifier.predict(flattened_image)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        print(f"Image: {image_path}")
        print(f"The image is predicted as: {predicted_label}\n")
    else:
        print(f"Failed to load the image: {image_path}\n")
