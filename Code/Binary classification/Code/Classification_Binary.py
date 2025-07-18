import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Image preprocessing function
def load_and_preprocess_image(file_path):
    try:
        image = cv2.imread(file_path)
        if image is not None:
            return cv2.resize(image, (256, 256))
        else:
            print(f"Failed to load image: {file_path}")
            return None
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

# Load dataset
data = pd.read_csv('C:\\AGGLUTINATION\\Image\\Binary_classification_dataset.csv') #Insert dataset filepath here
label_encoder = LabelEncoder()
data['labels'] = label_encoder.fit_transform(data['labels'])
class_names = [str(c) for c in label_encoder.classes_]
encoded_labels = data['labels'].tolist()

# Train-test split
file_paths = data['Filepath'].tolist()
X_train, X_test, y_train, y_test = train_test_split(file_paths, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels)

# Load and flatten images
def preprocess_images(paths, labels):
    images, labels_final = [], []
    for path, label in zip(paths, labels):
        img = load_and_preprocess_image(path)
        if img is not None:
            images.append(img.flatten())
            labels_final.append(label)
    return images, labels_final

X_train_flat, y_train_final = preprocess_images(X_train, y_train)
X_test_flat, y_test_final = preprocess_images(X_test, y_test)

# Train SVM
svm_classifier = svm.SVC(kernel='linear', probability=True)
svm_classifier.fit(X_train_flat, y_train_final)

# Evaluate
y_train_pred = svm_classifier.predict(X_train_flat)
y_test_pred = svm_classifier.predict(X_test_flat)
print(f"Train Accuracy: {accuracy_score(y_train_final, y_train_pred):.2f}")
print(f"Test Accuracy: {accuracy_score(y_test_final, y_test_pred):.2f}")

# Cross-validation
cv_scores = cross_val_score(svm_classifier, X_train_flat, y_train_final, cv=5)
print("CV Accuracy Mean:", np.mean(cv_scores))

# Classification report
class_report = classification_report(y_test_final, y_test_pred, target_names=class_names, output_dict=True)
print(classification_report(y_test_final, y_test_pred, target_names=class_names))
class_report_df = pd.DataFrame(class_report).transpose()

# Confusion matrix
conf_matrix = confusion_matrix(y_test_final, y_test_pred)

# Visualizations
plt.figure(figsize=(8, 6))
plt.plot(range(1, 6), cv_scores, marker='o')
plt.title('Cross-Validation Scores')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig('H:\\AGGLUTINATION\\Visualization\\Cross_Validation_Scores.png', dpi=600)
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(class_report_df.iloc[:-1, :-1], annot=True, fmt=".2f", cmap="Blues", cbar=False)
plt.title('Classification Report Heatmap')
plt.savefig('H:\\AGGLUTINATION\\Visualization\\classification_report_heatmap.png', dpi=600)
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('H:\\AGGLUTINATION\\Visualization\\Confusion_Matrix.png', dpi=600)
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x=label_encoder.inverse_transform(encoded_labels))
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.xticks(rotation=45)
plt.savefig('H:\\AGGLUTINATION\\Visualization\\Class_Distribution.png', dpi=600)
plt.show()

# ROC Curve (binary only)
if len(class_names) == 2:
    y_scores = svm_classifier.decision_function(X_test_flat)
    fpr, tpr, thresholds = roc_curve(y_test_final, y_scores)
    auc_score = roc_auc_score(y_test_final, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('H:\\AGGLUTINATION\\Visualization\\ROC_Curve.png', dpi=600)
    plt.show()

# Display incorrect predictions
incorrect_predictions = []
for i in range(len(X_test_flat)):
    if y_test_pred[i] != y_test_final[i]:
        incorrect_predictions.append((X_test[i], y_test_final[i], y_test_pred[i]))

print(f"Number of incorrect predictions: {len(incorrect_predictions)}")

for i, (image_path, actual, predicted) in enumerate(incorrect_predictions[:5]):
    image = load_and_preprocess_image(image_path)
    if image is not None:
        plt.figure(figsize=(4, 4))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        actual_label = label_encoder.inverse_transform([actual])[0]
        predicted_label = label_encoder.inverse_transform([predicted])[0]
        plt.title(f"Actual: {actual_label}, Predicted: {predicted_label}")
        plt.axis('off')
        save_path = f'H:\\AGGLUTINATION\\Visualization\\Incorrect_Prediction_{i+1}.png'
        plt.savefig(save_path, dpi=600)
        plt.show()
    else:
        print(f"Failed to load image: {image_path}")

# Predict on new images
new_image_paths = [
    r'H:\\AGGLUTINATION\\Image\\crop Weak\\W21.jpg',
    r'H:\\AGGLUTINATION\\Image\\crop Weak\\W22.jpg',
    r'H:\\AGGLUTINATION\\Image\\crop Strong\\S61.jpg',
    r'H:\\AGGLUTINATION\\Image\\crop Strong\\S62.jpg',
    r'H:\\AGGLUTINATION\\Image\\crop Moderate\\M27.jpg',
    r'H:\\AGGLUTINATION\\Image\\crop Moderate\\M28.jpg',
    r'H:\\AGGLUTINATION\\Image\\crop Moderate\\M29.jpg',
    r'H:\\AGGLUTINATION\\Image\\crop Moderate\\M30.jpg',
    r'H:\\AGGLUTINATION\\Image\\crop Moderate\\M31.jpg',
    r'H:\\AGGLUTINATION\\Image\\crop Moderate\\M32.jpg',
    r'H:\\AGGLUTINATION\\Image\\crop Negative\\N34.jpg',
    r'H:\\AGGLUTINATION\\Image\\crop Negative\\N35.jpg',
    r'H:\\AGGLUTINATION\\Image\\crop Negative\\N36.jpg',
    r'H:\\AGGLUTINATION\\Image\\crop Negative\\N78.jpg',
    r'H:\\AGGLUTINATION\\Image\\crop Negative\\N79.jpg',
    r'H:\\AGGLUTINATION\\Image\\crop Negative\\N80.jpg',
    r'H:\\AGGLUTINATION\\Image\\crop Negative\\N81.jpg'
]

print("\nPredictions on New Images:")
for path in new_image_paths:
    image = load_and_preprocess_image(path)
    if image is not None:
        flattened = image.flatten().reshape(1, -1)
        prediction = svm_classifier.predict(flattened)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        print(f"Image: {path}\nPrediction: {predicted_label}\n")
    else:
        print(f"Failed to load: {path}")

# Save to Excel
with pd.ExcelWriter('svm_analysis_results.xlsx', engine='xlsxwriter') as writer:
    pd.DataFrame({'Fold': range(1, 6), 'Accuracy': cv_scores}).to_excel(writer, sheet_name='CV_Scores', index=False)
    class_report_df.to_excel(writer, sheet_name='Classification_Report')
    pd.DataFrame(conf_matrix, index=class_names, columns=class_names).to_excel(writer, sheet_name='Confusion_Matrix')
    if len(class_names) == 2:
        pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Thresholds': thresholds}).to_excel(writer, sheet_name='ROC_Curve')
    pd.DataFrame({'Class': class_names, 'Count': pd.Series(encoded_labels).value_counts().sort_index().values}).to_excel(writer, sheet_name='Class_Distribution', index=False)

print("All results saved successfully.")
