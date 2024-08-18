

import numpy as np
import pandas as pd
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

file_path = 'DataSet/No1/No1.csv'
data = pd.read_csv(file_path)

imu_columns = ['Accelerometer x', 'Accelerometer y', 'Accelerometer z', 'Gyroscope x', 'Gyroscope y', 'Gyroscope z']
gps_columns = ['GPS x', 'GPS y', 'GPS z']
velocity_columns = ['Velocity x', 'Velocity y', 'Velocity z']
image_column = 'name'
target_column = 'Terrain'

data[image_column] = data[image_column].astype(str)
data = data[data[image_column] != '0']
data[target_column] = data[target_column].astype(str)
data = data[data[target_column] != 'none']
data = data[data[target_column] != 'nan']

class_counts = data[target_column].value_counts()
min_class_count = class_counts.min()
balanced_data = data.groupby(target_column).sample(n=min_class_count, random_state=42)
data = balanced_data

image_names = data[image_column].values
labels = data[target_column].values

label_mapping = {label: idx for idx, label in enumerate(np.unique(labels))}
encoded_labels = np.array([label_mapping[label] for label in labels])

image_folder = 'DataSet/No1/image'
image_data = []
valid_indices = []

for idx, image_name in enumerate(image_names):
    image_path = os.path.join(image_folder, image_name + '.png')
    try:
        image = Image.open(image_path)
        image = image.resize((32, 32))
        image_array = np.array(image)
        if image_array.shape[-1] != 3:
            if image_array.ndim == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            else:
                image_array = image_array[:, :, :3]
        image_data.append(image_array)
        valid_indices.append(idx)
    except FileNotFoundError:
        print(f"File not found: {image_path}")
    except Exception as e:
        print(f"Error loading {image_path}: {e}")

data = data.iloc[valid_indices]

image_names = data[image_column].values
labels = data[target_column].values

encoded_labels = np.array([label_mapping[label] for label in labels])
encoded_labels = np.eye(len(label_mapping))[encoded_labels]

image_data = np.array(image_data)
imu_data = data[imu_columns].values
gps_data = data[gps_columns].values
velocity_data = data[velocity_columns].values

image_data = image_data.astype('float32') / 255.0

image_data_flattened = image_data.reshape(len(image_data), -1)

camera_data = np.sum(image_data_flattened, axis=1).reshape(-1, 1)


X_combined = np.hstack([camera_data, imu_data, gps_data, velocity_data])
X_camera_only = camera_data
if X_combined.shape[0] != encoded_labels.shape[0] or X_camera_only.shape[0] != encoded_labels.shape[0]:
    raise ValueError("Mismatch between number of samples in input data and encoded labels.")

X_train_combined, X_test_combined, y_train, y_test = train_test_split(X_combined, np.argmax(encoded_labels, axis=1), test_size=0.3, random_state=42)

X_train_camera, X_test_camera, _, _ = train_test_split(X_camera_only, np.argmax(encoded_labels, axis=1), test_size=0.3, random_state=42)


rf_model_combined = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_combined.fit(X_train_combined, y_train)

y_pred_combined = rf_model_combined.predict(X_test_combined)

accuracy_combined = accuracy_score(y_test, y_pred_combined)
print(f'Accuracy Score with Combined Inputs: {accuracy_combined * 100:.2f}%')


rf_model_camera = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_camera.fit(X_train_camera, y_train)

y_pred_camera = rf_model_camera.predict(X_test_camera)

accuracy_camera = accuracy_score(y_test, y_pred_camera)
print(f'Accuracy Score with Camera-Only Inputs: {accuracy_camera * 100:.2f}%')


conf_matrix_combined = confusion_matrix(y_test, y_pred_combined)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_combined, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Combined Inputs')
plt.show()

class_report_combined = classification_report(y_test, y_pred_combined, target_names=label_mapping.keys())
print('Classification Report - Combined Inputs:')
print(class_report_combined)


conf_matrix_camera = confusion_matrix(y_test, y_pred_camera)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_camera, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Camera-Only Inputs')
plt.show()

class_report_camera = classification_report(y_test, y_pred_camera, target_names=label_mapping.keys())
print('Classification Report - Camera-Only Inputs:')
print(class_report_camera)

# Feature Importances for Combined Inputs
importances_combined = rf_model_combined.feature_importances_
feature_names_combined = ['Camera Data'] + imu_columns + gps_columns + velocity_columns

if len(feature_names_combined) != len(importances_combined):
    raise ValueError("Mismatch between number of features and number of importances.")

importance_df_combined = pd.DataFrame({'Feature': feature_names_combined, 'Importance': importances_combined})
importance_df_combined = importance_df_combined.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
plt.title('Feature Importances - Combined Inputs')
sns.barplot(x='Importance', y='Feature', data=importance_df_combined)
plt.show()


train_sizes_combined, train_scores_combined, test_scores_combined = learning_curve(
    rf_model_combined, X_combined, np.argmax(encoded_labels, axis=1), cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')

train_scores_mean_combined = np.mean(train_scores_combined, axis=1)
train_scores_std_combined = np.std(train_scores_combined, axis=1)
test_scores_mean_combined = np.mean(test_scores_combined, axis=1)
test_scores_std_combined = np.std(test_scores_combined, axis=1)

plt.figure(figsize=(12, 8))
plt.title('Learning Curve (Random Forest) - Combined Inputs')
plt.xlabel('Training Examples')
plt.ylabel('Accuracy')
plt.grid()
plt.fill_between(train_sizes_combined, train_scores_mean_combined - train_scores_std_combined, train_scores_mean_combined + train_scores_std_combined, alpha=0.1, color="r")
plt.fill_between(train_sizes_combined, test_scores_mean_combined - test_scores_std_combined, test_scores_mean_combined + test_scores_std_combined, alpha=0.1, color="g")
plt.plot(train_sizes_combined, train_scores_mean_combined, 'o-', color="r", label="Training score")
plt.plot(train_sizes_combined, test_scores_mean_combined, 'o-', color="g", label="Cross-validation score")
plt.legend(loc="best")
plt.show()


train_sizes_camera, train_scores_camera, test_scores_camera = learning_curve(
    rf_model_camera, X_camera_only, np.argmax(encoded_labels, axis=1), cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')

train_scores_mean_camera = np.mean(train_scores_camera, axis=1)
train_scores_std_camera = np.std(train_scores_camera, axis=1)
test_scores_mean_camera = np.mean(test_scores_camera, axis=1)
test_scores_std_camera = np.std(test_scores_camera, axis=1)

plt.figure(figsize=(12, 8))
plt.title('Learning Curve (Random Forest) - Camera-Only Inputs')
plt.xlabel('Training Examples')
plt.ylabel('Accuracy')
plt.grid()
plt.fill_between(train_sizes_camera, train_scores_mean_camera - train_scores_std_camera, train_scores_mean_camera + train_scores_std_camera, alpha=0.1, color="r")
plt.fill_between(train_sizes_camera, test_scores_mean_camera - test_scores_std_camera, test_scores_mean_camera + test_scores_std_camera, alpha=0.1, color="g")
plt.plot(train_sizes_camera, train_scores_mean_camera, 'o-', color="r", label="Training score")
plt.plot(train_sizes_camera, test_scores_mean_camera, 'o-', color="g", label="Cross-validation score")
plt.legend(loc="best")
plt.show()
