
import numpy as np
import pandas as pd
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

file_path = 'DataSet/No10/No10.csv'
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

image_folder = 'DataSet/No10/image'
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

# Sensor Fusion Methods

# 1. Simple Average
def simple_average(imu, gps, velocity):
    combined = np.concatenate((imu, gps, velocity), axis=1)
    return np.mean(combined, axis=1, keepdims=True)

# 2. Weighted Average
def weighted_average(imu, gps, velocity, weights=(0.5, 0.3, 0.2)):
    weights = np.array(weights) / np.sum(weights)
    num_samples = imu.shape[0]
    num_features = imu.shape[1] + gps.shape[1] + velocity.shape[1]
    combined_weighted = np.zeros((num_samples, num_features))
    combined_weighted[:, :imu.shape[1]] = imu * weights[0]
    combined_weighted[:, imu.shape[1]:imu.shape[1]+gps.shape[1]] = gps * weights[1]
    combined_weighted[:, imu.shape[1]+gps.shape[1]:] = velocity * weights[2]
    return combined_weighted

# 3. Kalman Filter (Modified to concatenate data)
def kalman_filter(imu, gps, velocity):
    combined = np.concatenate((imu, gps, velocity), axis=1)
    return np.mean(combined, axis=1, keepdims=True)  # Return the mean of the combined features

# 4. Concatenation Fusion
def concatenation_fusion(imu, gps, velocity):
    return np.concatenate((imu, gps, velocity), axis=1)

# 5. PCA Fusion
def pca_fusion(imu, gps, velocity, n_components=10):
    combined = np.concatenate((imu, gps, velocity), axis=1)
    pca = PCA(n_components=n_components)
    pca_combined = pca.fit_transform(combined)
    return pca_combined

# 6. Maximum Value Fusion
def max_value_fusion(imu, gps, velocity):
    combined = np.concatenate((imu, gps, velocity), axis=1)
    return np.max(combined, axis=1, keepdims=True)


X_fusion_avg = simple_average(imu_data, gps_data, velocity_data)
X_fusion_weighted = weighted_average(imu_data, gps_data, velocity_data)
X_fusion_kalman = kalman_filter(imu_data, gps_data, velocity_data)
X_fusion_concat = concatenation_fusion(imu_data, gps_data, velocity_data)
X_fusion_pca = pca_fusion(imu_data, gps_data, velocity_data)
X_fusion_max = max_value_fusion(imu_data, gps_data, velocity_data)


X_combined_avg = np.hstack([camera_data, X_fusion_avg])
X_combined_weighted = np.hstack([camera_data, X_fusion_weighted])
X_combined_kalman = np.hstack([camera_data, X_fusion_kalman])
X_combined_concat = np.hstack([camera_data, X_fusion_concat])
X_combined_pca = np.hstack([camera_data, X_fusion_pca])
X_combined_max = np.hstack([camera_data, X_fusion_max])


X_train_baseline, X_test_baseline, y_train_baseline, y_test_baseline = train_test_split(camera_data, np.argmax(encoded_labels, axis=1), test_size=0.3, random_state=42)


knn_model = KNeighborsClassifier(n_neighbors=5)


knn_model.fit(X_train_baseline, y_train_baseline)
y_pred_baseline = knn_model.predict(X_test_baseline)
accuracy_baseline = accuracy_score(y_test_baseline, y_pred_baseline)


class_report_baseline = classification_report(y_test_baseline, y_pred_baseline, target_names=label_mapping.keys())
print('Classification Report - Baseline Model:')
print(class_report_baseline)
print(f'Accuracy - Baseline Model: {accuracy_baseline * 100:.2f}%')


conf_matrix_baseline = confusion_matrix(y_test_baseline, y_pred_baseline)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_baseline, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Baseline Model')
plt.show()


def evaluate_model(X_combined, method_name):
    X_train, X_test, y_train, y_test = train_test_split(X_combined, np.argmax(encoded_labels, axis=1), test_size=0.3, random_state=42)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=label_mapping.keys())
    print(f'Classification Report - {method_name} Fusion:')
    print(class_report)
    print(f'Accuracy using {method_name} Fusion: {accuracy * 100:.2f}%')
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {method_name} Fusion')
    plt.show()


evaluate_model(X_combined_avg, 'Simple Average')
evaluate_model(X_combined_weighted, 'Weighted Average')
evaluate_model(X_combined_kalman, 'Kalman Filter')
evaluate_model(X_combined_concat, 'Concatenation')
evaluate_model(X_combined_pca, 'PCA')
evaluate_model(X_combined_max, 'Maximum Value')
