import numpy as np
import pandas as pd
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical


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
encoded_labels = np.array([label_mapping[label] for label in labels])  # 1D array
encoded_labels = to_categorical(encoded_labels, num_classes=len(label_mapping))


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


image_data = np.array(image_data)
imu_data = data[imu_columns].values
gps_data = data[gps_columns].values
velocity_data = data[velocity_columns].values


image_data = image_data.astype('float32') / 255.0


def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model


def pca_fusion(imu, gps, velocity, n_components=10):
    combined = np.concatenate((imu, gps, velocity), axis=1)
    pca = PCA(n_components=n_components)
    pca_combined = pca.fit_transform(combined)
    return pca_combined


X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(image_data, encoded_labels, test_size=0.3, random_state=42)
imu_train, imu_test = train_test_split(imu_data, test_size=0.3, random_state=42)
gps_train, gps_test = train_test_split(gps_data, test_size=0.3, random_state=42)
velocity_train, velocity_test = train_test_split(velocity_data, test_size=0.3, random_state=42)


cnn_model = create_cnn_model((32, 32, 3), len(label_mapping))
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = cnn_model.fit(X_train_img, y_train_img, epochs=200, batch_size=32, validation_split=0.2)


plt.figure(figsize=(12, 10))


plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Learning Curve for CNN Model - Accuracy')
plt.legend()


plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curve for CNN Model - Loss')
plt.legend()

plt.tight_layout()
plt.show()


cnn_feature_extractor = Sequential(cnn_model.layers[:-1])
train_features = cnn_feature_extractor.predict(X_train_img)
test_features = cnn_feature_extractor.predict(X_test_img)


X_fusion_pca_train = pca_fusion(imu_train, gps_train, velocity_train)
X_fusion_pca_test = pca_fusion(imu_test, gps_test, velocity_test)


X_combined_cnn_train = np.hstack([train_features, X_fusion_pca_train])
X_combined_cnn_test = np.hstack([test_features, X_fusion_pca_test])


lgb_model_cnn = lgb.LGBMClassifier()  # Use default parameters
lgb_model_cnn.fit(X_combined_cnn_train, np.argmax(y_train_img, axis=1))
y_pred_cnn = lgb_model_cnn.predict(X_combined_cnn_test)


accuracy_cnn = accuracy_score(np.argmax(y_test_img, axis=1), y_pred_cnn)
class_report_cnn = classification_report(np.argmax(y_test_img, axis=1), y_pred_cnn, target_names=label_mapping.keys())
conf_matrix_cnn = confusion_matrix(np.argmax(y_test_img, axis=1), y_pred_cnn)


print('Classification Report - With CNN:')
print(class_report_cnn)
print(f'Accuracy (With CNN): {accuracy_cnn * 100:.2f}%')


X_fusion_pca_no_cnn = pca_fusion(imu_data, gps_data, velocity_data)
X_train_no_cnn, X_test_no_cnn, y_train_no_cnn, y_test_no_cnn = train_test_split(X_fusion_pca_no_cnn, np.argmax(encoded_labels, axis=1), test_size=0.3, random_state=42)


lgb_model_no_cnn = lgb.LGBMClassifier()  # Use default parameters
lgb_model_no_cnn.fit(X_train_no_cnn, y_train_no_cnn)
y_pred_no_cnn = lgb_model_no_cnn.predict(X_test_no_cnn)


accuracy_no_cnn = accuracy_score(y_test_no_cnn, y_pred_no_cnn)
class_report_no_cnn = classification_report(y_test_no_cnn, y_pred_no_cnn, target_names=label_mapping.keys())
conf_matrix_no_cnn = confusion_matrix(y_test_no_cnn, y_pred_no_cnn)


print('Classification Report - No CNN:')
print(class_report_no_cnn)
print(f'Accuracy (No CNN): {accuracy_no_cnn * 100:.2f}%')


plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_cnn, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - With CNN')
plt.show()


plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_no_cnn, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - No CNN')
plt.show()
