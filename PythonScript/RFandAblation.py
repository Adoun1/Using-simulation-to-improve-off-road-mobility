
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


imu_columns = ['Accelerometer x', 'Accelerometer y', 'Accelerometer z', 'Gyroscope x', 'Gyroscope y', 'Gyroscope z']
gps_columns = ['GPS x', 'GPS y', 'GPS z']
velocity_columns = ['Velocity x', 'Velocity y', 'Velocity z']
image_column = 'name'
target_column = 'Terrain'


def load_and_preprocess_data():
    file_path = 'DataSet/No11/No11.csv'
    data = pd.read_csv(file_path)

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

    image_folder = 'DataSet/No11/image'
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

    return camera_data, imu_data, gps_data, velocity_data, encoded_labels, label_mapping


def train_and_evaluate_model(X_train, X_test, y_train, y_test, feature_names, label_mapping):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy Score: {accuracy * 100:.2f}%')

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.keys(),
                yticklabels=label_mapping.keys())
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    class_report = classification_report(y_test, y_pred, target_names=label_mapping.keys())
    print('Classification Report:')
    print(class_report)

    importances = rf_model.feature_importances_
    if len(feature_names) != len(importances):
        raise ValueError("Mismatch between number of features and number of importances.")

    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.show()

    return accuracy


def plot_learning_curve(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(12, 8))
    plt.title('Learning Curve (Random Forest)')
    plt.xlabel('Training Examples')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()



camera_data, imu_data, gps_data, velocity_data, encoded_labels, label_mapping = load_and_preprocess_data()


feature_sets = {
    'All Features': np.hstack([camera_data, imu_data, gps_data, velocity_data]),
    'Without IMU': np.hstack([camera_data, gps_data, velocity_data]),
    'Without GPS': np.hstack([camera_data, imu_data, velocity_data]),
    'Without Velocity': np.hstack([camera_data, imu_data, gps_data])
}

feature_names_sets = {
    'All Features': ['Camera Data'] + imu_columns + gps_columns + velocity_columns,
    'Without IMU': ['Camera Data'] + gps_columns + velocity_columns,
    'Without GPS': ['Camera Data'] + imu_columns + velocity_columns,
    'Without Velocity': ['Camera Data'] + imu_columns + gps_columns
}


results = {}
for key in feature_sets:
    X_combined = feature_sets[key]
    feature_names = feature_names_sets[key]

    X_train, X_test, y_train, y_test = train_test_split(X_combined, np.argmax(encoded_labels, axis=1), test_size=0.3,
                                                        random_state=42)

    print(f"Evaluating model with {key}...")
    accuracy = train_and_evaluate_model(X_train, X_test, y_train, y_test, feature_names, label_mapping)
    results[key] = accuracy


print("\nSummary of Results:")
for key in results:
    print(f"Model with {key}: Accuracy = {results[key] * 100:.2f}%")


for key in feature_sets:
    X_combined = feature_sets[key]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    plot_learning_curve(model, X_combined, np.argmax(encoded_labels, axis=1))
