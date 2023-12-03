import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize empty lists to store data and labels
data = []
labels = []

# Define the folder containing data
data_folder = 'data_june'

# List of class folders
class_folders = ['control_Nucleous', 'MV_DATA', 'SARS_Nucleous']

# Loop through class folders
for class_index, class_folder in enumerate(class_folders):
    class_path = os.path.join(data_folder, class_folder)
    
    # Loop through files in each class folder
    for filename in os.listdir(class_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(class_path, filename)
            
            # Read the data from the text file and extract the first two columns
            df = pd.read_csv(file_path, sep='\t', header=None, usecols=[0, 1])
            features = df.values
            if df.shape[0] == 1600 and df.shape[1] == 2:
                # Append the data and corresponding label
                data.append(features)
                labels.append(class_index)

# Convert data and labels to numpy arrays
data = np.stack(data, axis=0)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print("--------------",X_train.shape)
def build_resnet(input_shape, num_classes):
    input_layer = keras.Input(shape=input_shape)
    
    # Initial convolutional layer
    x = layers.Conv1D(64, 7, strides=2, padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Stack of residual blocks (add more if needed)
    for _ in range(3):  # You can adjust the number of residual blocks
        residual = x
        x = layers.Conv1D(64, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(64, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)

    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Fully connected layer
    x = layers.Dense(128, activation='relu')(x)
    
    # Output layer with softmax activation for 3-class classification
    output_layer = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer, name='resnet')
    
    return model

# Define input shape and number of classes
input_shape = (1600, 2)  # Each instance has a 1600x2 feature matrix
num_classes = 3  # Three classes in your dataset

# Build the ResNet model
model = build_resnet(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print a summary of the model architecture
model.summary()

# Convert labels to one-hot encoding
y_train_one_hot = keras.utils.to_categorical(y_train, num_classes)

# Train the model using your data and one-hot encoded labels
model.fit(X_train, y_train_one_hot, epochs=300, batch_size=32)  # Adjust epochs and batch_size as needed

# Convert test labels to one-hot encoding (if not already done)
y_test_one_hot = keras.utils.to_categorical(y_test, num_classes)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test_one_hot)

# Print the test loss and accuracy
print(f"ResNet Test Loss: {test_loss:.4f}")
print(f"ResNet Test Accuracy: {test_accuracy * 100:.2f}%")

# Make predictions on the training set
y_pred_train = np.argmax(model.predict(X_train), axis=1)

# Make predictions on the test set
y_pred_test = np.argmax(model.predict(X_test), axis=1)

# Make predictions on the whole dataset
data_one_hot = keras.utils.to_categorical(labels, num_classes)
y_pred_whole = np.argmax(model.predict(data), axis=1)

# Create confusion matrices
conf_matrix_train = confusion_matrix(y_train, y_pred_train)
conf_matrix_test = confusion_matrix(y_test, y_pred_test)
conf_matrix_whole = confusion_matrix(labels, y_pred_whole)

# Plot confusion matrices
def plot_confusion_matrix(matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Plot confusion matrices for training set, test set, and whole dataset
plot_confusion_matrix(conf_matrix_train, "Training Set Confusion Matrix")
plot_confusion_matrix(conf_matrix_test, "Test Set Confusion Matrix")
plot_confusion_matrix(conf_matrix_whole, "Whole Dataset Confusion Matrix")