import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
#                                print(features)
    #             print(features.shape)

                # Append the data and corresponding label
                data.append(features)
                labels.append(class_index)

# Convert data and labels to numpy arrays
# data= np.array(data)

print(type(data), type(data[0]))
data = np.stack(data, axis=0)
# data  = np.stack(np.array(data))
labels = np.array(labels)


# After training, you can use the trained network for one-shot learning.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the residual block (same as before)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build the ResNet model
def build_resnet(input_shape, num_classes):
    input_layer = keras.Input(shape=input_shape)
    
    # Initial convolutional layer
    x = layers.Conv1D(64, 7, strides=2, padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Stack of residual blocks (same as before)
    
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

# Assuming you have a test set X_test and y_test
