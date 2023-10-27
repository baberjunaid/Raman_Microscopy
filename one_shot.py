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


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Function to generate triplets
def generate_triplets(data, labels):
    triplets = []
    for class_label in set(labels):
        class_indices = np.where(labels == class_label)[0]
        if len(class_indices) < 2:
            continue  # Skip classes with fewer than 2 samples
        anchor, positive = np.random.choice(class_indices, size=2, replace=False)
        negative = np.random.choice(np.where(labels != class_label)[0])
        triplets.append((data[anchor], data[positive], data[negative]))
    return np.array(triplets)

# Load and preprocess data (assuming data and labels are correctly loaded)
# You may need to reshape data to (num_samples, 1600, 2) if not already in that shape
def trip_classification(data, labels):
        
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Define the Siamese-like network
    input_shape = (1600, 2)
    anchor_input = keras.Input(shape=input_shape, name='anchor_input')
    positive_input = keras.Input(shape=input_shape, name='positive_input')
    negative_input = keras.Input(shape=input_shape, name='negative_input')

    shared_network = keras.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
    ])

    anchor_embedding = shared_network(anchor_input)
    positive_embedding = shared_network(positive_input)
    negative_embedding = shared_network(negative_input)

    merged_vector = layers.concatenate([anchor_embedding, positive_embedding, negative_embedding], axis=-1)

    triplet_network = keras.Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_vector)

    # Define triplet loss function
    def triplet_loss(y_true, y_pred):
        anchor, positive, negative = y_pred[:, :64], y_pred[:, 64:128], y_pred[:, 128:]
        positive_distance = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        negative_distance = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        loss = tf.maximum(0.0, positive_distance - negative_distance + 0.2)  # Adjust margin as needed
        return loss

    # Compile the model with triplet loss
    triplet_network.compile(optimizer='adam', loss=triplet_loss)

    # Generate triplets for training
    triplets = generate_triplets(X_train, y_train)

    # Training loop
    batch_size = 64
    num_epochs = 50

    for epoch in range(num_epochs):
        np.random.shuffle(triplets)
        for i in range(0, len(triplets), batch_size):
            batch_triplets = triplets[i:i+batch_size]
            anchor_batch, positive_batch, negative_batch = batch_triplets[:, 0], batch_triplets[:, 1], batch_triplets[:, 2]
            triplet_network.train_on_batch([anchor_batch, positive_batch, negative_batch], np.zeros((len(batch_triplets),)))

        # Calculate and print training accuracy
        training_predictions = triplet_network.predict([X_train, X_train, X_train])
        training_distances = np.sqrt(np.sum(np.square(training_predictions[:, :64] - training_predictions[:, 64:128]), axis=-1))
        training_accuracy = np.mean(training_distances <= 0.2)  # Adjust margin as needed

        # Calculate and print test accuracy
        test_predictions = triplet_network.predict([X_test, X_test, X_test])
        test_distances = np.sqrt(np.sum(np.square(test_predictions[:, :64] - test_predictions[:, 64:128]), axis=-1))
        test_accuracy = np.mean(test_distances <= 0.2)  # Adjust margin as needed

        print(f'Epoch {epoch+1}/{num_epochs}, Training Accuracy: {training_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}')
        # Save the trained model to a file
        triplet_network.save('triplet_model.h5')

trip_classification(data,labels)

