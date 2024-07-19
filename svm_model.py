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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize the SVM model
svm_model = SVC(kernel='linear', C=1.0, random_state=42)

# Train the SVM model
svm_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)  # Flatten the input data

# Make predictions on the test set
