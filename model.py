# %%

# Import necessary libraries
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from itertools import product
from sklearn import metrics

from keras.models import Sequential
from keras.layers import LSTM, Dense
from language_selection_gui import select_language  # Ensure this import works based on your directory structure

# Choose the language for model training
languages = ['English', 'Korean']
selected_language = select_language(languages)

# Set the path to the data directory for the selected language
PATH = os.path.join('data', selected_language)

# Ensure the selected language directory exists
if not os.path.exists(PATH):
    raise ValueError(f"Data for {selected_language} does not exist.")

# Create an array of actions (signs) labels by listing the contents of the data directory
actions = np.array([action for action in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, action))])

# Define the number of sequences and frames
sequences = 30
frames = 10

# Create a label map to map each action label to a numeric value
label_map = {label: num for num, label in enumerate(actions)}

# Initialize empty lists to store landmarks and labels
landmarks, labels = [], []

# Iterate over actions and sequences to load landmarks and corresponding labels
for action, sequence in product(actions, range(sequences)):
    temp = []
    for frame in range(frames):
        npy_path = os.path.join(PATH, action, str(sequence), f"{frame}.npy")
        if os.path.exists(npy_path):  # Ensure the .npy file exists before loading
            npy = np.load(npy_path)
            temp.append(npy)
    if temp:  # Ensure the temp list is not empty
        landmarks.append(temp)
        labels.append(label_map[action])

# Proceed only if landmarks data is available
if landmarks:
    # Convert landmarks and labels to numpy arrays
    X, Y = np.array(landmarks), to_categorical(labels).astype(int)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=34, stratify=Y)

    # Define the model architecture
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, activation='relu', input_shape=(10, 126)))
    model.add(LSTM(64, return_sequences=True, activation='relu'))
    model.add(LSTM(32, return_sequences=False, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    # Compile the model with Adam optimizer and categorical cross-entropy loss
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # Train the model
    model.fit(X_train, Y_train, epochs=100)

    # Save the trained model with the selected language in its name
    model.save(f'my_model_{selected_language}.py')

    # Make predictions on the test set
    predictions = np.argmax(model.predict(X_test), axis=1)
    # Get the true labels from the test set
    test_labels = np.argmax(Y_test, axis=1)

    # Calculate the accuracy of the predictions
    accuracy = metrics.accuracy_score(test_labels, predictions)
    print(f"Model accuracy: {accuracy}")
else:
    print("No data available for training.")
