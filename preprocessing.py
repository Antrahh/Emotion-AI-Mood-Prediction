import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Path to dataset
dataset_path = "dataset"

# Emotion labels
emotions = ["happy", "sad", "angry", "neutral"]

data = []
labels = []

# Reading images
for emotion in emotions:
    folder_path = os.path.join(dataset_path, emotion)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        # Read image in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Resize to 48x48
        img = cv2.resize(img, (48, 48))

        data.append(img)
        labels.append(emotions.index(emotion))

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Normalize pixel values (0-255 → 0-1)
data = data / 255.0

# Reshape for CNN input
data = data.reshape(-1, 48, 48, 1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))