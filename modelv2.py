import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans  # Import KMeans
import seaborn as sn
import cv2
import os

# Function to preprocess images
def preprocess_img(img):
    # Check if the image is already grayscale
    if len(img.shape) == 2:
        # Add a channel dimension for compatibility with Conv2D
        img = np.expand_dims(img, axis=-1)
    else:
        # Convert the image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Add a channel dimension for compatibility with Conv2D
        img = np.expand_dims(img, axis=-1)
    # Resize the image
    img = cv2.resize(img, (32, 32))
    # Normalize pixel values to be between 0 and 1
    img = img / 255.0
    return img

def grayscale(img):
    # Check if the image is already grayscale
    if len(img.shape) == 2:
        return img
    else:
        # Convert the image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray_img

# read_malware_images function
def read_malware_images(directory):
    count = 0
    images = []
    label = []
    class_to_int = {}  # Dictionary to map class names to integers

    classes_list = os.listdir(directory)
    noOfClasses = 31

    for class_index in range(noOfClasses):
        class_name = classes_list[class_index]
        class_to_int[class_name] = class_index

        img_list = os.listdir(os.path.join(directory, class_name))

        for img_name in img_list:
            img_path = os.path.join(directory, class_name, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (32, 32))  # Resize images to a common size
            img = preprocess_img(img)

            images.append(img)
            label.append(class_name)  # Use the class name as the label

    return np.array(images), np.array(label), class_to_int


def cluster_images(images, num_clusters):
    images_flat = images.reshape(images.shape[0], -1)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(images_flat)
    cluster_labels = kmeans.labels_
    return cluster_labels

# Load malware images
malware_images, malware_labels, class_to_int = read_malware_images("C:/Users/venur/Downloads/pythoncode/malware_dataset/train")

# Print out class names not found in class_to_int dictionary
unknown_labels = [label for label in malware_labels if label not in class_to_int]
print(f"Labels not found in class_to_int dictionary: {unknown_labels}")

# Convert string labels to integer labels
y_train = [class_to_int[label] for label in malware_labels]
y_train = np.array(y_train)

# Determine the number of classes dynamically
noOfClasses = len(class_to_int)

X_train, X_test, y_train, y_test = train_test_split(
    malware_images, y_train, test_size=0.2, random_state=42
)

X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = np.expand_dims(X_train, axis=-1)

# Convert labels to categorical
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Model architecture
model = Sequential()
# Add convolutional layers, pooling layers, and dense layers based on your requirements
# Example:
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(noOfClasses, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Monitoring and Visualization
class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('accuracy'))

history_callback = AccuracyHistory()

# Training the model with data augmentation
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) / 32,
    validation_data=(X_test, y_test),
    epochs=10,
    callbacks=[history_callback]
)

# Save the model
model.save('malware_modelv2.h5')

# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])

# Confusion matrix
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.savefig('malware_confusion_matrixv2.png', dpi=300, bbox_inches='tight')

# Visualize learning curves
plt.figure(figsize=(10, 7))
plt.plot(range(1, 11), history_callback.acc, marker='o')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('accuracy_over_epochsv2.png', dpi=300, bbox_inches='tight')
plt.show()
