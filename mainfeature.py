import cv2
import numpy as np
from tensorflow import keras
import tkinter as tk
from tkinter import filedialog
from keras.preprocessing.image import ImageDataGenerator
import datetime
import os
from skimage.feature import hog
from sklearn.cluster import KMeans
from modelv2 import grayscale
import time

threshold = 0.75  # THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
#PATH TO MODEL
model = keras.models.load_model('C:/Users/venur/Downloads/pythoncode/malware_modelv2.h5')  # loading trained malware detection model

# Pre-processing for detecting malware in an image
def preprocess_img(imgBGR, erode_dilate=True):
    return imgBGR

def contour_detect(img_bin, min_area, max_area=-1, wh_ratio=2.0):
    rects = []
    contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return rects

    max_area = img_bin.shape[0] * img_bin.shape[1] if max_area < 0 else max_area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area and area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            if 1.0 * w / h < wh_ratio and 1.0 * h / w < wh_ratio:
                rects.append([x, y, w, h])
    return rects

def preprocessing(img):
    img = grayscale(img)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

def extract_hog_features(img):
    # image to grayscale
    gray_img = grayscale(img)

    # Extract HOG features
    features, _ = hog(gray_img, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=True)

    return features

def cluster_images(images, num_clusters):
    images_flat = images.reshape(images.shape[0], -1)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(images_flat)
    cluster_labels = kmeans.labels_
    return cluster_labels

def detect_malware(img, malware_classes):
    # Convert the image to grayscale if it's a color image
    if img.shape[-1] == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img

    # Apply a simple thresholding to detect edges
    _, thresholded = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)

    # Ensure the thresholded image is of type np.uint8
    thresholded = np.uint8(thresholded)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Set a threshold for the number of contours to consider it as malware
    malware_threshold = 5

    # Check if the number of contours exceeds the threshold
    if len(contours) > malware_threshold:
        result = "Malware Detected!"
        # Extract the detected class from the list of malware classes
        detected_class = get_detected_class(img_path, malware_classes)
    else:
        result = "No Malware Detected"
        detected_class = "N/A"

    return thresholded, result, detected_class

def get_detected_class(img_path, malware_classes):
    # Extract the class name from the image path
    class_name = img_path.split("/")[-2]  
    # Check if the extracted class name is in the list of malware classes
    if class_name in malware_classes:
        return class_name
    else:
        return "Unknown"

#Auto scanning a foldeR
def scan_and_test_images(directory, malware_classes):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)

                if img is not None:
                    # Preprocess the image
                    img = preprocess_img(img)
                    gray_img = grayscale(img)

                    # batch and channel dimension
                    img_for_augmentation = np.expand_dims(np.expand_dims(gray_img, axis=0), axis=-1)
                    # Reshaping the input data for data augmentation
                    img_for_augmentation = img_for_augmentation / 255.0

                    # data augmentation
                    augmented_images = datagen.flow(img_for_augmentation, batch_size=1)
                    augmented_img = next(augmented_images)[0]

                    # malware detection on both the original and augmented images
                    detected_original, result_original, malware_class_original = detect_malware(gray_img, malware_classes)
                    detected_augmented, result_augmented, _ = detect_malware(augmented_img, malware_classes)

                    # results
                    cv2.imshow("Original Image", detected_original)
                    cv2.imshow("Augmented Image", detected_augmented)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    # detection summary for the original image
                    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    summary_original = f"Original Image: {img_path}\nResult: {result_original}\nMalware Class: {malware_class_original}\nDetection Accuracy: {threshold}\nDate and Time: {current_datetime}\n"

                    # Print and save the summary
                    print("Detection Summary for Original Image:")
                    print(summary_original)

                    with open('detection_summary.txt', 'a') as f:
                        f.write("\n\nDetection Summary for Original Image:\n")
                        f.write(summary_original)

# Chossing a specific  image
def choose_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    root.destroy()  # This will close the Tkinter window after selecting the file
    return file_path

#Folder for auto scanning
def choose_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Folder Containing Images")
    root.destroy()
    return folder_path

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

#Main function
if __name__ == '__main__':
    while True:
        img_path = choose_image() #Promt for choosing an image
        if not img_path:
            print("No image selected. Exiting.")
            break  # Break the loop if no image is selected
        elif img_path.lower() == 'quit':
            print("Quitting scanning.")
            break  # Break the loop if the user enters 'quit'
        else:
            # Load the image
            img = cv2.imread(img_path)
            img = preprocess_img(img)
            # Convert the image to grayscale
            gray_img = grayscale(img)
            # Add a batch and channel dimension
            img_for_augmentation = np.expand_dims(np.expand_dims(gray_img, axis=0), axis=-1)
            img_for_augmentation = img_for_augmentation / 255.0  # Normalize pixel values to be between 0 and 1

            augmented_images = datagen.flow(img_for_augmentation, batch_size=1)
            augmented_img = next(augmented_images)[0]

            # List of malware classes
            malware_classes = os.listdir("C:/Users/venur/Downloads/pythoncode/malware_dataset/train")
            detected_original, result_original, malware_class_original = detect_malware(gray_img, malware_classes)
            detected_augmented, result_augmented, _ = detect_malware(augmented_img, malware_classes)

            # Generate cluster summary
            cluster_labels_original = cluster_images(gray_img, num_clusters=5)
            cluster_summary = f"Cluster Summary: {cluster_labels_original}\n"

            # Generate feature technique summary
            hog_features_original = extract_hog_features(gray_img)
            feature_summary = f"Feature Technique Summary: HOG Features - {hog_features_original}\n"

            # Display the results
            cv2.imshow("Original Image", detected_original)
            cv2.imshow("Augmented Image", detected_augmented)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Generate detection summary for the original image
            current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            summary_original = f"Original Image: {img_path}\nResult: {result_original}\nMalware Class: {malware_class_original}\n" \
                               f"Detection Accuracy: {threshold}\nDate and Time: {current_datetime}\n" \
                               f"{cluster_summary}{feature_summary}"

            # Print and save the summary
            print("Detection Summary for Original Image:")
            print(summary_original)

            with open('detection_summary.txt', 'w') as f:
                f.write("Detection Summary for Original Image:\n")
                f.write(summary_original)

            time.sleep(5)  # delay of 5 seconds before scanning the next image
