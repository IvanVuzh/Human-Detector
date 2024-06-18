import json
import os
import cv2
import numpy as np
from sklearn.svm import LinearSVC
from joblib import dump
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

def load_annotations(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def extract_positive_samples(data, img_folder, max_extracted_samples = 0):
    positives = []
    index = 1
    for annotation in data['annotations']:
        if annotation['iscrowd'] == 1: 
            continue
        
        image_id = annotation['image_id']
        image_info = next(item for item in data['images'] if item['id'] == image_id)
        img_path = os.path.join(img_folder, image_info['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Image at {img_path} could not be loaded.")
            continue
        x, y, w, h = annotation['bbox']
        person_img = img[int(y):int(y+h), int(x):int(x+w)]
        if person_img.size != 0:
            positives.append(cv2.resize(person_img, (64, 128)))
        if (index % 1000 == 0):
            print(index, max_extracted_samples)
            print(f"extracted {index} positive samples")
            if (index >= max_extracted_samples):
                return positives
        index += 1
    return positives

def extract_positive_samples_from_csv(csv_file, img_folder):
    # Load CSV file
    data = pd.read_csv(csv_file)
    print("Csv annotations loaded")

    positives = []
    positive_filenames = next(os.walk(img_folder), (None, None, []))[2]
    rows_to_delete = []
    # Loop through all rows in the CSV (works better than loop through images)
    for index, row in data.iterrows():
        if (index % 100 == 0): print(f"Оброблено {index} рядків")
        image_id = row['ImageID']
        x_min = row['XMin']
        x_max = row['XMax']
        y_min = row['YMin']
        y_max = row['YMax']
        
        # Construct image path
        img_path = os.path.join(img_folder, f"{image_id}.jpg")
        if (not f"{image_id}.jpg" in positive_filenames): 
            rows_to_delete.append(index)
            continue
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Image at {img_path} could not be loaded.")
            continue
        
        # Convert normalized coordinates to absolute pixel values
        height, width, _ = img.shape
        x = int(x_min * width)
        y = int(y_min * height)
        w = int((x_max - x_min) * width)
        h = int((y_max - y_min) * height)
        
        # Extract the region of interest (ROI)
        person_img = img[y:y+h, x:x+w]
        
        if person_img.size != 0:
            resized_img = cv2.resize(person_img, (64, 128))
            
            positives.append(resized_img)
    
    updated_data = data.drop(rows_to_delete)
    updated_data.to_csv('updated_annotations.csv', index=False)
    
    return positives

def extract_negative_samples(img_folder, positive_size=0):
    negatives = []
    negative_filenames = next(os.walk(img_folder), (None, None, []))[2]
    files_count = len(negative_filenames)
    for index, filename in enumerate(negative_filenames):
        if positive_size > 0 and index + 1 > positive_size: break
        img_path = os.path.join(img_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Image at {img_path} could not be loaded.")
            continue
        # Ensure that negative samples do not include persons (needs proper implementation)
        negatives.append(cv2.resize(img, (64, 128)))
        if (index > 0 and (index + 1) % 100 == 0): print(f"extracted {index + 1} negative samples")
        if (index + 1 == files_count or index + 1 == positive_size): print(f"extracted {index + 1} negative samples")
    return negatives

def extract_more_negative_samples(img_folder, max_extracted_samples = 0):
    negatives = []
    index = 1
    for filename in os.listdir(img_folder):
        image_path = os.path.join(img_folder, filename)
        image = cv2.imread(image_path)
        h, w, _ = image.shape
        image = cv2.resize(image, (max(w, 65), max(h, 129)))
        h, w, _ = image.shape
        for _ in range(25):  # Extract multiple random samples from each negative image
            if max_extracted_samples > 0 and index == max_extracted_samples: return negatives
            xmin = np.random.randint(0, w - 64)
            ymin = np.random.randint(0, h - 128)
            xmax = xmin + 64
            ymax = ymin + 128
            img = image[int(ymin):int(ymax), int(xmin):int(xmax)]
            negatives.append(img)
            if index % 100 == 0: print(f"extracted {index} negative samples")
            index += 1
    return negatives

def compute_hog_features(images):
    hog = cv2.HOGDescriptor(_winSize=(64, 128), _blockSize=(16, 16), _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)
    features = []
    img_count = len(images)
    divider = 100 if img_count <= 5000 else 1000
    for index, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hog_features = hog.compute(gray)
        features.append(hog_features)
        if (index > 0 and (index + 1) % divider == 0): print(f"computed HOG for {index + 1} images")
        if (index + 1 == img_count): print(f"computed HOG for {index + 1} images")
    return features

def augment_samples(samples):
    augmented_samples = []
    for img in samples:
        # Original
        augmented_samples.append(img)
        # Flipped
        augmented_samples.append(cv2.flip(img, 1))
        # Rotated
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        for angle in [90, 180, 270]:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h))
            augmented_samples.append(rotated)
    return augmented_samples

# Paths to your JSON and image folders
train_json = 'F:/MastersDiploma/inria_dataset/human.big.coco_64000/labels.json'
train_img_folder = 'F:/MastersDiploma/inria_dataset/human.big.coco_64000/data'

big_train_data_route = 'F:/MastersDiploma/inria_dataset/human.big/data'
big_train_data_all_annotations_route = 'F:/MastersDiploma/inria_dataset/human.big/detections.csv'
big_train_data_annotations_route = 'F:/MastersDiploma/inria_dataset/human.big/updated_annotations.csv'

negative_img_folder = 'F:/MastersDiploma/inria_dataset/non-human.big/PASS_dataset/general'

# Load annotations
train_data = load_annotations(train_json)
print("loaded annotations")

# Extract samples
max_extracted_samples = 100000
# positive_samples = extract_positive_samples_from_csv(big_train_data_annotations_route, big_train_data_route)
# positive_samples = augment_samples(positive_samples)
negative_samples = extract_more_negative_samples(negative_img_folder, max_extracted_samples)
positive_samples = extract_positive_samples(train_data, train_img_folder, max_extracted_samples)
print(f"extracted all samples {len(positive_samples)} {len(negative_samples)}")

# Compute HOG features
positive_features = compute_hog_features(positive_samples)
negative_features = compute_hog_features(negative_samples)

# Reshape features for SVM
positive_features = np.array(positive_features, dtype=np.float32).reshape(len(positive_features), -1)
negative_features = np.array(negative_features, dtype=np.float32).reshape(len(negative_features), -1)

# Create labels
labels = np.hstack((np.ones(len(positive_features)), np.zeros(len(negative_features))))

# Stack features and labels
features = np.vstack((positive_features, negative_features))

# Train Linear SVM
svm = LinearSVC(max_iter=10000)
svm.fit(features, labels)

predictions = svm.predict(features)
accuracy = accuracy_score(labels, predictions)
print(f"Training Accuracy: {accuracy}")

conf_matrix = confusion_matrix(labels, predictions)
print(f"Confusion Matrix:\n{conf_matrix}")

print("calculated svm")

# Save the model
dump(svm, 'hog_svm_model.joblib')
print("created model.joblib")
