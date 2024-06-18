import cv2
import numpy as np
from joblib import load
import argparse

def main(file, result_name, inverted):
    if inverted == True:
        # Load the trained inverted SVM model
        svm = load('inverted_hog_svm_model.joblib')
        print("using inverted model")
    else:
        # Load the trained SVM model
        svm = load('hog_svm_model.joblib')
        
    # Get the support vectors and intercept from the trained SVM
    svm_coefs = svm.coef_.ravel()
    svm_intercept = svm.intercept_[0]

    # Set up HOGDescriptor with trained SVM
    hog = cv2.HOGDescriptor()

    svm_detector = np.append(svm_coefs, svm_intercept)

    expected_detector_size = hog.getDescriptorSize() + 1
    if len(svm_detector) != expected_detector_size:
        print(f"Error: Detector size mismatch. Expected {expected_detector_size}, got {len(svm_detector)}")
        exit()
    else: hog.setSVMDetector(svm_detector)
    
    def detect_person(img, hog):
        if img is None:
            print("Error: Image could not be loaded.")
            return img
        rects, _ = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.05)
        for (x, y, w, h) in rects:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return img

    # Test image path
    test_img = cv2.imread(file)
    result_img = detect_person(test_img, hog)

    # Display the result
    if result_img is not None:
        cv2.imwrite(f"{result_name}.jpg", result_img)
    else:
        print("Test image could not be processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that finds poeple on image.")
    parser.add_argument("-file", required=True, type=str, help="Represents ABSOLUTE path to the target image")
    parser.add_argument("-n", required=True, type=str, help="Name to save result image as (without file extension)")
    parser.add_argument("-inv", action="store_true", help="Use the inverted model (optional)")

    args = parser.parse_args()

    main(args.file, args.n, args.inv)