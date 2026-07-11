import cv2
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# ---------- CONFIGURATION ----------
DATASET_PATH = "/Users/debasmibasu/Documents/SIS/SIGN LANGUAGE/Indian"
MODEL_PATH = "isl_model.pkl"  # For saving/loading trained features

# ---------- STEP 1: Feature Extraction ----------
def extract_features(image):
    """
    Extract features from a hand sign image using basic edge-based representation.
    """
    img_resized = cv2.resize(image, (64, 64))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges.flatten()

# ---------- STEP 2: Load and Train Model ----------
def train_model():
    """
    Train feature-based model using images from dataset.
    """
    print("Training model from dataset...")
    letter_features = {}

    for letter in os.listdir(DATASET_PATH):
        letter_path = os.path.join(DATASET_PATH, letter)
        if not os.path.isdir(letter_path):
            continue

        print(f"Loading images for letter: {letter}")
        features_list = []

        images = [img for img in os.listdir(letter_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_name in images[:20]:  # Use first 20 images per letter
            img_path = os.path.join(letter_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                features = extract_features(img)
                features_list.append(features)

        if features_list:
            letter_features[letter] = np.mean(features_list, axis=0)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(letter_features, f)

    print(f"Model trained and saved to {MODEL_PATH}")
    return letter_features

# ---------- STEP 3: Load Model ----------
def load_model():
    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        print("No existing model found. Training new model...")
        return train_model()

# ---------- STEP 4: Predict Letter ----------
def predict_letter(frame, letter_features):
    """
    Predict which ISL letter the given frame represents.
    """
    features = extract_features(frame)
    best_match, best_score = None, -1

    for letter, ref_features in letter_features.items():
        similarity = cosine_similarity([features], [ref_features])[0][0]
        if similarity > best_score:
            best_score = similarity
            best_match = letter

    return best_match, best_score

# ---------- STEP 5: Hand Detection ----------
def detect_hand_region(frame):
    """
    Detect hand region based on skin color segmentation (HSV).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result, mask

def is_hand_present(frame):
    """
    Check if a significant portion of the frame contains skin tone pixels.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    skin_pixels = cv2.countNonZero(mask)
    total_pixels = frame.shape[0] * frame.shape[1]
    skin_percentage = (skin_pixels / total_pixels) * 100

    return skin_percentage > 5, skin_percentage

# ---------- STEP 6: Main Recognition Loop ----------
def recognize_signs():
    """
    Capture webcam feed and perform ISL letter recognition.
    """
    letter_features = load_model()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("\n=== ISL Letter Recognition ===")
    print("Show hand signs to the camera")
    print("Press 'q' to quit | 'r' retrain | 'c' clear word")
    print("================================\n")

    recognized_word = ""
    last_letter = None
    stable_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        height, width = frame.shape[:2]

        hand_present, skin_percentage = is_hand_present(frame)
        predicted_letter, confidence = None, 0

        if hand_present:
            roi = frame.copy()
            predicted_letter, confidence = predict_letter(roi, letter_features)

            if predicted_letter == last_letter:
                stable_count += 1
            else:
                stable_count = 0
                last_letter = predicted_letter

            if stable_count == 15 and predicted_letter:
                if not recognized_word or recognized_word[-1] != predicted_letter:
                    recognized_word += predicted_letter
                    print(f"Added letter: {predicted_letter} | Word so far: {recognized_word}")

            # Color-code confidence levels
            if confidence > 0.7:
                color = (0, 255, 0)
            elif confidence > 0.4:
                color = (0, 255, 255)
            else:
                color = (0, 165, 255)

            cv2.putText(display_frame, f"Letter: {predicted_letter} ({confidence:.2f})", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(display_frame, f"Hand detected ({skin_percentage:.1f}%)", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            stable_count = 0
            last_letter = None
            cv2.putText(display_frame, "Waiting for hand...", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 150, 150), 3)

        # Display recognized word
        cv2.putText(display_frame, f"Word: {recognized_word}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_frame, "Q: Quit | R: Retrain | C: Clear",
                    (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('ISL Letter Recognition', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("\nRetraining model...")
            letter_features = train_model()
            print("Model retrained!")
        elif key == ord('c'):
            recognized_word = ""
            print("Word cleared!")

    cap.release()
    cv2.destroyAllWindows()

    if recognized_word:
        print(f"\nFinal recognized word: {recognized_word}")

# ---------- STEP 7: Run Program ----------
if __name__ == "__main__":
    recognize_signs()
