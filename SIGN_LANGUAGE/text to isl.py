import cv2
import os
import time

# ---------- STEP 1: Folder setup ----------
# Change this path to where your "Indian" dataset is located
DATASET_PATH = "/Users/debasmibasu/Documents/SIS/SIGN LANGUAGE/Indian"

# ---------- STEP 2: Convert sentence to ISL gloss ----------
def to_gloss(sentence):
    sentence = sentence.lower().replace('?', '').replace('.', '')
    words = sentence.split()
    
    # Simple rule-based gloss mapping (expand later)
    mapping = {
        "what": "WHAT",
        "your": "YOUR",
        "name": "NAME",
        "is": "",  # often dropped in ISL
        "my": "MY",
        "how": "HOW",
        "you": "YOU",
        "are": "",
    }
    
    gloss = [mapping.get(w, w.upper()) for w in words if mapping.get(w, w.upper())]
    return ' '.join(gloss)

# ---------- STEP 3: Show sign images ----------
def show_sign_images(gloss_sentence):
    for word in gloss_sentence.split():
        print(f"Showing sign for: {word}")
        for letter in word:
            folder_path = os.path.join(DATASET_PATH, letter.upper())
            if not os.path.exists(folder_path):
                print(f"[!] Folder not found for {letter.upper()}")
                continue

            # Pick first image in that folder
            images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not images:
                print(f"[!] No images in {folder_path}")
                continue

            img_path = os.path.join(folder_path, images[0])
            img = cv2.imread(img_path)

            if img is not None:
                cv2.imshow("ISL Sign", img)
                cv2.waitKey(1000)  # show for 1 second per letter
            else:
                print(f"[!] Could not load image: {img_path}")
    cv2.destroyAllWindows()

# ---------- STEP 4: Run program ----------
if __name__ == "__main__":
    sentence = input("Enter a sentence: ")
    gloss = to_gloss(sentence)
    print(f"\nConverted to ISL Gloss: {gloss}\n")
    show_sign_images(gloss)
