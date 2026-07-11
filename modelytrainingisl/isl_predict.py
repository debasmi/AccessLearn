

import os
import sys
import json
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn


mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

class ISLNet(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 256),       nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.net(x)

def extract_landmarks(image_bgr, hands):
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if not results.multi_hand_landmarks:
        return None, results

    lm = results.multi_hand_landmarks[0].landmark
    xs = np.array([p.x for p in lm])
    ys = np.array([p.y for p in lm])
    xs -= xs[0]; ys -= ys[0]
    span = max(xs.max() - xs.min(), ys.max() - ys.min(), 1e-6)
    xs /= span;  ys /= span

    feats = np.concatenate([xs, ys]).astype(np.float32)
    return feats, results

class ISLPredictor:
    def __init__(self, model_dir):
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
            self.le = pickle.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ISLNet(cfg["input_dim"], cfg["num_classes"]).to(self.device)
        self.model.load_state_dict(
            torch.load(os.path.join(model_dir, "isl_best_model.pt"), map_location=self.device)
        )
        self.model.eval()
        print(f"✅  Model loaded | {cfg['num_classes']} classes | device={self.device}")

    def predict(self, feats: np.ndarray):
        """feats: shape (42,)"""
        x = torch.tensor(feats).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        top3_idx  = probs.argsort()[::-1][:3]
        top3 = [(self.le.classes_[i], float(probs[i])) for i in top3_idx]
        return top3


def predict_image(predictor, image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌  Cannot read image: {image_path}")
        return

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                        min_detection_confidence=0.3) as hands:
        feats, results = extract_landmarks(img, hands)

    if feats is None:
        print("❌  No hand detected in image.")
        return

    top3 = predictor.predict(feats)
    print(f"\n🖐️  Prediction for: {image_path}")
    for rank, (cls, prob) in enumerate(top3, 1):
        bar = "█" * int(prob * 30)
        print(f"  #{rank}  {cls:>3}  {prob:.3f}  {bar}")

    if results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(img, results.multi_hand_landmarks[0],
                                  mp_hands.HAND_CONNECTIONS,
                                  mp_draw_styles.get_default_hand_landmarks_style(),
                                  mp_draw_styles.get_default_hand_connections_style())

    pred_label, pred_conf = top3[0]
    cv2.putText(img, f"{pred_label}  {pred_conf:.2f}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 0), 3)

    out_path = "isl_result.jpg"
    cv2.imwrite(out_path, img)
    print(f"\n💾  Annotated image saved → {out_path}")


def run_webcam(predictor):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌  Cannot open webcam.")
        return

    print("\n📷  Webcam live — press Q to quit\n")
    SMOOTHING = 5
    history   = []

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            feats, results = extract_landmarks(frame, hands)

            label, conf = "—", 0.0
            if feats is not None:
                top3 = predictor.predict(feats)
                history.append(top3[0][0])
                if len(history) > SMOOTHING:
                    history.pop(0)
                from collections import Counter
                label = Counter(history).most_common(1)[0][0]
                conf  = top3[0][1]

                if results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS,
                        mp_draw_styles.get_default_hand_landmarks_style(),
                        mp_draw_styles.get_default_hand_connections_style())

                
                for i, (cls, p) in enumerate(top3):
                    y = 60 + i * 40
                    cv2.putText(frame, f"{cls}: {p:.2f}", (frame.shape[1]-160, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 100), 2)

         
            cv2.rectangle(frame, (0, 0), (220, 80), (0, 0, 0), -1)
            cv2.putText(frame, label, (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 4)
            cv2.putText(frame, f"{conf:.2f}", (140, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 220), 2)

            cv2.imshow("ISL Recognition — press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="ISL Sign Language Inference")
    parser.add_argument("--model_dir", required=True, help="Folder with isl_best_model.pt etc.")
    parser.add_argument("--image",  default=None, help="Path to a single image to predict")
    parser.add_argument("--webcam", action="store_true", help="Run live webcam inference")
    args = parser.parse_args()

    if not args.image and not args.webcam:
        parser.error("Provide --image <path> or --webcam")

    predictor = ISLPredictor(args.model_dir)

    if args.image:
        predict_image(predictor, args.image)
    if args.webcam:
        run_webcam(predictor)


if __name__ == "__main__":
    main()
