"""
ISL Question-Answer Pipeline
==============================
Full loop:
  1. Capture ISL signs via webcam  →  decoded text (question)
  2. Send question to Gemini API   →  text answer
  3. Display answer as ISL signs   →  animated hand graphics

Usage:
  python isl_qa_pipeline.py \
      --model_dir  isl_model \
      --data_dir   /path/to/Indian \
      --gemini_key YOUR_GEMINI_KEY          # or set env var GEMINI_API_KEY

Controls (during sign capture):
  SPACE   →  finish signing / submit question
  ENTER   →  same as SPACE
  BACKSPACE → delete last recognised character
  Q / ESC →  quit

Controls (during sign playback):
  SPACE   →  pause / resume
  Q / ESC →  skip to next character / stop
"""

import os
import sys
import json
import time
import pickle
import argparse
import warnings
import requests
from collections import Counter, deque

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# GEMINI CONFIGURATION  — replace placeholder or pass --gemini_key
# -----------------------------------------------------------------------------
GEMINI_API_KEY_PLACEHOLDER = "ownapikey"
GEMINI_MODEL               = "gemini-2.0-flash"
GEMINI_ENDPOINT            = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
# -----------------------------------------------------------------------------

mp_hands       = mp.solutions.hands
mp_drawing     = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =============================================================================
# 1.  ISL Recognition Model
# =============================================================================

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
    return np.concatenate([xs, ys]).astype(np.float32), results


class ISLPredictor:
    def __init__(self, model_dir: str):
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
            self.le = pickle.load(f)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = ISLNet(cfg["input_dim"], cfg["num_classes"]).to(self.device)
        self.model.load_state_dict(
            torch.load(os.path.join(model_dir, "isl_best_model.pt"),
                       map_location=self.device)
        )
        self.model.eval()
        print(f"Recogniser loaded — {cfg['num_classes']} classes | {self.device}")

    def predict(self, feats: np.ndarray):
        x = torch.tensor(feats).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1).squeeze().cpu().numpy()
        top3_idx = probs.argsort()[::-1][:3]
        return [(self.le.classes_[i], float(probs[i])) for i in top3_idx]


# =============================================================================
# 2.  Gemini
# =============================================================================

def ask_gemini(question: str, api_key: str) -> str:
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": (
            "You are answering a question asked in Indian Sign Language. "
            "Give a SHORT, clear answer suitable for sign language display — "
            "simple words, no punctuation except spaces, maximum 30 words.\n\n"
            f"Question: {question}"
        )}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 120},
    }
    try:
        resp = requests.post(f"{GEMINI_ENDPOINT}?key={api_key}",
                             headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip().upper()
        return "".join(c for c in text if c.isalnum() or c == " ") or "SORRY NO ANSWER"
    except requests.exceptions.HTTPError:
        return f"GEMINI ERROR {resp.status_code}"
    except Exception:
        return "CONNECTION ERROR"


# =============================================================================
# 3.  Sign Output — animated hand graphics
# =============================================================================

W, H    = 720, 760
HAND_W  = 460
HAND_H  = 460
HAND_X  = (W - HAND_W) // 2
HAND_Y  = 60
PANEL_Y = HAND_Y + HAND_H + 20

BG = (18, 18, 24);   PALM_COLOR = (55, 55, 75);   BONE_COLOR = (100, 210, 180)
JOINT_COLOR = (255, 255, 255);  TIP_COLOR = (80, 200, 255)
ACCENT = (80, 200, 140);  DIM = (90, 90, 110);  WHITE = (240, 240, 240)
SHADOW = (35, 35, 50)

FINGER_CONNECTIONS = [
    [(0,1),(1,2),(2,3),(3,4)], [(0,5),(5,6),(6,7),(7,8)],
    [(0,9),(9,10),(10,11),(11,12)], [(0,13),(13,14),(14,15),(15,16)],
    [(0,17),(17,18),(18,19),(19,20)],
]
PALM_CONNECTIONS = [(0,5),(5,9),(9,13),(13,17),(0,17)]
FINGERTIPS = [4, 8, 12, 16, 20];  KNUCKLES = [5, 9, 13, 17]


def _lm_to_pts(landmarks, t=0.0):
    xs = np.array([p[0] for p in landmarks])
    ys = np.array([p[1] for p in landmarks])
    m  = 0.12
    xs = (xs - xs.min()) / (xs.max() - xs.min() + 1e-6)
    ys = (ys - ys.min()) / (ys.max() - ys.min() + 1e-6)
    px = (xs * (1 - 2*m) + m) * HAND_W + HAND_X + np.cos(t * 0.9) * 3
    py = (ys * (1 - 2*m) + m) * HAND_H + HAND_Y + np.sin(t * 1.8) * 6
    return px.astype(int), py.astype(int)


def _draw_hand(canvas, px, py, t=0.0, glow=0.0):
    ox, oy = 12, 14
    for finger in FINGER_CONNECTIONS:
        for a, b in finger:
            cv2.line(canvas, (px[a]+ox, py[a]+oy), (px[b]+ox, py[b]+oy), SHADOW, 12, cv2.LINE_AA)
    for a, b in PALM_CONNECTIONS:
        cv2.line(canvas, (px[a]+ox, py[a]+oy), (px[b]+ox, py[b]+oy), SHADOW, 14, cv2.LINE_AA)
    hull = cv2.convexHull(np.array([[px[i], py[i]] for i in [0,1,5,9,13,17]], dtype=np.int32))
    cv2.fillConvexPoly(canvas, hull, PALM_COLOR)
    cv2.polylines(canvas, [hull], True, BONE_COLOR, 2, cv2.LINE_AA)
    for finger in FINGER_CONNECTIONS:
        for si, (a, b) in enumerate(finger):
            cv2.line(canvas, (px[a], py[a]), (px[b], py[b]),
                     BONE_COLOR, max(2, 9 - si*2), cv2.LINE_AA)
    for i in range(21):
        if i in FINGERTIPS:
            g = int(abs(np.sin(glow + i*0.7)) * 12)
            cv2.circle(canvas, (px[i], py[i]), 11+g, TIP_COLOR, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px[i], py[i]), 8, WHITE, 1, cv2.LINE_AA)
        elif i in KNUCKLES:
            cv2.circle(canvas, (px[i], py[i]), 7, JOINT_COLOR, -1, cv2.LINE_AA)
        elif i == 0:
            cv2.circle(canvas, (px[i], py[i]), 10, PALM_COLOR, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px[i], py[i]), 10, BONE_COLOR, 2, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (px[i], py[i]), 5, JOINT_COLOR, -1, cv2.LINE_AA)


def _render_sign_frame(landmarks, char, pos, total, full_text, t, glow, transition=1.0):
    canvas = np.full((H, W, 3), BG, dtype=np.uint8)
    for gx in range(0, W, 40): cv2.line(canvas, (gx, 0), (gx, H), (28,28,36), 1)
    for gy in range(0, H, 40): cv2.line(canvas, (0, gy), (W, gy), (28,28,36), 1)
    cv2.rectangle(canvas, (HAND_X-10, HAND_Y-10), (HAND_X+HAND_W+10, HAND_Y+HAND_H+10), (40,40,60), 2)

    if landmarks:
        _draw_hand(canvas, *_lm_to_pts(landmarks, t), t, glow)
    else:
        cv2.putText(canvas, "?", (W//2-30, HAND_Y+HAND_H//2+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, DIM, 6, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    badge = 90
    cv2.rectangle(canvas, (20, 20), (20+badge, 20+badge), ACCENT, -1)
    fs = 2.5 if len(char) == 1 else 1.2
    (tw, th), _ = cv2.getTextSize(char, font, fs, 3)
    cv2.putText(canvas, char, (20+(badge-tw)//2, 20+(badge+th)//2-2), font, fs, BG, 3, cv2.LINE_AA)
    cv2.putText(canvas, "ISL", (20, 128), font, 0.5, DIM, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{pos}/{total}", (W-110, 52), font, 0.7, DIM, 1, cv2.LINE_AA)
    bx, by = W-110, 62
    cv2.rectangle(canvas, (bx, by), (bx+90, by+5), (50,50,70), -1)
    cv2.rectangle(canvas, (bx, by), (bx+int(90*pos/max(total,1)), by+5), ACCENT, -1)

    strip_y = PANEL_Y + 10
    cv2.line(canvas, (40, strip_y-10), (W-40, strip_y-10), (45,45,65), 1)
    cw = 28;  sx = max(40, (W - len(full_text)*cw)//2)
    for i, c in enumerate(full_text):
        x = sx + i*cw
        if x > W-30: break
        if i == pos-1:
            cv2.rectangle(canvas, (x-4, strip_y-2), (x+cw-8, strip_y+28), ACCENT, -1)
            cv2.putText(canvas, c, (x, strip_y+22), font, 0.85, BG, 2, cv2.LINE_AA)
        else:
            cv2.putText(canvas, c if c != ' ' else "·", (x, strip_y+22), font, 0.7,
                        (180,180,200) if i < pos-1 else DIM, 1, cv2.LINE_AA)

    cv2.putText(canvas, "SPACE pause   Q exit", (W//2-100, H-18), font, 0.45, DIM, 1, cv2.LINE_AA)
    if transition < 1.0:
        cv2.addWeighted(np.full_like(canvas, BG), 1-transition, canvas, transition, 0, canvas)
    return canvas


def get_landmark_for_char(data_dir: str, char: str):
    import random
    folder = os.path.join(data_dir, char)
    if not os.path.isdir(folder): return None
    imgs = [os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS]
    if not imgs: return None
    random.shuffle(imgs)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                        min_detection_confidence=0.3) as h:
        for p in imgs[:10]:
            img = cv2.imread(p)
            if img is None: continue
            r = h.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if r.multi_hand_landmarks:
                lm = r.multi_hand_landmarks[0].landmark
                return [(pt.x, pt.y, pt.z) for pt in lm]
    return None


def build_sign_index(data_dir: str) -> dict:
    index = {}
    print("\nBuilding sign landmark index …")
    for cls in sorted(os.listdir(data_dir)):
        if not os.path.isdir(os.path.join(data_dir, cls)): continue
        lm = get_landmark_for_char(data_dir, cls)
        if lm: index[cls.upper()] = lm
    print(f"{len(index)} signs indexed\n")
    return index


def display_answer_as_signs(text: str, sign_index: dict, speed: float = 1.2):
    text_upper = text.upper()
    chars      = list(text_upper)
    if not chars: return

    win = "ISL Answer  —  Q to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)
    sign_count = sum(1 for c in chars if c != ' ')
    sign_pos = 0; paused = False; i = 0

    while i < len(chars):
        char = chars[i]
        if char == ' ':
            t_end = time.time() + speed * 0.5
            while time.time() < t_end:
                canvas = np.full((H, W, 3), BG, dtype=np.uint8)
                cv2.putText(canvas, "[ WORD BREAK ]", (W//2-120, H//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, DIM, 2, cv2.LINE_AA)
                cv2.imshow(win, canvas)
                if cv2.waitKey(30) & 0xFF in (ord('q'), ord('Q'), 27):
                    cv2.destroyAllWindows(); return
            i += 1; continue

        sign_pos += 1
        landmarks = sign_index.get(char)
        t_start   = time.time()

        while True:
            elapsed = time.time() - t_start
            if elapsed >= speed and not paused: break
            t   = time.time()
            frame = _render_sign_frame(landmarks, char, sign_pos, sign_count,
                                       text_upper, t, t*3.0, min(1.0, elapsed/0.2))
            cv2.imshow(win, frame)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), ord('Q'), 27): cv2.destroyAllWindows(); return
            elif key == ord(' '): paused = not paused
        i += 1

    done = np.full((H, W, 3), BG, dtype=np.uint8)
    cv2.putText(done, "Done!", (W//2-80, H//2-20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, ACCENT, 3, cv2.LINE_AA)
    cv2.putText(done, text_upper[:50], (40, H//2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 1, cv2.LINE_AA)
    cv2.putText(done, "Press any key for next question", (40, H//2+90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, DIM, 1, cv2.LINE_AA)
    cv2.imshow(win, done); cv2.waitKey(0); cv2.destroyAllWindows()


# =============================================================================
# 4.  Sign Input — webcam question capture
# =============================================================================

HUD_BG  = (20, 20, 30);   HUD_GRN = (80, 220, 140);  HUD_YEL = (80, 200, 220)
HUD_WHT = (230, 230, 230); HUD_DIM = (100, 100, 120)


def _draw_capture_hud(frame, question_so_far, current_char, conf, status):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 90), HUD_BG, -1)
    cv2.putText(frame, current_char if current_char else "—", (20, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 2.8, HUD_GRN, 4, cv2.LINE_AA)
    bx, by, bw = 130, 30, 180
    cv2.rectangle(frame, (bx, by), (bx+bw, by+20), (50,50,70), -1)
    cv2.rectangle(frame, (bx, by), (bx+int(bw*conf), by+20), HUD_GRN, -1)
    cv2.putText(frame, f"{conf*100:.0f}%", (bx+bw+8, by+16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, HUD_DIM, 1, cv2.LINE_AA)
    cv2.putText(frame, ("Q: " + question_so_far)[-48:], (10, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, HUD_YEL, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, h-38), (w, h), HUD_BG, -1)
    cv2.putText(frame, "SPACE/ENTER submit   BACKSPACE delete   Q quit",
                (10, h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, HUD_DIM, 1, cv2.LINE_AA)
    cv2.putText(frame, status, (w//2-120, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, HUD_WHT, 1, cv2.LINE_AA)


def capture_question_from_signs(predictor: ISLPredictor):
    """
    Webcam loop: hold a sign for HOLD_FRAMES to lock it in.
    SPACE/ENTER submits; BACKSPACE deletes last letter; Q/ESC quits.
    Returns decoded question string, or None to quit.
    """
    HOLD_FRAMES = 18    # frames to hold before locking (~0.6 s at 30 fps)
    SMOOTH_WIN  = 8     # majority-vote window
    CONF_THRESH = 0.65  # min confidence

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam."); return None

    question = []; history = deque(maxlen=SMOOTH_WIN)
    hold_count = 0; last_locked = None
    status = "Sign a letter and hold it to type"

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            feats, results = extract_landmarks(frame, hands)
            current_char = ""; conf = 0.0

            if feats is not None:
                top3 = predictor.predict(feats)
                best_label, best_conf = top3[0]
                if best_conf >= CONF_THRESH:
                    history.append(best_label)
                else:
                    history.clear()

                if len(history) == SMOOTH_WIN:
                    voted = Counter(history).most_common(1)[0][0]
                    current_char = voted; conf = best_conf
                    if voted == last_locked:
                        hold_count += 1
                    else:
                        hold_count = 1; last_locked = voted
                    if hold_count == HOLD_FRAMES:
                        question.append(voted)
                        hold_count = 0; history.clear()
                        status = f"Added '{voted}' — sign next letter"
                else:
                    hold_count = 0; last_locked = None

                if results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.multi_hand_landmarks[0],
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw_styles.get_default_hand_landmarks_style(),
                        mp_draw_styles.get_default_hand_connections_style())

                if hold_count > 0:
                    cx, cy = frame.shape[1]//2, frame.shape[0]//2
                    angle = int(360 * hold_count / HOLD_FRAMES)
                    cv2.ellipse(frame, (cx, cy), (70, 70), -90, 0, angle, HUD_GRN, 6, cv2.LINE_AA)

                for rank, (cls, p) in enumerate(top3):
                    cv2.putText(frame, f"{cls}: {p:.2f}",
                                (frame.shape[1]-160, 120+rank*36),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, HUD_YEL, 2)

            _draw_capture_hud(frame, "".join(question), current_char, conf, status)
            cv2.imshow("ISL Input  —  Sign your question", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord(' '), 13):           # SPACE or ENTER
                if question: break
                else: status = "Nothing signed yet!"
            elif key == 8:                       # BACKSPACE
                if question:
                    removed = question.pop(); status = f"Removed '{removed}'"
                    hold_count = 0; history.clear()
            elif key in (ord('q'), ord('Q'), 27):
                cap.release(); cv2.destroyAllWindows(); return None

    cap.release(); cv2.destroyAllWindows()
    return "".join(question) if question else None


# =============================================================================
# 5.  Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ISL Q&A Pipeline")
    parser.add_argument("--model_dir",  required=True,
                        help="Folder with isl_best_model.pt, config.json, label_encoder.pkl")
    parser.add_argument("--data_dir",   required=True,
                        help="Indian/ dataset folder (for sign display)")
    parser.add_argument("--gemini_key", default=None,
                        help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--speed",      type=float, default=1.2,
                        help="Seconds per sign during answer playback (default 1.2)")
    args = parser.parse_args()

    api_key = args.gemini_key or os.environ.get("GEMINI_API_KEY") or GEMINI_API_KEY_PLACEHOLDER
    if api_key == GEMINI_API_KEY_PLACEHOLDER:
        print("No Gemini API key provided.")
        print("Pass --gemini_key YOUR_KEY  or  export GEMINI_API_KEY=YOUR_KEY")
        sys.exit(1)

    print("=" * 56)
    print("  ISL Question → Gemini → ISL Answer Pipeline")
    print("=" * 56)

    predictor  = ISLPredictor(args.model_dir)
    sign_index = build_sign_index(args.data_dir)

    session = 0
    while True:
        session += 1
        print(f"\n--- Round {session} | Sign your question (SPACE to submit, Q to quit) ---\n")

        # Step 1: capture question via webcam
        question = capture_question_from_signs(predictor)
        if question is None:
            print("\nGoodbye!"); break

        print(f"\nQuestion decoded: '{question}'")

        # Step 2: query Gemini
        print("Querying Gemini …")
        answer = ask_gemini(question, api_key)
        print(f"Gemini answer:    '{answer}'")

        # Step 3: display answer as ISL signs
        print("Displaying answer as ISL signs …\n")
        display_answer_as_signs(answer, sign_index, speed=args.speed)

        print("\nPress Enter for another question, or type 'q' to quit.")
        try:
            choice = input("→ ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            choice = "q"
        if choice in ("q", "quit", "exit"):
            print("Goodbye!"); break

    print("\nPipeline exited cleanly.")


if __name__ == "__main__":
    main()
