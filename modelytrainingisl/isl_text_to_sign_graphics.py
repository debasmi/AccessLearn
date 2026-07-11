"""
Indian Sign Language (ISL) - Text to Sign with ANIMATED HAND GRAPHICS
======================================================================
Draws a real animated hand skeleton using MediaPipe landmarks —
no raw images, just clean vector hand graphics rendered with OpenCV.

Usage:
  python isl_text_to_sign_graphics.py --data_dir "/path/to/Indian"
  python isl_text_to_sign_graphics.py --data_dir "/path/to/Indian" --text "HELLO"
  python isl_text_to_sign_graphics.py --data_dir "/path/to/Indian" --speed 1.5
"""

import os
import sys
import time
import random
import argparse
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands

# ── Canvas config ─────────────────────────────────────────────────────────────
W, H         = 720, 760
HAND_W       = 460   # hand drawing area
HAND_H       = 460
HAND_X       = (W - HAND_W) // 2
HAND_Y       = 60
PANEL_Y      = HAND_Y + HAND_H + 20

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Colors (BGR) ─────────────────────────────────────────────────────────────
BG           = (18,  18,  24)
PALM_COLOR   = (55,  55,  75)
BONE_COLOR   = (100, 210, 180)
JOINT_COLOR  = (255, 255, 255)
TIP_COLOR    = (80,  200, 255)
ACCENT       = (80,  200, 140)
DIM          = (90,  90,  110)
WHITE        = (240, 240, 240)
SHADOW       = (35,  35,  50)

# MediaPipe hand connection groups
FINGER_CONNECTIONS = [
    # thumb
    [(0,1),(1,2),(2,3),(3,4)],
    # index
    [(0,5),(5,6),(6,7),(7,8)],
    # middle
    [(0,9),(9,10),(10,11),(11,12)],
    # ring
    [(0,13),(13,14),(14,15),(15,16)],
    # pinky
    [(0,17),(17,18),(18,19),(19,20)],
]
PALM_CONNECTIONS = [(0,5),(5,9),(9,13),(13,17),(0,17)]
FINGERTIPS = [4, 8, 12, 16, 20]
KNUCKLES   = [5, 9, 13, 17]


# ════════════════════════════════════════════════════════════════════════════
# 1. Extract landmarks from dataset images
# ════════════════════════════════════════════════════════════════════════════

def get_landmarks_for_sign(data_dir: str, char: str):
    """
    Find a good image for this character and extract its 21 MediaPipe landmarks.
    Returns list of (x,y,z) normalized coords, or None.
    """
    folder = os.path.join(data_dir, char)
    if not os.path.isdir(folder):
        return None

    images = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
    ]
    if not images:
        return None

    random.shuffle(images)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                        min_detection_confidence=0.3) as hands:
        for img_path in images[:10]:  # try up to 10
            img = cv2.imread(img_path)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0].landmark
                return [(p.x, p.y, p.z) for p in lm]
    return None


def build_sign_landmark_index(data_dir: str) -> dict:
    """Build a dict: char -> list of (x,y,z) tuples"""
    index = {}
    chars = sorted(os.listdir(data_dir))
    print(f"\n📂  Scanning {len(chars)} classes for landmarks...\n")
    for cls in chars:
        if not os.path.isdir(os.path.join(data_dir, cls)):
            continue
        lm = get_landmarks_for_sign(data_dir, cls)
        if lm:
            index[cls.upper()] = lm
            print(f"  ✅ {cls.upper()}", end="  ", flush=True)
        else:
            print(f"  ❌ {cls.upper()}", end="  ", flush=True)
    print(f"\n\n✅  Landmark index built: {len(index)} signs\n")
    return index


# ════════════════════════════════════════════════════════════════════════════
# 2. Hand drawing engine
# ════════════════════════════════════════════════════════════════════════════

def landmarks_to_canvas_pts(landmarks, x0, y0, w, h, t=0.0):
    """
    Convert normalized mediapipe landmarks to pixel coords in the drawing area.
    Flips y (mediapipe y goes down, canvas y goes down but we want hand right-way-up).
    Applies a gentle idle float animation using t.
    """
    xs = np.array([p[0] for p in landmarks])
    ys = np.array([p[1] for p in landmarks])

    # Center & scale to drawing area with margin
    margin = 0.12
    xs = (xs - xs.min()) / (xs.max() - xs.min() + 1e-6)
    ys = (ys - ys.min()) / (ys.max() - ys.min() + 1e-6)

    px = (xs * (1 - 2*margin) + margin) * w + x0
    py = (ys * (1 - 2*margin) + margin) * h + y0

    # Gentle idle float
    float_y = np.sin(t * 1.8) * 6
    float_x = np.cos(t * 0.9) * 3
    px += float_x
    py += float_y

    return px.astype(int), py.astype(int)


def draw_shadow_hand(canvas, px, py, alpha=0.4):
    """Draw a blurred/offset shadow under the hand."""
    ox, oy = 12, 14
    for finger in FINGER_CONNECTIONS:
        for a, b in finger:
            p1 = (int(px[a]) + ox, int(py[a]) + oy)
            p2 = (int(px[b]) + ox, int(py[b]) + oy)
            cv2.line(canvas, p1, p2, SHADOW, 12, cv2.LINE_AA)
    for a, b in PALM_CONNECTIONS:
        cv2.line(canvas, (px[a]+ox, py[a]+oy), (px[b]+ox, py[b]+oy), SHADOW, 14, cv2.LINE_AA)


def draw_hand(canvas, px, py, t=0.0, glow_phase=0.0):
    """Draw a stylized vector hand skeleton."""

    # ── Shadow ────────────────────────────────────────────────────────────
    draw_shadow_hand(canvas, px, py)

    # ── Palm fill (convex hull of palm points) ────────────────────────────
    palm_pts = np.array([[px[i], py[i]] for i in [0, 1, 5, 9, 13, 17]], dtype=np.int32)
    hull = cv2.convexHull(palm_pts)
    cv2.fillConvexPoly(canvas, hull, PALM_COLOR)
    cv2.polylines(canvas, [hull], True, BONE_COLOR, 2, cv2.LINE_AA)

    # ── Finger bones ──────────────────────────────────────────────────────
    for finger_idx, finger in enumerate(FINGER_CONNECTIONS):
        for seg_idx, (a, b) in enumerate(finger):
            # Taper thickness from base to tip
            thickness = max(2, 9 - seg_idx * 2)
            color = BONE_COLOR
            cv2.line(canvas, (px[a], py[a]), (px[b], py[b]), color, thickness, cv2.LINE_AA)

    # ── Joints ────────────────────────────────────────────────────────────
    for i in range(21):
        if i in FINGERTIPS:
            # Animated glowing fingertips
            glow = int(abs(np.sin(glow_phase + i * 0.7)) * 12)
            cv2.circle(canvas, (px[i], py[i]), 11 + glow, (*TIP_COLOR, 60), -1, cv2.LINE_AA)
            cv2.circle(canvas, (px[i], py[i]), 8, TIP_COLOR, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px[i], py[i]), 8, WHITE, 1, cv2.LINE_AA)
        elif i in KNUCKLES:
            cv2.circle(canvas, (px[i], py[i]), 7, JOINT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px[i], py[i]), 7, BONE_COLOR, 1, cv2.LINE_AA)
        elif i == 0:
            # Wrist
            cv2.circle(canvas, (px[i], py[i]), 10, PALM_COLOR, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px[i], py[i]), 10, BONE_COLOR, 2, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (px[i], py[i]), 5, JOINT_COLOR, -1, cv2.LINE_AA)


# ════════════════════════════════════════════════════════════════════════════
# 3. Full frame renderer
# ════════════════════════════════════════════════════════════════════════════

def render_frame(landmarks, char, position, total, full_text, t, glow_phase, transition=1.0):
    """Render one animation frame."""
    canvas = np.full((H, W, 3), BG, dtype=np.uint8)

    # ── Background grid (subtle) ──────────────────────────────────────────
    grid_color = (28, 28, 36)
    for gx in range(0, W, 40):
        cv2.line(canvas, (gx, 0), (gx, H), grid_color, 1)
    for gy in range(0, H, 40):
        cv2.line(canvas, (0, gy), (W, gy), grid_color, 1)

    # ── Hand area card ────────────────────────────────────────────────────
    card_pad = 20
    cv2.rectangle(canvas,
                  (HAND_X - card_pad, HAND_Y - card_pad),
                  (HAND_X + HAND_W + card_pad, HAND_Y + HAND_H + card_pad),
                  (30, 30, 42), -1)
    cv2.rectangle(canvas,
                  (HAND_X - card_pad, HAND_Y - card_pad),
                  (HAND_X + HAND_W + card_pad, HAND_Y + HAND_H + card_pad),
                  (60, 60, 90), 1)

    # ── Draw hand ─────────────────────────────────────────────────────────
    if landmarks:
        px, py = landmarks_to_canvas_pts(landmarks, HAND_X, HAND_Y, HAND_W, HAND_H, t)
        alpha_lm = [(lm[0], lm[1], lm[2]) for lm in landmarks]
        draw_hand(canvas, px, py, t, glow_phase)
    else:
        # No landmark: show question mark
        cv2.putText(canvas, "?", (W//2 - 30, HAND_Y + HAND_H//2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, DIM, 4, cv2.LINE_AA)
        cv2.putText(canvas, "no sign data", (W//2 - 90, HAND_Y + HAND_H//2 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, DIM, 1, cv2.LINE_AA)

    # ── Character badge (top left) ────────────────────────────────────────
    badge_size = 72
    cv2.rectangle(canvas, (20, 20), (20 + badge_size, 20 + badge_size), ACCENT, -1)
    cv2.rectangle(canvas, (20, 20), (20 + badge_size, 20 + badge_size), WHITE, 1)
    font_scale = 2.8 if len(char) == 1 else 1.4
    (tw, th), _ = cv2.getTextSize(char, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)
    cv2.putText(canvas, char,
                (20 + (badge_size - tw)//2, 20 + (badge_size + th)//2 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, BG, 3, cv2.LINE_AA)

    # ── ISL label ─────────────────────────────────────────────────────────
    cv2.putText(canvas, "ISL", (20, 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, DIM, 1, cv2.LINE_AA)

    # ── Progress (top right) ──────────────────────────────────────────────
    prog = f"{position} / {total}"
    cv2.putText(canvas, prog, (W - 110, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, DIM, 1, cv2.LINE_AA)

    # ── Progress bar ──────────────────────────────────────────────────────
    bar_x, bar_y = W - 110, 62
    bar_w, bar_h = 90, 5
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 70), -1)
    filled = int(bar_w * position / max(total, 1))
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), ACCENT, -1)

    # ── Text strip (bottom) ───────────────────────────────────────────────
    strip_y = PANEL_Y + 10
    cv2.line(canvas, (40, strip_y - 10), (W - 40, strip_y - 10), (45, 45, 65), 1)

    char_w = 28
    text_px_total = len(full_text) * char_w
    start_x = max(40, (W - text_px_total) // 2)

    for i, c in enumerate(full_text):
        x = start_x + i * char_w
        if x > W - 30:
            break
        if i == (position - 1):
            # Current char highlighted
            cv2.rectangle(canvas, (x - 4, strip_y - 2), (x + char_w - 8, strip_y + 28), ACCENT, -1)
            cv2.putText(canvas, c, (x, strip_y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, BG, 2, cv2.LINE_AA)
        else:
            color = WHITE if c != ' ' else DIM
            cv2.putText(canvas, c if c != ' ' else "·", (x, strip_y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (180, 180, 200) if i < (position - 1) else DIM,
                        1, cv2.LINE_AA)

    # ── Hint ──────────────────────────────────────────────────────────────
    hint = "Q  exit     SPACE  pause"
    cv2.putText(canvas, hint, (W//2 - 110, H - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, DIM, 1, cv2.LINE_AA)

    # ── Transition fade-in ────────────────────────────────────────────────
    if transition < 1.0:
        overlay = np.full_like(canvas, BG)
        alpha = int((1.0 - transition) * 255)
        cv2.addWeighted(overlay, (1 - transition), canvas, transition, 0, canvas)

    return canvas


def render_space_frame(position, total, full_text, t):
    canvas = np.full((H, W, 3), BG, dtype=np.uint8)
    grid_color = (28, 28, 36)
    for gx in range(0, W, 40): cv2.line(canvas, (gx, 0), (gx, H), grid_color, 1)
    for gy in range(0, H, 40): cv2.line(canvas, (0, gy), (W, gy), grid_color, 1)

    # Animated word-break indicator
    for i in range(5):
        alpha = abs(np.sin(t * 2 + i * 0.7))
        x = W//2 - 80 + i * 40
        y = HAND_Y + HAND_H//2
        r = int(6 + alpha * 4)
        c = int(40 + alpha * 60)
        cv2.circle(canvas, (x, y), r, (c, c, c+20), -1, cv2.LINE_AA)

    cv2.putText(canvas, "word break", (W//2 - 65, HAND_Y + HAND_H//2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, DIM, 1, cv2.LINE_AA)

    strip_y = PANEL_Y + 10
    cv2.line(canvas, (40, strip_y - 10), (W - 40, strip_y - 10), (45, 45, 65), 1)
    char_w = 28
    start_x = max(40, (W - len(full_text) * char_w) // 2)
    for i, c in enumerate(full_text):
        x = start_x + i * char_w
        if x > W - 30: break
        col = (120, 120, 140) if i < position else DIM
        cv2.putText(canvas, c if c != ' ' else "·", (x, strip_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 1, cv2.LINE_AA)
    return canvas


def render_done_frame(full_text):
    canvas = np.full((H, W, 3), BG, dtype=np.uint8)
    grid_color = (28, 28, 36)
    for gx in range(0, W, 40): cv2.line(canvas, (gx, 0), (gx, H), grid_color, 1)
    for gy in range(0, H, 40): cv2.line(canvas, (0, gy), (W, gy), grid_color, 1)

    cv2.putText(canvas, "Done", (W//2 - 60, H//2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, ACCENT, 3, cv2.LINE_AA)
    cv2.putText(canvas, f'"{full_text}"', (W//2 - min(len(full_text)*9, W//2 - 40), H//2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, "press any key", (W//2 - 70, H//2 + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, DIM, 1, cv2.LINE_AA)
    return canvas


# ════════════════════════════════════════════════════════════════════════════
# 4. Animation loop
# ════════════════════════════════════════════════════════════════════════════

def show_text_as_signs(text: str, sign_index: dict, speed: float):
    text_upper = text.upper()
    chars = list(text_upper)
    if not chars:
        return

    window = "ISL Sign Language  —  press Q to quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, W, H)

    sign_chars = [c for c in chars if c != ' ']
    sign_count = len(sign_chars)
    sign_pos = 0
    paused = False

    i = 0
    while i < len(chars):
        char = chars[i]

        if char == ' ':
            start = time.time()
            while time.time() - start < speed * 0.6:
                t = time.time()
                frame = render_space_frame(i, len(chars), text_upper, t)
                cv2.imshow(window, frame)
                key = cv2.waitKey(30) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    cv2.destroyAllWindows()
                    return
            i += 1
            continue

        sign_pos += 1
        landmarks = sign_index.get(char)

        # Animate for `speed` seconds
        start = time.time()
        while True:
            elapsed = time.time() - start
            if elapsed >= speed and not paused:
                break

            t = time.time()
            transition = min(1.0, elapsed / 0.2)   # 200ms fade-in
            glow_phase = t * 3.0

            frame = render_frame(
                landmarks, char,
                sign_pos, sign_count,
                text_upper, t, glow_phase, transition
            )
            cv2.imshow(window, frame)

            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):
                paused = not paused

        i += 1

    # End screen
    cv2.imshow(window, render_done_frame(text_upper))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ════════════════════════════════════════════════════════════════════════════
# 5. Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ISL Text → Animated Sign Graphics")
    parser.add_argument("--data_dir", required=True,
                        help="Path to Indian/ dataset folder")
    parser.add_argument("--text", default=None,
                        help="Text to show (optional)")
    parser.add_argument("--speed", type=float, default=1.5,
                        help="Seconds per sign (default: 1.5)")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"❌  Folder not found: {args.data_dir}")
        sys.exit(1)

    sign_index = build_sign_landmark_index(args.data_dir)

    if args.text:
        show_text_as_signs(args.text, sign_index, args.speed)
    else:
        print("─" * 50)
        print("  ISL Text → Sign Graphics  |  Interactive")
        print("  Type text and press Enter. Type 'quit' to exit.")
        print("─" * 50 + "\n")
        while True:
            try:
                text = input("✏️  Enter text: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n👋  Bye!"); break
            if text.lower() in ("quit", "exit", "q"):
                print("👋  Bye!"); break
            if not text:
                continue
            show_text_as_signs(text, sign_index, args.speed)


if __name__ == "__main__":
    main()