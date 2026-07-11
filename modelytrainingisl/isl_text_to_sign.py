"""
Indian Sign Language (ISL) - Text to Sign Display
slideshow using images from dataset.
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


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DISPLAY_W, DISPLAY_H = 600, 600
PANEL_H = 120         
CANVAS_H = DISPLAY_H + PANEL_H

BG_COLOR      = (15,  15,  15)
PANEL_COLOR   = (30,  30,  30)
ACCENT_COLOR  = (0,  200, 120)
TEXT_COLOR    = (240, 240, 240)
DIM_COLOR     = (120, 120, 120)
BORDER_COLOR  = (60,  60,  60)


def build_sign_index(data_dir: str) -> dict:
    """
    Returns dict mapping uppercase char → list of image paths.
    e.g. {'A': ['Indian/A/1.jpg', ...], '0': ['Indian/0/1.jpg', ...], ...}
    """
    index = {}
    if not os.path.isdir(data_dir):
        print(f"❌  Dataset folder not found: {data_dir}")
        sys.exit(1)

    for folder in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        key = folder.upper()
        images = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
        ]
        if images:
            index[key] = images

    print(f"✅  Loaded {len(index)} signs: {sorted(index.keys())}")
    return index


def make_canvas():
    canvas = np.full((CANVAS_H, DISPLAY_W, 3), BG_COLOR, dtype=np.uint8)
    canvas[DISPLAY_H:, :] = PANEL_COLOR
    return canvas


def put_text_centered(img, text, y, font_scale, color, thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (img.shape[1] - tw) // 2
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def render_sign_frame(sign_img_path, char, position, total, full_text, speed):
    """Build a display frame for one sign."""
    canvas = make_canvas()

    img = cv2.imread(sign_img_path)
    if img is not None:
        img = cv2.resize(img, (DISPLAY_W - 40, DISPLAY_H - 40))
        canvas[20:20 + img.shape[0], 20:20 + img.shape[1]] = img
    else:
        put_text_centered(canvas, "?", DISPLAY_H // 2, 4, DIM_COLOR, 6)

    cv2.rectangle(canvas, (10, 10), (DISPLAY_W - 10, DISPLAY_H - 10), BORDER_COLOR, 2)

    cv2.rectangle(canvas, (10, 10), (80, 80), ACCENT_COLOR, -1)
    cv2.putText(canvas, char, (18, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4, cv2.LINE_AA)


    prog_text = f"{position}/{total}"
    cv2.putText(canvas, prog_text, (DISPLAY_W - 90, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, DIM_COLOR, 2, cv2.LINE_AA)

   
    panel_y = DISPLAY_H + 10

   
    display_text = ""
    for i, c in enumerate(full_text):
        display_text += c  

  
    char_w = 22
    start_x = max(10, (DISPLAY_W - len(full_text) * char_w) // 2)
    for i, c in enumerate(full_text):
        color = ACCENT_COLOR if i == (position - 1) else DIM_COLOR
        x = start_x + i * char_w
        if x > DISPLAY_W - 20:
            break
        cv2.putText(canvas, c, (x, panel_y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

  
    hint = f"speed: {speed}s/sign  |  Q to skip"
    put_text_centered(canvas, hint, CANVAS_H - 15, 0.45, DIM_COLOR, 1)

    return canvas


def render_space_frame(position, total, full_text, speed):
    """Show a 'SPACE' pause frame."""
    canvas = make_canvas()
    put_text_centered(canvas, "[ SPACE ]", DISPLAY_H // 2 - 30, 1.5, DIM_COLOR, 2)
    put_text_centered(canvas, "word break", DISPLAY_H // 2 + 30, 0.7, DIM_COLOR, 1)
    cv2.rectangle(canvas, (10, 10), (DISPLAY_W - 10, DISPLAY_H - 10), BORDER_COLOR, 2)

    panel_y = DISPLAY_H + 10
    char_w = 22
    start_x = max(10, (DISPLAY_W - len(full_text) * char_w) // 2)
    for i, c in enumerate(full_text):
        color = ACCENT_COLOR if i == (position - 1) else DIM_COLOR
        x = start_x + i * char_w
        if x > DISPLAY_W - 20:
            break
        cv2.putText(canvas, c, (x, panel_y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    hint = f"speed: {speed}s/sign  |  Q to skip"
    put_text_centered(canvas, hint, CANVAS_H - 15, 0.45, DIM_COLOR, 1)
    return canvas


def render_unknown_frame(char, position, total, full_text, speed):
    canvas = make_canvas()
    put_text_centered(canvas, f"'{char}'", DISPLAY_H // 2 - 20, 3.0, (80, 80, 200), 4)
    put_text_centered(canvas, "no sign image found", DISPLAY_H // 2 + 50, 0.7, DIM_COLOR, 1)
    cv2.rectangle(canvas, (10, 10), (DISPLAY_W - 10, DISPLAY_H - 10), BORDER_COLOR, 2)
    hint = f"speed: {speed}s/sign  |  Q to skip"
    put_text_centered(canvas, hint, CANVAS_H - 15, 0.45, DIM_COLOR, 1)
    return canvas


def show_text_as_signs(text: str, sign_index: dict, speed: float):
    """Display each character in text as its ISL sign image."""
    text_upper = text.upper()
    chars = list(text_upper)

    if not chars:
        return

    window = "ISL Text to Sign  —  press Q to quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, DISPLAY_W, CANVAS_H)

    sign_count = sum(1 for c in chars if c != ' ')
    sign_pos = 0

    for i, char in enumerate(chars):
        if char == ' ':
            frame = render_space_frame(i + 1, len(chars), text_upper, speed)
            cv2.imshow(window, frame)
            delay_ms = int(speed * 500)
            if cv2.waitKey(delay_ms) & 0xFF in (ord('q'), ord('Q'), 27):
                break
            continue

        sign_pos += 1

        if char not in sign_index:
            frame = render_unknown_frame(char, sign_pos, sign_count, text_upper, speed)
            cv2.imshow(window, frame)
            if cv2.waitKey(int(speed * 1000)) & 0xFF in (ord('q'), ord('Q'), 27):
                break
            continue

        img_path = random.choice(sign_index[char])
        frame = render_sign_frame(img_path, char, sign_pos, sign_count, text_upper, speed)
        cv2.imshow(window, frame)

        delay_ms = int(speed * 1000)
        start = time.time()
        while (time.time() - start) < speed:
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                cv2.destroyAllWindows()
                return
          
            cv2.imshow(window, frame)

   
    end_canvas = make_canvas()
    put_text_centered(end_canvas, "Done!", DISPLAY_H // 2 - 20, 2.0, ACCENT_COLOR, 3)
    put_text_centered(end_canvas, f'"{text_upper}"', DISPLAY_H // 2 + 40, 0.8, TEXT_COLOR, 2)
    put_text_centered(end_canvas, "Press any key to continue", CANVAS_H - 20, 0.5, DIM_COLOR, 1)
    cv2.imshow(window, end_canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="ISL Text → Sign Slideshow")
    parser.add_argument("--data_dir", required=True,
                        help="Path to Indian/ dataset folder (e.g. .../SIGN LANGUAGE/Indian)")
    parser.add_argument("--text", default=None,
                        help="Text to show (optional — if omitted, enters interactive mode)")
    parser.add_argument("--speed", type=float, default=1.2,
                        help="Seconds per sign (default: 1.2)")
    args = parser.parse_args()

    sign_index = build_sign_index(args.data_dir)

    if args.text:
        print(f"\n🖐️  Showing: '{args.text}'")
        show_text_as_signs(args.text, sign_index, args.speed)
    else:
        
        print("\n" + "─" * 50)
        print("  ISL Text → Sign  |  Interactive Mode")
        print("  Type any text and press Enter to see signs")
        print("  Type 'quit' or press Ctrl+C to exit")
        print("─" * 50 + "\n")

        while True:
            try:
                text = input("✏️  Enter text: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n👋  Bye!")
                break

            if text.lower() in ("quit", "exit", "q"):
                print("👋  Bye!")
                break

            if not text:
                continue

            print(f"🖐️  Showing signs for: '{text.upper()}'")
            show_text_as_signs(text, sign_index, args.speed)
            print()


if __name__ == "__main__":
    main()
