# modules/sign_detector.py  — MediaPipe hand detection → letter recognition

import time

try:
    import cv2
    import mediapipe as mp
    import numpy as np
    READY = True
except ImportError:
    READY = False

# ── Simple landmark-based classifier for ASL fingerspelling ─────────────────
# Each entry is (name, detector_fn).
# This is a lightweight heuristic version — replace with a trained model
# for production accuracy.

def _fingers_up(hand_landmarks):
    """Return list [thumb, index, middle, ring, pinky] → 1=up, 0=down."""
    lm = hand_landmarks.landmark
    tips   = [4, 8, 12, 16, 20]
    bases  = [2, 5, 9, 13, 17]
    result = []
    # Thumb: compare x instead of y
    result.append(1 if lm[tips[0]].x < lm[bases[0]].x else 0)
    for i in range(1, 5):
        result.append(1 if lm[tips[i]].y < lm[bases[i]].y else 0)
    return result


def _classify_gesture(hand_landmarks) -> str:
    """Heuristic ASL letter classification."""
    f = _fingers_up(hand_landmarks)
    lm = hand_landmarks.landmark

    # Basic patterns
    if f == [0, 1, 0, 0, 0]: return "D"
    if f == [0, 1, 1, 0, 0]: return "V"   # also K
    if f == [0, 1, 1, 1, 0]: return "W"
    if f == [0, 1, 1, 1, 1]: return "B"
    if f == [1, 1, 1, 1, 1]: return "B"
    if f == [1, 0, 0, 0, 1]: return "Y"
    if f == [1, 1, 0, 0, 1]: return "L"   # approx
    if f == [0, 0, 0, 0, 0]: return "S"   # fist
    if f == [1, 0, 0, 0, 0]: return "A"   # thumb out

    # Check distance index–thumb for O / C
    d = abs(lm[8].x - lm[4].x) + abs(lm[8].y - lm[4].y)
    if d < 0.06 and f[1] == 0 and f[2] == 0:
        return "O"
    if d < 0.12 and f[1] == 1 and f[2] == 0:
        return "D"

    return "?"


def detect_sign_from_camera(duration_sec: int = 10) -> str:
    """
    Open camera, run MediaPipe hand detection for up to duration_sec seconds.
    Return the most-detected letter, or empty string if none.
    """
    if not READY:
        return ""

    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    counts: dict = {}

    cap = cv2.VideoCapture(0)
    deadline = time.time() + duration_sec

    with mp_hands.Hands(min_detection_confidence=0.7,
                        min_tracking_confidence=0.5) as hands:
        while time.time() < deadline:
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hl, mp_hands.HAND_CONNECTIONS)
                    letter = _classify_gesture(hl)
                    if letter != "?":
                        counts[letter] = counts.get(letter, 0) + 1

            # Overlay
            remaining = max(0, int(deadline - time.time()))
            cv2.putText(frame, f"Sign detection: {remaining}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 120), 2)
            top = max(counts, key=counts.get) if counts else "—"
            cv2.putText(frame, f"Best: {top}",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 200, 255), 3)
            cv2.imshow("AccessLearn — Sign Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return max(counts, key=counts.get) if counts else ""
