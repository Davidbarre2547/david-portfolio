import os


import cv2
import cv2.data
import numpy as np


# Variables
RED = "\033[31;1m"
BLUE = "\033[34;1m"
GREEN = "\033[32;1m"
PURPLE = "\033[35;1m"
RESET = "\033[0;0m"


def apply_overlay(frame, overlay, x, y):
    """
    Blend a BGRA overlay onto the frame at position (x, y) using per-pixel alpha.
    """


    h, w = overlay.shape[:2]


    # ---Boundary Checks ---
    # Crop overlay if it goes outside frame bounds
    if x < 0:
        overlay = overlay[:, -x:]
        w = overlay.shape[1]
        x = 0
    if y < 0:
        overlay = overlay[-y:, :]
        h = overlay.shape[0]
        y = 0
    if x + w > frame.shape[1]:
        overlay = overlay[:, :frame.shape[1] - x]
        w = overlay.shape[1]
    if y + h > frame.shape[0]:
        overlay = overlay[:frame.shape[0] - y, :]
        h = overlay.shape[0]


    # Ensure overlay has alpha channel
    if overlay.shape[2] < 4:
        bgr = overlay
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        alpha = np.where(gray > 240, 0, 255).astype(np.uint8)
        overlay = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        overlay[:, :, 3] = alpha


    # Split channels
    b, g, r, a = cv2.split(overlay)
    # Optional: Blur alpha channel for smooth edges
    a = cv2.GaussianBlur(a, (5, 5), 0)
    alpha = a.astype(float)/ 255.0
    alpha = np.dstack([alpha, alpha, alpha]) # shape (h, w, 3)


    # Region of interest on the frame
    roi = frame[y:y+h, x:x+w].astype(float)
    overlay_rgb = cv2.merge([b, g, r]).astype(float)


    # Blend: out = α . overlay + (1 - α) . background
    blended = alpha * overlay_rgb + (1 - alpha) * roi
    frame[y:y+h, x:x+w] = blended.astype(np.uint8)


    return frame


def main():
    # loading face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


    # Load overlay images
    overlay_path = "sunglasses.png"


    if not os.path.isfile(overlay_path):
        print(f"{RED}Error: Overlay image not found at {overlay_path}")
        print(f"{PURPLE}Please provide a valid path to an overlay image with transparency.")
        return

    # Load the overlay with alpha channel (RGBA)
    overlay_orig = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)


    if overlay_orig is None:
        print(f"{RED}Error: Failed to load overlay image.")
        return

    # Start video capture
    cap = cv2.VideoCapture(0)


    if not cap.isOpened():
        print(f"{RED}Error: Could not open video capture.")
        return

    print(F"{PURPLE}Face filter started. Press 'q' to quit.")


    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"{RED}Error: Failed to capture frame.")
            break


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )


        for (x, y, w, h) in faces:
            # Scale overlay relative to face width
            overlay_w = int(1.1 * w)
            overlay_h = int(overlay_w * overlay_orig.shape[0] / overlay_orig.shape[1])


            # Resize Once
            overlay = cv2.resize(overlay_orig, (overlay_w, overlay_h), interpolation=cv2.INTER_AREA)


            # PositionL center horizontally, slightly below top of face
            x_offset = x - int(0.05  * w)
            y_offset = y +  int(0.04 * h)


            # Apply overlay
            frame = apply_overlay(frame, overlay, x_offset, y_offset)


        cv2.imshow('Face Filter', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
