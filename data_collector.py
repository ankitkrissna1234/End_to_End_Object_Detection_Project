import os
import cv2
import time
import uuid

IMAGE_PATH = "CollectedImages"
labels = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]
number_of_images = 20

for label in labels:
    img_path = os.path.join(IMAGE_PATH, label)
    os.makedirs(img_path, exist_ok=True)

    # Count existing images
    existing_images = len([f for f in os.listdir(img_path) if f.endswith(".jpg")])
    img_count = existing_images

    # Check if collection is already complete for this label
    if img_count >= number_of_images:
        print(f"Image collection for {label} is already complete.")
        continue

    # Open camera
    cap = cv2.VideoCapture(0)
    print(f"Collecting images for {label}. {img_count} images already exist.")
    time.sleep(3)

    while img_count < number_of_images:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("frame", frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):  # Press 's' to save the image
            imagename = os.path.join(
                IMAGE_PATH, label, label + "." + "{}.jpg".format(str(uuid.uuid1()))
            )
            cv2.imwrite(imagename, frame)
            print(f"Saved image {img_count + 1} for {label}")
            img_count += 1
            time.sleep(2)

        elif key == ord("q"):  # Press 'q' to quit early
            break

    cap.release()
    cv2.destroyAllWindows()

print("Image collection complete.")
