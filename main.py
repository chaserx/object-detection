import cv2
from ultralytics import YOLO
from coco_classes import class_names

# Constants
CONFIDENCE_THRESHOLD = 0.5
WINDOW_NAME = "Object Detection"
# MODEL_PATH = "yolo-Weights/yolov8n.pt"
MODEL_PATH = "yolo-Weights/yolo11n.pt"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


def get_rgb_from_confidence(confidence):
    """Get the RGB color from the confidence threshold."""
    # confidence is an integer between 0 and 100
    confidence = int(confidence * 100)
    colors = {
        "color1": (92, 63, 0),  # #003f5c
        "color2": (141, 80, 88),  # #58508d
        "color3": (144, 80, 188),  # #bc5090
        "color4": (97, 99, 255),  # #ff6361
        "color5": (0, 166, 255),  # #ffa600
    }

    # segment confidence into 5 equal parts
    segments = 5
    segment_size = 100 / segments
    for i in range(segments):
        if confidence < segment_size * (i + 1):
            return colors[f"color{i + 1}"]


def initialize_camera():
    """Initialize and configure the camera."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")
    return cap


def draw_detection(img, box, class_name, confidence):
    """Draw bounding box and label for a detected object."""
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Draw bounding box
    # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
    cv2.rectangle(img, (x1, y1), (x2, y2), get_rgb_from_confidence(confidence), 3)

    # Draw label
    label = f"{class_name} {confidence:.2f}"
    cv2.putText(
        img,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        get_rgb_from_confidence(confidence),
        2,
    )


def main():
    try:
        # Initialize model and camera
        model = YOLO(MODEL_PATH)
        cap = initialize_camera()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Run detection
            results = model(frame, stream=True)

            # Process detections
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    confidence = float(box.conf[0])
                    if confidence < CONFIDENCE_THRESHOLD:
                        continue

                    class_id = int(box.cls[0])
                    class_name = class_names[class_id]
                    draw_detection(frame, box, class_name, confidence)

            # Display the frame
            cv2.imshow(WINDOW_NAME, frame)

            # Break loop on 'ESC' key
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
