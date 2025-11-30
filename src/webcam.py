import cv2 as cv
import mediapipe as mp
import os
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dotenv import load_dotenv

load_dotenv()


def use_webcam():
    # Captures video from the default webcam (index 0)
    video_capture = cv.VideoCapture(0)

    # keeps a constant loop to read frames from the webcam
    while True:
        ret, frame = video_capture.read()

        # flips the frame horizontally for a mirror-like effect
        flipped_frame = cv.flip(frame, 1)

        if ret:
            detect_objects(flipped_frame)
        # waits for the 'q' key to be pressed to exit the loop
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()  # Releases the camera hardware so other apps can use your webcam afterward
    cv.destroyAllWindows()  # Closes every OpenCV window


def visualize(image_copy, detection_result):
    """
    Detection result sample data

    DetectionResult(
        detections=[
            Detection(
                bounding_box=BoundingBox(origin_x=260, origin_y=65, width=1399, height=1028),
                categories=[
                    Category(index=None, score=0.91796875, display_name=None, category_name='person')
                ],
                keypoints=[]
            )
        ]
    )
    """

    for detection in detection_result.detections:
        bbox = detection.bounding_box
        x = bbox.origin_x  # left edge of the bounding box
        y = bbox.origin_y  # top edge of the bounding box
        w = bbox.width  # width of the bounding box / how far right it travels
        h = bbox.height  # height of the bounding box / how far down it travels
        x2 = x + w  # how far right it travels from the left edge
        y2 = y + h  # how far down it travels from the top edge
        green = (0, 255, 0)
        border_thickness = 2
        text_thickness = 2

        object_name = detection.categories[0].category_name
        object_score = round(detection.categories[0].score * 100, 2)
        label = f"{object_name}: {object_score}%"

        cv.rectangle(image_copy, (x, y), (x2, y2), green, border_thickness)
        cv.putText(
            image_copy,
            label,
            (x, y - 10),  # Position: above the box (10 pixels up from top edge)
            cv.FONT_HERSHEY_SIMPLEX,
            1,  # Font scale (size)
            green,
            text_thickness,
        )

    return image_copy


def detect_objects(frame):
    MODEL_PATH = os.getenv("MODEL_PATH")

    if MODEL_PATH is None:
        raise ValueError("MODEL_PATH environment variable is not set.")

    # Create an ObjectDetector object.
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.ObjectDetectorOptions(
        base_options=base_options, score_threshold=0.5
    )
    detector = vision.ObjectDetector.create_from_options(options)

    # Load the input image.
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Detect objects in the input image.
    detection_result = detector.detect(image)

    # # Visualize the detection result.
    image_copy = np.copy(image.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    cv.imshow("Webcam", annotated_image)


def main():
    use_webcam()
