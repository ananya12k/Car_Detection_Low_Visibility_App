import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO

FOCAL_LENGTH = 800
REAL_CAR_HEIGHT = 1.5


# Dehazing and enhancement functions
def dark_channel(image, size=15):
    dark_channel_img = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel_img = cv2.erode(dark_channel_img, kernel)
    return dark_channel_img


def estimate_atmospheric_light(image, dark_channel_img, top_percent=0.001):
    num_pixels = image.shape[0] * image.shape[1]
    num_brightest = int(max(num_pixels * top_percent, 1))
    dark_flat = dark_channel_img.ravel()
    indices = np.argsort(dark_flat)[-num_brightest:]
    brightest_pixels = np.array([image.reshape(-1, 3)[i] for i in indices])
    atmospheric_light = np.percentile(brightest_pixels, 90, axis=0)
    return atmospheric_light


def estimate_transmission(image, atmospheric_light, omega=0.95, size=15):
    norm_image = image / atmospheric_light
    transmission = 1 - omega * dark_channel(norm_image, size)
    transmission = np.clip(transmission, 0.1, 1)
    return transmission


def recover_radiance(image, transmission, atmospheric_light, t0=0.1):
    transmission = np.expand_dims(transmission, axis=2)
    radiance = (image - atmospheric_light) / np.maximum(transmission, t0) + atmospheric_light
    return np.clip(radiance, 0, 1)


def apply_clahe(image, clip_limit=3.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return enhanced_img


# YOLO model for object detection
model = YOLO("yolo-Weights/yolov8n.pt")


# Process a single frame
def process_frame(frame):
    img = frame.astype(np.float32) / 255.0
    dark_channel_img = dark_channel(img)
    atmospheric_light = estimate_atmospheric_light(img, dark_channel_img)
    transmission = estimate_transmission(img, atmospheric_light)
    dehazed_img = recover_radiance(img, transmission, atmospheric_light)
    dehazed_img = (dehazed_img * 255).astype(np.uint8)
    dehazed_img = apply_clahe(dehazed_img)

    # YOLOv8 detection
    results = model(dehazed_img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            pixel_height = y2 - y1
            # Estimate distance
            distance = (FOCAL_LENGTH * REAL_CAR_HEIGHT) / pixel_height

            # Draw detection boxes and info
            cv2.rectangle(dehazed_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(dehazed_img, f"Dist: {distance:.2f}m", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

    return dehazed_img


# Video processing function
def process_video(video):
    cap = cv2.VideoCapture(video)
    processed_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)
        processed_frames.append(processed_frame)

    cap.release()

    # Combine processed frames into a video
    output_video = "processed_video.mp4"
    height, width, _ = processed_frames[0].shape
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for frame in processed_frames:
        out.write(frame)
    out.release()

    return output_video


# Define Gradio UI
def gradio_app(video):
    # Process video and return processed video
    processed_video = process_video(video)
    return processed_video


# Interface
iface = gr.Interface(
    fn=gradio_app,
    inputs=gr.Video(label="Upload a foggy driving video"),
    outputs=gr.Video(label="Processed Video with Fog Removal and Object Detection"),
    title="Foggy Driving Video Processing",
    description="This app processes foggy driving videos to remove fog and detect vehicles, estimating their distance.",
)

# Launch the Gradio app
iface.launch(share=True)
