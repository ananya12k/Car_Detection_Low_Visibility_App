import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from ultralytics import YOLO

# Constants for distance estimation
FOCAL_LENGTH = 800  # Example value, should be calibrated for the camera
REAL_CAR_HEIGHT = 1.5  # Approximate average height of a car in meters

# Dehazing and enhancement functions (same as before)
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

# Main Window Class
class VideoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Foggy Video Processing")
        self.setGeometry(100, 100, 800, 600)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.initUI()
        self.model = YOLO("yolo-Weights/yolov8n.pt")
        self.cap = None
        self.output_frame = None
        self.trackers = []  # List to store object trackers
        self.car_ids = 0
        self.object_info = {}  # Dictionary to store distances and speeds

    def initUI(self):
        layout = QVBoxLayout()

        self.uploadButton = QPushButton("Upload Video")
        self.uploadButton.clicked.connect(self.upload_video)
        layout.addWidget(self.uploadButton)

        self.originalLabel = QLabel("Original Video")
        self.processedLabel = QLabel("Processed Video")
        layout.addWidget(self.originalLabel)
        layout.addWidget(self.processedLabel)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def upload_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi)")
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            self.timer.start(30)

    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            self.timer.stop()
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            return

        # Process the frame for dehazing and object detection
        original_frame = frame.copy()
        processed_frame = self.process_frame(frame)

        # Convert frames to QImage and show
        self.display_frame(self.originalLabel, original_frame)
        self.display_frame(self.processedLabel, processed_frame)

    def process_frame(self, frame):
        # Apply dehazing
        img = frame.astype(np.float32) / 255.0
        dark_channel_img = dark_channel(img)
        atmospheric_light = estimate_atmospheric_light(img, dark_channel_img)
        transmission = estimate_transmission(img, atmospheric_light)
        dehazed_img = recover_radiance(img, transmission, atmospheric_light)
        dehazed_img = (dehazed_img * 255).astype(np.uint8)
        dehazed_img = apply_clahe(dehazed_img)

        # YOLOv8 detection
        results = self.model(dehazed_img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pixel_height = y2 - y1
                # Estimate distance
                distance = (FOCAL_LENGTH * REAL_CAR_HEIGHT) / pixel_height

                # Create a new tracker for each detected object
                tracker = cv2.TrackerCSRT_create()
                bbox = (x1, y1, x2 - x1, y2 - y1)
                tracker.init(dehazed_img, bbox)
                self.trackers.append(tracker)

                # Assign ID to the car and store initial distance and time
                car_id = self.car_ids
                self.car_ids += 1
                self.object_info[car_id] = {"distance": distance, "prev_distance": distance, "speed": 0}

                # Draw bounding boxes and distance info
                cv2.rectangle(dehazed_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(dehazed_img, f"Dist: {distance:.2f}m", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update trackers and estimate speed
        self.update_trackers(dehazed_img)
        return dehazed_img

    def update_trackers(self, frame):
        for i, tracker in enumerate(self.trackers):
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = map(int, bbox)
                pixel_height = h
                distance = (FOCAL_LENGTH * REAL_CAR_HEIGHT) / pixel_height

                # Calculate speed
                car_id = i
                prev_distance = self.object_info[car_id]["prev_distance"]
                speed = abs(distance - prev_distance) / (1 / 30.0)  # Assuming 30 FPS

                # Update object info
                self.object_info[car_id]["prev_distance"] = distance
                self.object_info[car_id]["speed"] = speed

                # Draw bounding boxes, speed, and distance info
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Speed: {speed:.2f}m/s", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    def display_frame(self, label, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qt_image).scaled(label.width(), label.height()))

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoApp()
    window.show()
    sys.exit(app.exec_())
