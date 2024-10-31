import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, \
    QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from ultralytics import YOLO
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

FOCAL_LENGTH = 800
REAL_CAR_HEIGHT = 1.5


# Dehazing and enhancement functions
def dark_channel(image, size=7):
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


def estimate_transmission(image, atmospheric_light, omega=0.95, size=7):
    norm_image = image / atmospheric_light
    transmission = 1 - omega * dark_channel(norm_image, size)
    return np.clip(transmission, 0.1, 1)


def recover_radiance(image, transmission, atmospheric_light, t0=0.1):
    transmission = np.expand_dims(transmission, axis=2)
    radiance = (image - atmospheric_light) / np.maximum(transmission, t0) + atmospheric_light
    return np.clip(radiance, 0, 1)


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


class VideoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Foggy Video Processing")
        self.setGeometry(100, 100, 900, 700)
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QPushButton { background-color: #3a3a3a; color: #ffffff; font-size: 16px; padding: 10px; border: 1px solid #555555; border-radius: 5px; }
            QPushButton:hover { background-color: #505050; }
            QPushButton#exitButton { background-color: #ff0000; }
            QLabel { color: #ffffff; font-size: 16px; margin-top: 5px; margin-bottom: 5px; border: 1px solid #555555; padding: 5px; border-radius: 5px; background-color: #3a3a3a; }
        """)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.initUI()

        # Initialize YOLO
        self.model = YOLO("yolov8n.pt")

        # Initialize DeepSORT - no weights needed for deep_sort_realtime
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True
        )

        self.cap = None
        self.track_history = {}
        self.last_process_time = {}

    def initUI(self):
        layout = QVBoxLayout()
        button_layout = QHBoxLayout()

        self.uploadButton = QPushButton("Upload Video")
        self.uploadButton.setStyleSheet("background-color: #ff5733; color: white; font-size: 16px;")
        self.uploadButton.clicked.connect(self.upload_video)
        button_layout.addWidget(self.uploadButton)

        self.stopButton = QPushButton("Stop Processing")
        self.stopButton.setEnabled(False)
        self.stopButton.setStyleSheet("background-color: #008CBA; color: white; font-size: 16px;")
        self.stopButton.clicked.connect(self.stop_processing)
        button_layout.addWidget(self.stopButton)

        layout.addLayout(button_layout)

        video_layout = QVBoxLayout()

        self.originalTitle = QLabel("Original Video")
        self.originalTitle.setAlignment(Qt.AlignCenter)
        self.originalLabel = QLabel()
        self.originalLabel.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.originalTitle)
        video_layout.addWidget(self.originalLabel)

        self.processedTitle = QLabel("Processed Video")
        self.processedTitle.setAlignment(Qt.AlignCenter)
        self.processedLabel = QLabel()
        self.processedLabel.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.processedTitle)
        video_layout.addWidget(self.processedLabel)

        layout.addLayout(video_layout)

        self.exitButton = QPushButton("Exit")
        self.exitButton.setObjectName("exitButton")
        self.exitButton.clicked.connect(self.close)
        layout.addWidget(self.exitButton)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.setFixedSize(900, 800)

    def upload_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi)")
        if file_path:
            self.uploadButton.setText("Processing")
            self.uploadButton.setEnabled(False)
            self.stopButton.setEnabled(True)
            self.cap = cv2.VideoCapture(file_path)
            self.timer.start(30)

    def stop_processing(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        self.uploadButton.setText("Upload Video")
        self.uploadButton.setEnabled(True)
        self.stopButton.setEnabled(False)

    def calculate_speed(self, track_id, bbox, distance, current_time):
        if track_id in self.track_history and track_id in self.last_process_time:
            prev_bbox, prev_distance = self.track_history[track_id]
            time_diff = current_time - self.last_process_time[track_id]

            if time_diff > 0:
                # Calculate center points
                current_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                prev_center = ((prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2)

                # Calculate displacement in pixels and real-world distance
                pixel_displacement = np.sqrt(
                    (current_center[0] - prev_center[0]) ** 2 +
                    (current_center[1] - prev_center[1]) ** 2
                )

                depth_displacement = abs(distance - prev_distance)

                # Convert pixel displacement to meters using average distance
                avg_distance = (distance + prev_distance) / 2
                real_displacement = (pixel_displacement * avg_distance) / FOCAL_LENGTH

                # Calculate 3D displacement
                total_displacement = np.sqrt(real_displacement ** 2 + depth_displacement ** 2)

                # Calculate speed in km/h
                speed = (total_displacement / time_diff) * 3.6  # Convert m/s to km/h
                return min(speed, 150)  # Cap speed at 150 km/h to filter outliers

        return 0

    def process_frame(self, frame):
        current_time = cv2.getTickCount() / cv2.getTickFrequency()

        # Dehaze and enhance
        img = frame.astype(np.float32) / 255.0
        dark_channel_img = dark_channel(img)
        atmospheric_light = estimate_atmospheric_light(img, dark_channel_img)
        transmission = estimate_transmission(img, atmospheric_light)
        dehazed_img = recover_radiance(img, transmission, atmospheric_light)
        dehazed_img = (dehazed_img * 255).astype(np.uint8)
        enhanced_img = apply_clahe(dehazed_img)

        # Run YOLO detection
        results = self.model(enhanced_img, stream=True)
        detections = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if cls == 2:  # Class 2 is car in COCO dataset
                    detections.append(((x1, y1, x2, y2), conf, cls))

        # Update DeepSORT tracker
        tracks = self.tracker.update_tracks(detections, frame=enhanced_img)

        # Process and draw each track
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr()  # Gets (top, left, bottom, right)
            x1, y1, x2, y2 = map(int, bbox)

            # Calculate distance using height
            height = y2 - y1
            distance = (FOCAL_LENGTH * REAL_CAR_HEIGHT) / height

            # Calculate speed
            speed = self.calculate_speed(track_id, bbox, distance, current_time)

            # Update tracking history
            self.track_history[track_id] = (bbox, distance)
            self.last_process_time[track_id] = current_time

            # Draw bounding box and information
            cv2.rectangle(enhanced_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(enhanced_img, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(enhanced_img, f"Dist: {distance:.2f}m", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(enhanced_img, f"Speed: {speed:.1f}km/h", (x1, y2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return enhanced_img

    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            self.timer.stop()
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_processing()
            return

        original_frame = frame.copy()
        processed_frame = self.process_frame(frame)
        self.display_frame(self.originalLabel, original_frame)
        self.display_frame(self.processedLabel, processed_frame)

    def display_frame(self, label, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qimage = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimage))

    def closeEvent(self, event):
        self.stop_processing()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoApp()
    window.show()
    sys.exit(app.exec_())