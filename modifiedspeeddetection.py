import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from ultralytics import YOLO

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

def apply_clahe(image, clip_limit=1.0, tile_grid_size=(4, 4)):
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
        self.model = YOLO("yolo-Weights/yolov8n.pt")
        self.cap = None
        self.previous_centroids = {}

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

    def process_frame(self, frame):
        img = frame.astype(np.float32) / 255.0
        dark_channel_img = dark_channel(img)
        atmospheric_light = estimate_atmospheric_light(img, dark_channel_img)
        transmission = estimate_transmission(img, atmospheric_light)
        dehazed_img = recover_radiance(img, transmission, atmospheric_light)
        dehazed_img = (dehazed_img * 255).astype(np.uint8)
        dehazed_img = apply_clahe(dehazed_img)

        results = self.model(dehazed_img, stream=True)
        current_centroids = {}
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pixel_height = y2 - y1
                distance = (FOCAL_LENGTH * REAL_CAR_HEIGHT) / pixel_height
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                current_centroids[centroid] = distance
                speed = self.estimate_speed(centroid, distance)

                cv2.rectangle(dehazed_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(dehazed_img, f"Dist: {distance:.2f}m", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)
                cv2.putText(dehazed_img, f"Speed: {speed:.2f}m/s", (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

        self.previous_centroids = current_centroids
        return dehazed_img

    def estimate_speed(self, centroid, distance):
        if centroid in self.previous_centroids:
            prev_distance = self.previous_centroids[centroid]
            displacement = abs(distance - prev_distance)
            time_interval = 1 / 30.0
            speed = displacement / time_interval
            return speed
        return 0

    def display_frame(self, label, frame):
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Create QImage from the RGB image
        qimage = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0],
                        rgb_image.strides[0], QImage.Format_RGB888)
        # Update the label with the QPixmap
        label.setPixmap(QPixmap.fromImage(qimage))

    def closeEvent(self, event):
        self.stop_processing()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoApp()
    window.show()
    sys.exit(app.exec_())
