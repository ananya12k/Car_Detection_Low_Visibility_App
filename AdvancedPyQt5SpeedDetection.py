import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, \
    QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from ultralytics import YOLO
from PyQt5.QtCore import Qt
import filterpy.kalman as kalman

# Constants
FOCAL_LENGTH = 800
REAL_CAR_HEIGHT = 1.5
FRAME_SAMPLING_INTERVAL = 2  # Process every 3rd frame to reduce computational load


# Dehazing and enhancement functions
def dark_channel(image, size=15):
    """Improved dark channel computation with more robust handling."""
    # Convert to float and handle potential edge cases
    img_float = image.astype(np.float32)

    # Compute dark channel with improved robustness
    dark_channel_img = np.min(img_float, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel_img = cv2.erode(dark_channel_img, kernel)
    return dark_channel_img


def estimate_atmospheric_light(image, dark_channel_img, top_percent=0.001):
    """More robust atmospheric light estimation."""
    num_pixels = image.shape[0] * image.shape[1]
    num_brightest = int(max(num_pixels * top_percent, 1))

    # Flatten and sort
    dark_flat = dark_channel_img.ravel()
    indices = np.argsort(dark_flat)[-num_brightest:]

    # Safely extract brightest pixels
    try:
        brightest_pixels = np.array([image.reshape(-1, 3)[i] for i in indices])
        atmospheric_light = np.percentile(brightest_pixels, 90, axis=0)
    except Exception:
        # Fallback to mean if percentile fails
        atmospheric_light = np.mean(image, axis=(0, 1))

    return atmospheric_light


def estimate_transmission(image, atmospheric_light, omega=0.95, size=15):
    """Improved transmission map estimation."""
    # Prevent division by zero
    atmospheric_light = np.maximum(atmospheric_light, 1e-10)

    # Normalize image
    norm_image = image / atmospheric_light

    # Compute transmission
    transmission = 1 - omega * dark_channel(norm_image, size)
    transmission = np.clip(transmission, 0.1, 1)

    return transmission


def recover_radiance(image, transmission, atmospheric_light, t0=0.1):
    """More stable radiance recovery."""
    # Ensure dimensional consistency
    transmission = np.expand_dims(transmission, axis=2)

    # Improved radiance recovery with clipping
    radiance = (image - atmospheric_light) / np.maximum(transmission, t0) + atmospheric_light
    return np.clip(radiance, 0, 1)


def apply_advanced_enhancement(image, clip_limit=4.0, tile_grid_size=(8, 8)):
    """Advanced image enhancement using CLAHE with more aggressive parameters."""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Create CLAHE with improved parameters
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)

    # Merge and convert back
    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Optional: Add slight sharpening
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    enhanced_img = cv2.filter2D(enhanced_img, -1, kernel)

    return enhanced_img


class KalmanTracker:
    """Kalman Filter for more stable object tracking and speed estimation."""

    def __init__(self):
        # Initialize Kalman Filter
        self.kf = kalman.KalmanFilter(dim_x=4, dim_z=2)

        # State transition matrix
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

        # Measurement function
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])

        # Measurement noise
        self.kf.R = np.eye(2) * 0.01

        # Process noise
        self.kf.Q = np.eye(4) * 0.1

        # Initial state covariance
        self.kf.P *= 1000

        # Initial state
        self.kf.x = np.zeros(4)

    def predict(self, measurement):
        """Predict and update with new measurement."""
        self.kf.predict()
        self.kf.update(measurement)
        return self.kf.x[:2]


# Main Window Class
class VideoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Foggy Video Processing")
        self.setGeometry(100, 100, 900, 700)
        self.kalman_trackers = {}

        # Frame counter for sampling
        self.frame_counter = 0
        # Style settings
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QPushButton {
                background-color: #3a3a3a;
                color: #ffffff;
                font-size: 16px;
                padding: 10px;
                border: 1px solid #555555;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QPushButton#exitButton {
                background-color: #ff0000;
            }
            QLabel {
                color: #ffffff;
                font-size: 16px;
                margin-top: 5px;
                margin-bottom: 5px;
                border: 1px solid #555555;
                padding: 5px;
                border-radius: 5px;
                background-color: #3a3a3a;
            }
        """)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.initUI()
        self.model = YOLO("yolo-Weights/yolov8n.pt")
        self.cap = None
        self.output_frame = None
        self.previous_centroids = {}

    def initUI(self):
        layout = QVBoxLayout()

        # Create a horizontal layout for buttons
        button_layout = QHBoxLayout()

        # Upload Button
        self.uploadButton = QPushButton("Upload Video")
        self.uploadButton.setStyleSheet("background-color: #ff5733; color: white; font-size: 16px;")
        self.uploadButton.clicked.connect(self.upload_video)
        button_layout.addWidget(self.uploadButton)

        # Stop Processing Button
        self.stopButton = QPushButton("Stop Processing")
        self.stopButton.setEnabled(False)
        self.stopButton.setStyleSheet("background-color: #008CBA; color: white; font-size: 16px;")  # Different color
        self.stopButton.clicked.connect(self.stop_processing)
        button_layout.addWidget(self.stopButton)

        # Add the button layout to the main layout
        layout.addLayout(button_layout)

        # Create a vertical layout for video displays
        video_layout = QVBoxLayout()

        # Original Video Section
        self.originalTitle = QLabel("Original Video")
        self.originalTitle.setAlignment(Qt.AlignCenter)
        self.originalLabel = QLabel()
        self.originalLabel.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.originalTitle)
        video_layout.addWidget(self.originalLabel)

        # Processed Video Section
        self.processedTitle = QLabel("Processed Video")
        self.processedTitle.setAlignment(Qt.AlignCenter)
        self.processedLabel = QLabel()
        self.processedLabel.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.processedTitle)
        video_layout.addWidget(self.processedLabel)

        # Add the video layout to the main layout
        layout.addLayout(video_layout)

        # Exit Button
        self.exitButton = QPushButton("Exit")
        self.exitButton.setObjectName("exitButton")
        self.exitButton.clicked.connect(self.close)
        layout.addWidget(self.exitButton)

        # Set main layout
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Set fixed size for the main window
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
            self.timer.stop()
            self.cap.release()
            return

        # Process the frame for dehazing and object detection
        original_frame = frame.copy()
        processed_frame = self.process_frame(frame)

        # Convert frames to QImage and show
        self.display_frame(self.originalLabel, original_frame)
        self.display_frame(self.processedLabel, processed_frame)



    def estimate_speed(self, centroid, distance):
        if centroid in self.previous_centroids:
            prev_distance = self.previous_centroids[centroid]
            displacement = abs(distance - prev_distance)
            time_interval = 1 / 30.0
            speed = displacement / time_interval
            return speed
        return 0

    def process_frame(self, frame):
        # Frame sampling to reduce computational load
        self.frame_counter += 1
        if self.frame_counter % FRAME_SAMPLING_INTERVAL != 0:
            return frame  # Return original frame if not processing

        # Improved dehazing process
        img = frame.astype(np.float32) / 255.0
        dark_channel_img = dark_channel(img)
        atmospheric_light = estimate_atmospheric_light(img, dark_channel_img)
        transmission = estimate_transmission(img, atmospheric_light)
        dehazed_img = recover_radiance(img, transmission, atmospheric_light)
        dehazed_img = (dehazed_img * 255).astype(np.uint8)

        # Advanced enhancement
        dehazed_img = apply_advanced_enhancement(dehazed_img)

        # YOLOv8 detection
        results = self.model(dehazed_img, stream=True)
        current_centroids = {}

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pixel_height = y2 - y1
                distance = (FOCAL_LENGTH * REAL_CAR_HEIGHT) / pixel_height
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Use Kalman filter for smoothing
                if centroid not in self.kalman_trackers:
                    self.kalman_trackers[centroid] = KalmanTracker()

                smoothed_centroid = self.kalman_trackers[centroid].predict(np.array(centroid))
                current_centroids[tuple(smoothed_centroid)] = distance

                speed = self.estimate_speed(tuple(smoothed_centroid), distance)

                # Drawing annotations
                cv2.rectangle(dehazed_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(dehazed_img, f"Dist: {distance:.2f}m", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(dehazed_img, f"Speed: {speed:.2f}m/s", (x1, y2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        self.previous_centroids = current_centroids
        return dehazed_img

    def display_frame(self, label, frame):
        """Improved frame display with high-quality scaling and sharpening."""
        # Convert to RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # High-quality resizing using bicubic interpolation
        resized_frame = cv2.resize(rgb_image,
                                   (label.width(), label.height()),
                                   interpolation=cv2.INTER_CUBIC)

        # Optional: Add slight sharpening to combat pixelation
        sharpen_kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
        sharpened_frame = cv2.filter2D(resized_frame, -1, sharpen_kernel)

        # Convert to QImage
        h, w, ch = sharpened_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(sharpened_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Set pixmap with no additional scaling
        label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()


# Main execution remains the same
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoApp()
    window.show()
    sys.exit(app.exec_())
