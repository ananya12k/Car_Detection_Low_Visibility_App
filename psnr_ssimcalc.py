import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog, QMessageBox
from skimage.metrics import structural_similarity as ssim


class ImageComparisonApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image PSNR and SSIM Calculator")

        # Set up layout
        layout = QVBoxLayout()

        # Labels for display
        self.label = QLabel("Upload two images to calculate PSNR and SSIM.")
        layout.addWidget(self.label)

        # Buttons to load images and calculate metrics
        self.load_image1_btn = QPushButton("Load Reference Image")
        self.load_image1_btn.clicked.connect(self.load_image1)
        layout.addWidget(self.load_image1_btn)

        self.load_image2_btn = QPushButton("Load Test Image")
        self.load_image2_btn.clicked.connect(self.load_image2)
        layout.addWidget(self.load_image2_btn)

        self.calc_metrics_btn = QPushButton("Calculate PSNR and SSIM")
        self.calc_metrics_btn.clicked.connect(self.calculate_metrics)
        layout.addWidget(self.calc_metrics_btn)

        self.result_label = QLabel("")
        layout.addWidget(self.result_label)

        self.setLayout(layout)

        self.image1 = None
        self.image2 = None

    def load_image1(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Reference Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            self.image1 = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            if self.image1 is None:
                QMessageBox.critical(self, "Error", "Could not load image.")
            else:
                self.label.setText("Reference Image Loaded.")

    def load_image2(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Test Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            self.image2 = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            if self.image2 is None:
                QMessageBox.critical(self, "Error", "Could not load image.")
            else:
                self.label.setText("Test Image Loaded.")

    def calculate_psnr(self, img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        return psnr

    def calculate_ssim(self, img1, img2):
        return ssim(img1, img2, data_range=img2.max() - img2.min())

    def calculate_metrics(self):
        if self.image1 is None or self.image2 is None:
            QMessageBox.warning(self, "Warning", "Please load both images before calculating metrics.")
            return

        # Resize images to match if they are not the same size
        if self.image1.shape != self.image2.shape:
            QMessageBox.warning(self, "Warning", "Images must be the same size for PSNR and SSIM calculation.")
            return

        psnr_value = self.calculate_psnr(self.image1, self.image2)
        ssim_value = self.calculate_ssim(self.image1, self.image2)

        # Display the results
        self.result_label.setText(f"PSNR: {psnr_value:.2f} dB\nSSIM: {ssim_value:.3f}")


# Run the application
app = QApplication(sys.argv)
window = ImageComparisonApp()
window.show()
sys.exit(app.exec_())
