# import cv2
# from ultralytics  import YOLO
# import math
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)
#
# # model
# model = YOLO("yolo-Weights/yolov8n.pt")
#
#
# # object classes
# classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#               "teddy bear", "hair drier", "toothbrush"
#               ]
#
#
# while True:
#     success, img = cap.read()
#     results = model(img, stream=True)
#
#     # coordinates
#     for r in results:
#         boxes = r.boxes
#
#         for box in boxes:
#             # bounding box
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
#
#             # put box in cam
#             cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
#
#             # confidence
#             confidence = math.ceil((box.conf[0]*100))/100
#             print("Confidence --->",confidence)
#
#             # class name
#             cls = int(box.cls[0])
#             print("Class name -->", classNames[cls])
#
#             # object details
#             org = [x1, y1]
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             fontScale = 1
#             color = (255, 0, 0)
#             thickness = 2
#
#             cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
#
#     cv2.imshow('Webcam', img)
#     if cv2.waitKey(1) == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import math
# from ultralytics import YOLO
#
# # Define the dehazing and image enhancement functions (same as before)
# def dark_channel(image, size=15):
#     dark_channel_img = np.min(image, axis=2)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
#     dark_channel_img = cv2.erode(dark_channel_img, kernel)
#     return dark_channel_img
#
# def estimate_atmospheric_light(image, dark_channel_img):
#     num_pixels = image.shape[0] * image.shape[1]
#     num_brightest = int(max(num_pixels * 0.001, 1))
#     dark_flat = dark_channel_img.ravel()
#     indices = np.argsort(dark_flat)[-num_brightest:]
#     brightest_pixels = np.array([image.reshape(-1, 3)[i] for i in indices])
#     atmospheric_light = np.mean(brightest_pixels, axis=0)
#     return atmospheric_light
#
# def estimate_transmission(image, atmospheric_light, omega=0.95, size=15):
#     norm_image = image / atmospheric_light
#     transmission = 1 - omega * dark_channel(norm_image, size)
#     transmission = np.clip(transmission, 0.1, 1)
#     return transmission
#
# def recover_radiance(image, transmission, atmospheric_light, t0=0.1):
#     transmission = np.expand_dims(transmission, axis=2)
#     radiance = (image - atmospheric_light) / transmission + atmospheric_light
#     return np.clip(radiance, 0, 1)
#
# def apply_clahe(image, clip_limit=3.0, tile_grid_size=(8, 8)):
#     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
#     l_clahe = clahe.apply(l)
#     lab_clahe = cv2.merge((l_clahe, a, b))
#     enhanced_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
#     return enhanced_img
#
# def increase_brightness(image, factor=1.2):
#     brightened_image = np.clip(image * factor, 0, 255)
#     return brightened_image.astype(np.uint8)
#
# def enhance_saturation(image, saturation_scale=1.25):
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv)
#     s = np.clip(s * saturation_scale, 0, 255).astype(np.uint8)
#     hsv_enhanced = cv2.merge((h, s, v))
#     color_enhanced_img = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
#     return color_enhanced_img
#
# # Initialize webcam capture
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)
#
# # Load YOLO model
# model = YOLO("yolo-Weights/yolov8n.pt")
#
# # Object classes
# classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#               "teddy bear", "hair drier", "toothbrush"]
#
# while True:
#     success, img = cap.read()
#     if not success:
#         print("Failed to capture image")
#         break
#
#     # Apply image enhancement techniques
#     img = apply_clahe(img)
#     img = increase_brightness(img)
#     img = enhance_saturation(img)
#
#     # Perform object detection on the enhanced image
#     results = model(img, stream=True)
#
#     # Coordinates
#     for r in results:
#         boxes = r.boxes
#
#         for box in boxes:
#             # Bounding box
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(y2), int(y2)  # convert to int values
#
#             # Put box in cam
#             cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
#
#             # Confidence
#             confidence = math.ceil((box.conf[0] * 100)) / 100
#             print("Confidence --->", confidence)
#
#             # Class name
#             cls = int(box.cls[0])
#             if cls < len(classNames):
#                 org = [x1, y1]
#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 fontScale = 1
#                 color = (255, 0, 0)
#                 thickness = 2
#                 cv2.putText(img, f"{classNames[cls]} {confidence:.2f}", org, font, fontScale, color, thickness)
#             else:
#                 print(f"Unknown class index: {cls}")
#
#     cv2.imshow('Webcam', img)
#     if cv2.waitKey(1) == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
import cv2
import numpy as np
import math
from ultralytics import YOLO

# Define the dehazing and image enhancement functions
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

def increase_brightness(image, factor=1.0):
    brightened_image = np.clip(image * factor, 0, 255)
    return brightened_image.astype(np.uint8)

def enhance_saturation(image, saturation_scale=1.0):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * saturation_scale, 0, 255).astype(np.uint8)
    hsv_enhanced = cv2.merge((h, s, v))
    color_enhanced_img = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    return color_enhanced_img

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype(np.uint8)
    return cv2.LUT(image, table)

def reduce_noise(image, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

# Initialize webcam capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

# Object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    # Apply dehazing and image enhancement
    img = img.astype(np.float32) / 255.0
    dark_channel_img = dark_channel(img)
    atmospheric_light = estimate_atmospheric_light(img, dark_channel_img)
    transmission = estimate_transmission(img, atmospheric_light)
    dehazed_img = recover_radiance(img, transmission, atmospheric_light)
    dehazed_img = (dehazed_img * 255).astype(np.uint8)

    # Apply CLAHE, brightness, saturation, gamma, and noise reduction
    dehazed_img = apply_clahe(dehazed_img)
    dehazed_img = increase_brightness(dehazed_img)
    dehazed_img = enhance_saturation(dehazed_img)
    dehazed_img = adjust_gamma(dehazed_img)
    dehazed_img = reduce_noise(dehazed_img)

    # Perform object detection on the enhanced image
    results = model(dehazed_img, stream=True)

    # Display detection results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(dehazed_img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            if cls < len(classNames):
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(dehazed_img, f"{classNames[cls]} {confidence:.2f}", org, font, fontScale, color, thickness)

    # Show the enhanced video
    cv2.imshow('Webcam', dehazed_img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
