"""
Face Detection and Preprocessing Pipeline
Uses OpenCV Haar Cascades + LBP for robust face detection
"""

import cv2
import numpy as np
import os
import base64
from io import BytesIO
from PIL import Image


HAARCASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
HAARCASCADE_ALT_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
PROFILE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_profileface.xml'

face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
face_cascade_alt = cv2.CascadeClassifier(HAARCASCADE_ALT_PATH)
profile_cascade = cv2.CascadeClassifier(PROFILE_CASCADE_PATH)


def decode_image_from_base64(b64_string):
    """Decode base64 image to numpy array (RGB)"""
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    img_bytes = base64.b64decode(b64_string)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def decode_image_from_bytes(img_bytes):
    """Decode image bytes to numpy array (RGB)"""
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def detect_faces(img_rgb, min_size=30):
    """
    Detect faces in image using multiple cascades.
    Returns list of (x, y, w, h) face regions.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = []

    # Primary detector
    detected = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_size, min_size),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(detected) > 0:
        faces.extend(detected.tolist())

    # Alt detector
    detected_alt = face_cascade_alt.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_size, min_size),
    )
    if len(detected_alt) > 0:
        faces.extend(detected_alt.tolist())

    # Profile detector
    detected_profile = profile_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(min_size, min_size),
    )
    if len(detected_profile) > 0:
        faces.extend(detected_profile.tolist())

    # Non-max suppression to remove duplicate detections
    if len(faces) > 0:
        faces = non_max_suppression(np.array(faces))

    return faces


def non_max_suppression(boxes, overlap_thresh=0.3):
    """Remove overlapping bounding boxes"""
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(float)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    return [boxes[i].astype(int).tolist() for i in pick]


def extract_face_region(img_rgb, bbox, margin=0.2):
    """
    Extract face region with margin from image.
    bbox: (x, y, w, h)
    """
    h_img, w_img = img_rgb.shape[:2]
    x, y, w, h = bbox

    # Add margin
    margin_x = int(w * margin)
    margin_y = int(h * margin)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(w_img, x + w + margin_x)
    y2 = min(h_img, y + h + margin_y)

    face = img_rgb[y1:y2, x1:x2]
    return face


def preprocess_face(face_rgb, target_size=64):
    """
    Preprocess face for CNN:
    - Resize to target_size x target_size
    - Apply histogram equalization per channel
    - Normalize to [0, 1]
    """
    # Resize
    face = cv2.resize(face_rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # Per-channel CLAHE for better contrast
    face_lab = cv2.cvtColor(face, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    face_lab[:, :, 0] = clahe.apply(face_lab[:, :, 0])
    face = cv2.cvtColor(face_lab, cv2.COLOR_LAB2RGB)

    return face


def image_to_base64(img_rgb):
    """Convert numpy RGB image to base64 PNG string"""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def detect_and_extract_all_faces(img_rgb):
    """
    Full pipeline: detect faces, extract, preprocess.
    Returns list of {'face_rgb', 'bbox', 'thumbnail_b64'}
    """
    faces_data = []
    bboxes = detect_faces(img_rgb)

    if not bboxes:
        # If no face detected, use whole image
        preprocessed = preprocess_face(img_rgb)
        thumbnail = image_to_base64(cv2.resize(img_rgb, (100, 100)))
        faces_data.append({
            'face_rgb': preprocessed,
            'bbox': [0, 0, img_rgb.shape[1], img_rgb.shape[0]],
            'thumbnail_b64': thumbnail,
            'detected': False
        })
    else:
        for bbox in bboxes:
            face_region = extract_face_region(img_rgb, bbox)
            preprocessed = preprocess_face(face_region)
            thumbnail = image_to_base64(cv2.resize(face_region, (100, 100)))
            faces_data.append({
                'face_rgb': preprocessed,
                'bbox': bbox,
                'thumbnail_b64': thumbnail,
                'detected': True
            })

    return faces_data
