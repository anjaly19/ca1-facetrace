"""
Face Recognition Engine
========================
Primary:  ArcFace R50 via ONNX Runtime (pretrained on 5.8M images)
Fallback: Custom CNN from scratch (NumPy only)

ArcFace produces 512-dimensional embeddings with state-of-the-art accuracy.
"""

import numpy as np
import cv2
import os
import sys

# ─── ArcFace ONNX Engine ──────────────────────────────────────────────────────

class ArcFaceEngine:
    """
    ArcFace recognition model via ONNX Runtime.
    Pretrained on MS1MV3: 5.8M images, 85k identities.
    Produces 512-dim L2-normalized embeddings.
    """
    def __init__(self, model_path):
        import onnxruntime as ort
        print(f"[ArcFace] Loading model: {model_path}")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.input_size = (input_shape[3], input_shape[2])  # (W, H)
        print(f"[ArcFace] Ready. Input size: {self.input_size}, Output: 512-dim")
        self.embedding_dim = 512

    def preprocess(self, face_rgb):
        """Preprocess face image for ArcFace input"""
        h, w = self.input_size[1], self.input_size[0]
        face = cv2.resize(face_rgb, (w, h))
        face = face.astype(np.float32)
        # Normalize to [-1, 1] (ArcFace standard)
        face = (face - 127.5) / 128.0
        # HWC -> CHW -> NCHW
        face = face.transpose(2, 0, 1)[np.newaxis]
        return face

    def extract_embedding(self, face_rgb):
        """Extract 512-dim L2-normalized embedding"""
        inp = self.preprocess(face_rgb)
        out = self.session.run(None, {self.input_name: inp})[0]
        emb = out[0]
        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.astype(np.float32)


# ─── Fallback Custom CNN (NumPy) ──────────────────────────────────────────────

class ConvLayer:
    def __init__(self, num_filters, filter_size, in_channels, stride=1, padding=0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        scale = np.sqrt(2.0 / (filter_size * filter_size * in_channels))
        self.filters = np.random.randn(num_filters, in_channels, filter_size, filter_size).astype(np.float32) * scale
        self.biases = np.zeros(num_filters, dtype=np.float32)

    def forward(self, x):
        batch, in_ch, h, w = x.shape
        p, s, f = self.padding, self.stride, self.filter_size
        if p > 0:
            x = np.pad(x, ((0,0),(0,0),(p,p),(p,p)))
        oh = (h + 2*p - f) // s + 1
        ow = (w + 2*p - f) // s + 1
        out = np.zeros((batch, self.num_filters, oh, ow), dtype=np.float32)
        for i in range(oh):
            for j in range(ow):
                xs = x[:, :, i*s:i*s+f, j*s:j*s+f]
                out[:, :, i, j] = np.tensordot(xs, self.filters, axes=([1,2,3],[1,2,3])) + self.biases
        return out

class MaxPool:
    def forward(self, x, p=2, s=2):
        b, c, h, w = x.shape
        oh, ow = (h-p)//s+1, (w-p)//s+1
        out = np.zeros((b, c, oh, ow), dtype=np.float32)
        for i in range(oh):
            for j in range(ow):
                out[:,:,i,j] = np.max(x[:,:,i*s:i*s+p,j*s:j*s+p], axis=(2,3))
        return out

class DenseLayer:
    def __init__(self, in_f, out_f):
        self.W = (np.random.randn(in_f, out_f) * np.sqrt(2.0/in_f)).astype(np.float32)
        self.b = np.zeros(out_f, dtype=np.float32)
    def forward(self, x):
        return x @ self.W + self.b

class FallbackCNN:
    """Custom CNN from scratch - 128-dim embeddings"""
    def __init__(self):
        np.random.seed(42)
        self.conv1 = ConvLayer(16, 3, 3, padding=1)
        self.conv2 = ConvLayer(32, 3, 16, padding=1)
        self.conv3 = ConvLayer(64, 3, 32, padding=1)
        self.conv4 = ConvLayer(64, 3, 64, padding=1)
        self.pool = MaxPool()
        self.fc1 = DenseLayer(1024, 256)
        self.fc2 = DenseLayer(256, 128)
        self.embedding_dim = 128
        print("[CNN] Custom CNN initialized (fallback mode)")

    def extract_embedding(self, face_rgb):
        face = cv2.resize(face_rgb, (64, 64)).astype(np.float32) / 255.0
        x = face.transpose(2,0,1)[np.newaxis]
        x = self.pool.forward(np.maximum(0, self.conv1.forward(x)))
        x = self.pool.forward(np.maximum(0, self.conv2.forward(x)))
        x = self.pool.forward(np.maximum(0, self.conv3.forward(x)))
        x = self.pool.forward(np.maximum(0, self.conv4.forward(x)))
        x = x.reshape(1, -1)
        x = np.maximum(0, self.fc1.forward(x))
        x = self.fc2.forward(x)[0]
        n = np.linalg.norm(x)
        return (x / n if n > 0 else x).astype(np.float32)


# ─── SCRFD Face Detector (ONNX) ───────────────────────────────────────────────

class SCRFDDetector:
    """SCRFD face detector from InsightFace - much more accurate than Haar cascades"""
    def __init__(self, model_path):
        import onnxruntime as ort
        print(f"[SCRFD] Loading detector: {model_path}")
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        inp = self.session.get_inputs()[0].shape
        self.input_size = (640, 640)
        self.center_cache = {}
        print("[SCRFD] Detector ready")

    def detect(self, img_rgb, threshold=0.5):
        """Detect faces. Returns list of (x1,y1,x2,y2,score) bboxes."""
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        ih, iw = self.input_size

        # Resize keeping aspect ratio
        scale = min(ih/h, iw/w)
        nh, nw = int(h*scale), int(w*scale)
        resized = cv2.resize(img, (nw, nh))
        padded = np.zeros((ih, iw, 3), dtype=np.uint8)
        padded[:nh, :nw] = resized

        blob = padded.astype(np.float32).transpose(2,0,1)[np.newaxis]

        try:
            outputs = self.session.run(None, {self.input_name: blob})
            # Parse outputs - simplified bbox extraction
            faces = []
            # outputs[0] = scores, outputs[1] = bboxes
            scores = outputs[0].flatten() if len(outputs) > 0 else []
            bboxes = outputs[1].reshape(-1, 4) if len(outputs) > 1 else []

            for i, score in enumerate(scores):
                if score >= threshold and i < len(bboxes):
                    x1, y1, x2, y2 = bboxes[i]
                    # Scale back to original image
                    x1 = max(0, int(x1 / scale))
                    y1 = max(0, int(y1 / scale))
                    x2 = min(w, int(x2 / scale))
                    y2 = min(h, int(y2 / scale))
                    if x2 > x1 and y2 > y1:
                        faces.append((x1, y1, x2, y2, float(score)))

            faces.sort(key=lambda x: x[4], reverse=True)
            return faces
        except Exception as e:
            return []


# ─── Haar Cascade Fallback Detector ──────────────────────────────────────────

class HaarDetector:
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.cascade2 = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
        )

    def detect(self, img_rgb, threshold=0.5):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = []
        for cascade in [self.cascade, self.cascade2]:
            det = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
            if len(det) > 0:
                for (x,y,w,h) in det:
                    faces.append((x, y, x+w, y+h, 0.9))
        # Deduplicate
        seen = set()
        unique = []
        for f in faces:
            key = (f[0]//20, f[1]//20)
            if key not in seen:
                seen.add(key)
                unique.append(f)
        return unique


# ─── Main Recognition Engine ─────────────────────────────────────────────────

class FaceRecognitionEngine:
    """
    Unified engine that uses ArcFace if available, else fallback CNN.
    """
    def __init__(self, arcface_path=None, detector_path=None):
        # Recognition model
        self.using_arcface = False
        if arcface_path and os.path.exists(arcface_path):
            try:
                self.recognizer = ArcFaceEngine(arcface_path)
                self.using_arcface = True
                print("[ENGINE] ✓ Using ArcFace (pretrained, 512-dim)")
            except Exception as e:
                print(f"[ENGINE] ArcFace failed: {e}, falling back to custom CNN")
                self.recognizer = FallbackCNN()
        else:
            self.recognizer = FallbackCNN()

        # Detection model
        self.using_scrfd = False
        if detector_path and os.path.exists(detector_path):
            try:
                self.detector = SCRFDDetector(detector_path)
                self.using_scrfd = True
                print("[ENGINE] ✓ Using SCRFD detector (pretrained)")
            except Exception as e:
                print(f"[ENGINE] SCRFD failed: {e}, using Haar cascades")
                self.detector = HaarDetector()
        else:
            self.detector = HaarDetector()
            print("[ENGINE] Using Haar cascade detector")

        self.embedding_dim = self.recognizer.embedding_dim

    def detect_faces(self, img_rgb):
        """Returns list of face crops (RGB) with metadata"""
        faces_data = []
        detections = self.detector.detect(img_rgb)

        if not detections:
            # Use whole image
            face = self._preprocess_crop(img_rgb)
            thumb = self._make_thumbnail(img_rgb)
            faces_data.append({'face_rgb': face, 'bbox': [0,0,img_rgb.shape[1],img_rgb.shape[0]],
                                'thumbnail_b64': thumb, 'detected': False, 'score': 0.5})
        else:
            h, w = img_rgb.shape[:2]
            for (x1, y1, x2, y2, score) in detections[:5]:
                # Add margin
                mx = int((x2-x1) * 0.2)
                my = int((y2-y1) * 0.2)
                cx1 = max(0, x1-mx); cy1 = max(0, y1-my)
                cx2 = min(w, x2+mx); cy2 = min(h, y2+my)
                crop = img_rgb[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue
                face = self._preprocess_crop(crop)
                thumb = self._make_thumbnail(crop)
                faces_data.append({
                    'face_rgb': face,
                    'bbox': [x1, y1, x2-x1, y2-y1],
                    'thumbnail_b64': thumb,
                    'detected': True,
                    'score': score
                })

        return faces_data

    def _preprocess_crop(self, img_rgb):
        """Preprocess face crop"""
        size = self.recognizer.input_size if hasattr(self.recognizer, 'input_size') else (112, 112)
        face = cv2.resize(img_rgb, size)
        return face

    def _make_thumbnail(self, img_rgb):
        import base64
        thumb = cv2.resize(img_rgb, (120, 120))
        thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode('.jpg', thumb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return f"data:image/jpeg;base64,{__import__('base64').b64encode(buf).decode()}"

    def extract_embedding(self, face_rgb):
        return self.recognizer.extract_embedding(face_rgb)

    @property
    def model_info(self):
        return {
            'recognizer': 'ArcFace R50 (ONNX)' if self.using_arcface else 'Custom CNN (NumPy)',
            'detector': 'SCRFD (ONNX)' if self.using_scrfd else 'Haar Cascade (OpenCV)',
            'embedding_dim': self.embedding_dim,
            'pretrained': self.using_arcface
        }
