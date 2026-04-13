"""
Custom CNN Implementation from Scratch using NumPy
Architecture: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> FC -> Embedding
This is a lightweight Siamese-style CNN for face embeddings.
"""

import numpy as np
import pickle
import os


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class ConvLayer:
    """2D Convolution Layer"""
    def __init__(self, num_filters, filter_size, in_channels, stride=1, padding=0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding
        # He initialization
        scale = np.sqrt(2.0 / (filter_size * filter_size * in_channels))
        self.filters = np.random.randn(num_filters, in_channels, filter_size, filter_size) * scale
        self.biases = np.zeros(num_filters)
        self.last_input = None

    def forward(self, x):
        self.last_input = x
        batch_size, in_channels, h, w = x.shape
        p = self.padding
        s = self.stride
        f = self.filter_size

        if p > 0:
            x_pad = np.pad(x, ((0,0),(0,0),(p,p),(p,p)), mode='constant')
        else:
            x_pad = x

        out_h = (h + 2*p - f) // s + 1
        out_w = (w + 2*p - f) // s + 1
        output = np.zeros((batch_size, self.num_filters, out_h, out_w))

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * s
                h_end = h_start + f
                w_start = j * s
                w_end = w_start + f
                x_slice = x_pad[:, :, h_start:h_end, w_start:w_end]
                # x_slice: (batch, in_ch, f, f)
                # filters: (num_filters, in_ch, f, f)
                output[:, :, i, j] = np.tensordot(x_slice, self.filters, axes=([1,2,3],[1,2,3])) + self.biases

        return output


class MaxPoolLayer:
    """Max Pooling Layer"""
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.last_input = None

    def forward(self, x):
        self.last_input = x
        batch_size, channels, h, w = x.shape
        p = self.pool_size
        s = self.stride
        out_h = (h - p) // s + 1
        out_w = (w - p) // s + 1
        output = np.zeros((batch_size, channels, out_h, out_w))

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * s
                h_end = h_start + p
                w_start = j * s
                w_end = w_start + p
                output[:, :, i, j] = np.max(x[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))

        return output


class BatchNormLayer:
    """Batch Normalization"""
    def __init__(self, num_features, eps=1e-8):
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.eps = eps
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.training = True

    def forward(self, x):
        if self.training:
            mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            var = np.var(x, axis=(0, 2, 3), keepdims=True)
            x_norm = (x - mean) / np.sqrt(var + self.eps)
        else:
            mean = self.running_mean.reshape(1, -1, 1, 1)
            var = self.running_var.reshape(1, -1, 1, 1)
            x_norm = (x - mean) / np.sqrt(var + self.eps)

        gamma = self.gamma.reshape(1, -1, 1, 1)
        beta = self.beta.reshape(1, -1, 1, 1)
        return gamma * x_norm + beta


class DropoutLayer:
    def __init__(self, rate=0.5):
        self.rate = rate
        self.training = True
        self.mask = None

    def forward(self, x):
        if self.training:
            self.mask = (np.random.rand(*x.shape) > self.rate) / (1 - self.rate)
            return x * self.mask
        return x


class DenseLayer:
    """Fully Connected Layer"""
    def __init__(self, in_features, out_features):
        scale = np.sqrt(2.0 / in_features)
        self.weights = np.random.randn(in_features, out_features) * scale
        self.biases = np.zeros(out_features)
        self.last_input = None

    def forward(self, x):
        self.last_input = x
        return np.dot(x, self.weights) + self.biases


class FaceCNN:
    """
    Custom CNN for Face Embedding Extraction
    Input: (batch, 3, 64, 64) RGB face images normalized to [0,1]
    Output: 128-dimensional L2-normalized embedding vector
    
    Architecture:
    - Conv(3->16, 3x3, pad=1) -> BN -> ReLU -> MaxPool(2)  => 16x32x32
    - Conv(16->32, 3x3, pad=1) -> BN -> ReLU -> MaxPool(2) => 32x16x16
    - Conv(32->64, 3x3, pad=1) -> BN -> ReLU -> MaxPool(2) => 64x8x8
    - Conv(64->64, 3x3, pad=1) -> BN -> ReLU -> MaxPool(2) => 64x4x4
    - Flatten => 1024
    - FC(1024 -> 256) -> ReLU -> Dropout
    - FC(256 -> 128) -> L2 Normalize
    """

    def __init__(self):
        np.random.seed(42)
        # Conv blocks
        self.conv1 = ConvLayer(16, 3, 3, padding=1)
        self.bn1 = BatchNormLayer(16)
        self.pool1 = MaxPoolLayer(2, 2)

        self.conv2 = ConvLayer(32, 3, 16, padding=1)
        self.bn2 = BatchNormLayer(32)
        self.pool2 = MaxPoolLayer(2, 2)

        self.conv3 = ConvLayer(64, 3, 32, padding=1)
        self.bn3 = BatchNormLayer(64)
        self.pool3 = MaxPoolLayer(2, 2)

        self.conv4 = ConvLayer(64, 3, 64, padding=1)
        self.bn4 = BatchNormLayer(64)
        self.pool4 = MaxPoolLayer(2, 2)

        # Dense layers (64 * 4 * 4 = 1024)
        self.fc1 = DenseLayer(1024, 256)
        self.dropout = DropoutLayer(0.3)
        self.fc2 = DenseLayer(256, 128)
        self.training = False

    def set_training(self, mode):
        self.training = mode
        for layer in [self.bn1, self.bn2, self.bn3, self.bn4]:
            layer.training = mode
        self.dropout.training = mode

    def forward(self, x):
        """
        x: numpy array shape (batch, 3, 64, 64), values in [0, 1]
        returns: (batch, 128) L2-normalized embeddings
        """
        # Block 1
        x = self.conv1.forward(x)
        x = self.bn1.forward(x)
        x = relu(x)
        x = self.pool1.forward(x)

        # Block 2
        x = self.conv2.forward(x)
        x = self.bn2.forward(x)
        x = relu(x)
        x = self.pool2.forward(x)

        # Block 3
        x = self.conv3.forward(x)
        x = self.bn3.forward(x)
        x = relu(x)
        x = self.pool3.forward(x)

        # Block 4
        x = self.conv4.forward(x)
        x = self.bn4.forward(x)
        x = relu(x)
        x = self.pool4.forward(x)

        # Flatten
        x = x.reshape(x.shape[0], -1)  # (batch, 1024)

        # FC layers
        x = self.fc1.forward(x)
        x = relu(x)
        x = self.dropout.forward(x)
        x = self.fc2.forward(x)

        # L2 normalize
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        x = x / norms

        return x

    def extract_embedding(self, face_img_rgb):
        """
        face_img_rgb: numpy array (H, W, 3) uint8
        returns: 128-dim embedding vector
        """
        import cv2
        # Resize to 64x64
        face = cv2.resize(face_img_rgb, (64, 64))
        # Normalize
        face = face.astype(np.float32) / 255.0
        # HWC -> CHW -> add batch
        face = face.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 64, 64)
        self.set_training(False)
        embedding = self.forward(face)
        return embedding[0]  # (128,)

    def save(self, path):
        data = {
            'conv1_filters': self.conv1.filters, 'conv1_biases': self.conv1.biases,
            'conv2_filters': self.conv2.filters, 'conv2_biases': self.conv2.biases,
            'conv3_filters': self.conv3.filters, 'conv3_biases': self.conv3.biases,
            'conv4_filters': self.conv4.filters, 'conv4_biases': self.conv4.biases,
            'bn1_gamma': self.bn1.gamma, 'bn1_beta': self.bn1.beta,
            'bn2_gamma': self.bn2.gamma, 'bn2_beta': self.bn2.beta,
            'bn3_gamma': self.bn3.gamma, 'bn3_beta': self.bn3.beta,
            'bn4_gamma': self.bn4.gamma, 'bn4_beta': self.bn4.beta,
            'fc1_weights': self.fc1.weights, 'fc1_biases': self.fc1.biases,
            'fc2_weights': self.fc2.weights, 'fc2_biases': self.fc2.biases,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"[CNN] Model saved to {path}")

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.conv1.filters = data['conv1_filters']; self.conv1.biases = data['conv1_biases']
        self.conv2.filters = data['conv2_filters']; self.conv2.biases = data['conv2_biases']
        self.conv3.filters = data['conv3_filters']; self.conv3.biases = data['conv3_biases']
        self.conv4.filters = data['conv4_filters']; self.conv4.biases = data['conv4_biases']
        self.bn1.gamma = data['bn1_gamma']; self.bn1.beta = data['bn1_beta']
        self.bn2.gamma = data['bn2_gamma']; self.bn2.beta = data['bn2_beta']
        self.bn3.gamma = data['bn3_gamma']; self.bn3.beta = data['bn3_beta']
        self.bn4.gamma = data['bn4_gamma']; self.bn4.beta = data['bn4_beta']
        self.fc1.weights = data['fc1_weights']; self.fc1.biases = data['fc1_biases']
        self.fc2.weights = data['fc2_weights']; self.fc2.biases = data['fc2_biases']
        print(f"[CNN] Model loaded from {path}")


# ─── Contrastive Loss Training (Siamese Network) ─────────────────────────────

def contrastive_loss(emb1, emb2, label, margin=1.0):
    """
    label = 1 if same person, 0 if different
    Loss = label * d^2 + (1-label) * max(0, margin-d)^2
    """
    d = np.linalg.norm(emb1 - emb2, axis=1)
    loss = label * d**2 + (1 - label) * np.maximum(0, margin - d)**2
    return np.mean(loss), d


def cosine_similarity(emb1, emb2):
    """Cosine similarity between two normalized embeddings"""
    return np.dot(emb1, emb2)  # both L2-normalized


def find_similar_faces(query_embedding, database, top_k=10, threshold=0.5):
    """
    database: list of {'id', 'embedding', 'image_path', 'source_url', 'title'}
    Returns top_k most similar faces above threshold
    """
    results = []
    for entry in database:
        db_emb = np.array(entry['embedding'])
        sim = cosine_similarity(query_embedding, db_emb)
        if sim >= threshold:
            results.append({
                'id': entry['id'],
                'similarity': float(sim),
                'image_path': entry.get('image_path', ''),
                'source_url': entry.get('source_url', ''),
                'title': entry.get('title', ''),
                'thumbnail': entry.get('thumbnail', '')
            })

    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_k]
