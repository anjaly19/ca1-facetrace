"""
Face Database with FAISS Fast Search
=====================================
Uses Facebook AI Similarity Search (FAISS) for fast nearest-neighbor search.
Falls back to NumPy cosine similarity if FAISS not installed.

FAISS can search 1 million faces in milliseconds.
"""

import os
import json
import numpy as np
import cv2
import uuid
import base64
from datetime import datetime

DB_DIR = os.path.join(os.path.dirname(__file__), '..', 'database')
FACES_DIR = os.path.join(DB_DIR, 'faces')
METADATA_FILE = os.path.join(DB_DIR, 'metadata.json')
EMBEDDINGS_FILE = os.path.join(DB_DIR, 'embeddings.npy')
FAISS_INDEX_FILE = os.path.join(DB_DIR, 'faiss.index')

os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
    print("[DB] FAISS available — using fast similarity search")
except ImportError:
    FAISS_AVAILABLE = False
    print("[DB] FAISS not installed — using NumPy search (still works, slightly slower)")


class FaceDatabase:
    def __init__(self, embedding_dim=512):
        self.embedding_dim = embedding_dim
        self.metadata = []
        self.embeddings = []
        self.faiss_index = None
        self._load()

    def _build_faiss_index(self):
        """Build FAISS index from current embeddings"""
        if not FAISS_AVAILABLE or not self.embeddings:
            return
        emb_matrix = np.array(self.embeddings, dtype=np.float32)
        # Use Inner Product index (cosine similarity with L2-normalized vectors)
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(emb_matrix)
        self.faiss_index = index
        print(f"[DB] FAISS index built with {index.ntotal} vectors")

    def _load(self):
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                self.metadata = json.load(f)

        if os.path.exists(EMBEDDINGS_FILE) and self.metadata:
            arr = np.load(EMBEDDINGS_FILE)
            # Handle dimension mismatch (e.g. switching from 128 to 512)
            if arr.shape[1] != self.embedding_dim:
                print(f"[DB] Embedding dim mismatch ({arr.shape[1]} vs {self.embedding_dim}). Clearing DB.")
                self.metadata = []
                self.embeddings = []
                self._save()
                return
            self.embeddings = list(arr)
            print(f"[DB] Loaded {len(self.embeddings)} embeddings ({self.embedding_dim}-dim)")

            if FAISS_AVAILABLE:
                # Try loading saved FAISS index
                if os.path.exists(FAISS_INDEX_FILE):
                    try:
                        self.faiss_index = faiss.read_index(FAISS_INDEX_FILE)
                        print(f"[DB] FAISS index loaded ({self.faiss_index.ntotal} vectors)")
                    except:
                        self._build_faiss_index()
                else:
                    self._build_faiss_index()

    def _save(self):
        with open(METADATA_FILE, 'w') as f:
            json.dump(self.metadata, f)
        if self.embeddings:
            np.save(EMBEDDINGS_FILE, np.array(self.embeddings, dtype=np.float32))
            if FAISS_AVAILABLE and self.faiss_index:
                faiss.write_index(self.faiss_index, FAISS_INDEX_FILE)

    def add_face(self, embedding, image_rgb, source_url='', title='', description=''):
        face_id = str(uuid.uuid4())[:8]

        # Save face image
        face_path = os.path.join(FACES_DIR, f"{face_id}.jpg")
        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(face_path, img_bgr)

        # Thumbnail
        thumb = cv2.resize(img_bgr, (120, 120))
        _, buf = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
        thumb_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"

        record = {
            'id': face_id,
            'image_path': face_path,
            'source_url': source_url,
            'title': title,
            'description': description,
            'thumbnail': thumb_b64,
            'added_at': datetime.now().isoformat()
        }

        emb = np.array(embedding, dtype=np.float32)
        self.embeddings.append(emb)
        self.metadata.append(record)

        # Add to FAISS index
        if FAISS_AVAILABLE:
            if self.faiss_index is None:
                self._build_faiss_index()
            else:
                self.faiss_index.add(emb[np.newaxis])

        return face_id

    def add_batch(self, embeddings, images_rgb, records):
        """Bulk add for fast LFW indexing"""
        emb_batch = []
        for emb, img, record in zip(embeddings, images_rgb, records):
            face_id = str(uuid.uuid4())[:8]
            record['id'] = face_id
            record['added_at'] = datetime.now().isoformat()

            # Save image
            face_path = os.path.join(FACES_DIR, f"{face_id}.jpg")
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(face_path, img_bgr)

            # Thumbnail
            thumb = cv2.resize(img_bgr, (120, 120))
            _, buf = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
            record['thumbnail'] = f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"
            record['image_path'] = face_path

            emb_f = np.array(emb, dtype=np.float32)
            self.embeddings.append(emb_f)
            self.metadata.append(record)
            emb_batch.append(emb_f)

        # Rebuild FAISS index in bulk (much faster than adding one by one)
        if FAISS_AVAILABLE and emb_batch:
            if self.faiss_index is None:
                self._build_faiss_index()
            else:
                self.faiss_index.add(np.array(emb_batch, dtype=np.float32))

        self._save()

    def search(self, query_embedding, top_k=20, threshold=0.3):
        if not self.embeddings:
            return []

        query = np.array(query_embedding, dtype=np.float32)

        if FAISS_AVAILABLE and self.faiss_index and self.faiss_index.ntotal > 0:
            # FAISS search - extremely fast
            k = min(top_k * 3, self.faiss_index.ntotal)
            scores, indices = self.faiss_index.search(query[np.newaxis], k)
            scores = scores[0]
            indices = indices[0]

            results = []
            for score, idx in zip(scores, indices):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                if float(score) < threshold:
                    continue
                meta = self.metadata[idx].copy()
                meta['similarity'] = float(score)
                meta['similarity_percent'] = round(float(score) * 100, 1)
                results.append(meta)

            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:top_k]

        else:
            # NumPy fallback
            db_matrix = np.array(self.embeddings, dtype=np.float32)
            similarities = db_matrix @ query
            above = similarities >= threshold
            indices = np.where(above)[0]
            if len(indices) == 0:
                return []
            sorted_idx = indices[np.argsort(similarities[indices])[::-1]][:top_k]
            results = []
            for idx in sorted_idx:
                meta = self.metadata[idx].copy()
                meta['similarity'] = float(similarities[idx])
                meta['similarity_percent'] = round(float(similarities[idx]) * 100, 1)
                results.append(meta)
            return results

    def remove_face(self, face_id):
        idx = next((i for i, m in enumerate(self.metadata) if m['id'] == face_id), None)
        if idx is None:
            return False
        path = self.metadata[idx].get('image_path', '')
        if os.path.exists(path):
            os.remove(path)
        self.embeddings.pop(idx)
        self.metadata.pop(idx)
        # Rebuild FAISS index
        if FAISS_AVAILABLE:
            self._build_faiss_index()
        self._save()
        return True

    def get_all_faces(self, page=1, per_page=20):
        start = (page-1)*per_page
        return self.metadata[start:start+per_page], len(self.metadata)

    def clear_all(self):
        for m in self.metadata:
            p = m.get('image_path', '')
            if os.path.exists(p):
                os.remove(p)
        self.embeddings = []
        self.metadata = []
        self.faiss_index = None
        self._save()

    def get_stats(self):
        emb_size = os.path.getsize(EMBEDDINGS_FILE)/1024/1024 if os.path.exists(EMBEDDINGS_FILE) else 0
        return {
            'total_faces': len(self.metadata),
            'embedding_dim': self.embedding_dim,
            'faiss_enabled': FAISS_AVAILABLE,
            'database_size_mb': round(emb_size, 2),
            'search_engine': 'FAISS' if FAISS_AVAILABLE else 'NumPy'
        }
