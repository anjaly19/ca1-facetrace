"""
FaceTrace V2 - Flask Backend
ArcFace + FAISS powered face search
"""

import os
import sys
import json
import base64
import time
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory

sys.path.insert(0, os.path.dirname(__file__))

from model_manager import ensure_arcface_model
from recognition_engine import FaceRecognitionEngine
from face_database import FaceDatabase

app = Flask(__name__, static_folder='../frontend', static_url_path='')

# ─── Initialize ──────────────────────────────────────────────────────────────
print("[SERVER] FaceTrace V2 starting...")
print("[SERVER] Loading ArcFace model (first run downloads ~85MB)...")

arcface_path, det_path = ensure_arcface_model()
engine = FaceRecognitionEngine(arcface_path, det_path)

print(f"[SERVER] Engine ready: {engine.model_info}")

db = FaceDatabase(embedding_dim=engine.embedding_dim)
print(f"[SERVER] Database: {len(db.metadata)} faces indexed")

if len(db.metadata) == 0:
    print("[SERVER] Empty database — run index_lfw.py to index your LFW dataset")


# ─── CORS ────────────────────────────────────────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, DELETE, OPTIONS'
    return response

@app.route('/api/<path:path>', methods=['OPTIONS'])
def options(path):
    return jsonify({}), 200


# ─── Search ───────────────────────────────────────────────────────────────────
@app.route('/api/search', methods=['POST'])
def search():
    start_time = time.time()
    try:
        # Get image
        if request.files and 'image' in request.files:
            img_bytes = request.files['image'].read()
        elif request.is_json and 'image_b64' in request.json:
            b64 = request.json['image_b64']
            if ',' in b64: b64 = b64.split(',')[1]
            img_bytes = base64.b64decode(b64)
        else:
            return jsonify({'error': 'No image provided'}), 400

        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({'error': 'Cannot decode image'}), 400
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        threshold = float(request.form.get('threshold', 0.3) if request.form else request.json.get('threshold', 0.3))
        top_k = int(request.form.get('top_k', 20) if request.form else request.json.get('top_k', 20))

        # Detect faces
        faces_data = engine.detect_faces(img_rgb)

        all_results = []
        face_thumbnails = []

        for face_info in faces_data:
            emb = engine.extract_embedding(face_info['face_rgb'])
            results = db.search(emb, top_k=top_k, threshold=threshold)
            all_results.extend(results)
            face_thumbnails.append({
                'thumbnail': face_info['thumbnail_b64'],
                'bbox': face_info['bbox'],
                'detected': face_info['detected'],
                'match_count': len(results)
            })

        # Deduplicate
        seen = set()
        unique = []
        for r in sorted(all_results, key=lambda x: x['similarity'], reverse=True):
            if r['id'] not in seen:
                seen.add(r['id'])
                unique.append(r)

        # Group by person
        grouped = {}
        for r in unique:
            name = r.get('title', 'Unknown')
            if name not in grouped:
                grouped[name] = {
                    'name': name,
                    'best_similarity': r['similarity'],
                    'best_similarity_percent': r['similarity_percent'],
                    'matches': []
                }
            grouped[name]['matches'].append(r)

        persons = sorted(grouped.values(), key=lambda x: x['best_similarity'], reverse=True)

        elapsed_ms = int((time.time() - start_time) * 1000)

        return jsonify({
            'success': True,
            'faces_detected': len(faces_data),
            'results': unique[:top_k],
            'persons': persons[:top_k],
            'total_results': len(unique),
            'total_persons': len(persons),
            'search_time_ms': elapsed_ms,
            'face_thumbnails': face_thumbnails,
            'database_size': len(db.metadata),
            'model_info': engine.model_info
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─── Add Face ─────────────────────────────────────────────────────────────────
@app.route('/api/add-face', methods=['POST'])
def add_face():
    try:
        if request.files and 'image' in request.files:
            img_bytes = request.files['image'].read()
        elif request.is_json and 'image_b64' in request.json:
            b64 = request.json['image_b64']
            if ',' in b64: b64 = b64.split(',')[1]
            img_bytes = base64.b64decode(b64)
        else:
            return jsonify({'error': 'No image provided'}), 400

        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        title = request.form.get('title', '') or (request.json.get('title','') if request.is_json else '')
        source_url = request.form.get('source_url', '') or (request.json.get('source_url','') if request.is_json else '')
        description = request.form.get('description', '') or (request.json.get('description','') if request.is_json else '')

        faces_data = engine.detect_faces(img_rgb)
        added = []
        for face_info in faces_data:
            emb = engine.extract_embedding(face_info['face_rgb'])
            fid = db.add_face(emb, face_info['face_rgb'],
                              source_url=source_url,
                              title=title or 'Unknown',
                              description=description)
            added.append(fid)

        return jsonify({'success': True, 'faces_added': len(added),
                        'face_ids': added, 'total_in_db': len(db.metadata)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── Database endpoints ────────────────────────────────────────────────────────
@app.route('/api/database')
def list_database():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    faces, total = db.get_all_faces(page, per_page)
    return jsonify({'faces': faces, 'total': total, 'page': page,
                    'pages': (total+per_page-1)//per_page})

@app.route('/api/face/<face_id>', methods=['DELETE'])
def remove_face(face_id):
    ok = db.remove_face(face_id)
    return jsonify({'success': ok, 'total_in_db': len(db.metadata)}) if ok else (jsonify({'error': 'Not found'}), 404)

@app.route('/api/stats')
def stats():
    s = db.get_stats()
    s['model_info'] = engine.model_info
    return jsonify(s)

@app.route('/api/clear', methods=['POST'])
def clear():
    db.clear_all()
    return jsonify({'success': True, 'total_in_db': 0,
                    'message': 'Database cleared. Run index_lfw.py to re-index.'})

@app.route('/api/faces/<filename>')
def serve_face(filename):
    return send_from_directory(os.path.join(os.path.dirname(__file__), '..', 'database', 'faces'), filename)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    frontend = os.path.join(os.path.dirname(__file__), '..', 'frontend')
    full = os.path.join(frontend, path)
    if path and os.path.exists(full):
        return send_from_directory(frontend, path)
    return send_from_directory(frontend, 'index.html')


if __name__ == '__main__':
    print(f"\n[SERVER] ✓ ArcFace: {engine.model_info['recognizer']}")
    print(f"[SERVER] ✓ Detector: {engine.model_info['detector']}")
    print(f"[SERVER] ✓ Search: {db.get_stats()['search_engine']}")
    print(f"[SERVER] ✓ Faces in DB: {len(db.metadata)}")
    print(f"\n[SERVER] 🚀 http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
