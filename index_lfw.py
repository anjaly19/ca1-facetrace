#!/usr/bin/env python3
"""
FaceTrace V2 - LFW Dataset Indexer
====================================
Uses ArcFace pretrained model for high-accuracy face embeddings.
Much more accurate than the previous custom CNN version.

Usage:
    python index_lfw.py --lfw-path "C:\\path\\to\\lfw_top50"
    python index_lfw.py --lfw-path ./lfw --max-per-person 5 --batch-size 32
"""

import os
import sys
import argparse
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from model_manager import ensure_arcface_model
from recognition_engine import FaceRecognitionEngine
from face_database import FaceDatabase


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--lfw-path', required=True)
    p.add_argument('--max-per-person', type=int, default=10)
    p.add_argument('--min-photos', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=16, help='Batch size for processing')
    p.add_argument('--clear', action='store_true', help='Clear DB before indexing')
    p.add_argument('--limit', type=int, default=0, help='Limit persons (0=all)')
    p.add_argument('--skip-existing', action='store_true', default=True)
    return p.parse_args()


def load_image_rgb(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    args = parse_args()
    lfw_path = os.path.expanduser(args.lfw_path)

    if not os.path.isdir(lfw_path):
        print(f"❌ Path not found: {lfw_path}")
        sys.exit(1)

    print("=" * 65)
    print("  FaceTrace V2 — LFW Indexer (ArcFace Pretrained)")
    print("=" * 65)

    # Load ArcFace model
    print("\n🧠 Loading ArcFace model (downloads on first run ~85MB)...")
    arcface_path, det_path = ensure_arcface_model()

    engine = FaceRecognitionEngine(arcface_path, det_path)
    print(f"\n✓ Engine: {engine.model_info}")

    # Load database (with correct embedding dim)
    db = FaceDatabase(embedding_dim=engine.embedding_dim)

    if args.clear:
        print("🗑  Clearing database...")
        db.clear_all()

    # Discover persons
    persons = {}
    for name in sorted(os.listdir(lfw_path)):
        d = os.path.join(lfw_path, name)
        if not os.path.isdir(d):
            continue
        imgs = sorted([
            os.path.join(d, f) for f in os.listdir(d)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if len(imgs) >= args.min_photos:
            persons[name] = imgs

    if args.limit > 0:
        persons = dict(list(persons.items())[:args.limit])

    total = len(persons)
    total_imgs = sum(min(len(v), args.max_per_person) for v in persons.values())
    print(f"\n📁 Found {total} persons, ~{total_imgs} images to index")

    # Skip existing
    existing = set()
    if args.skip_existing:
        for m in db.metadata:
            existing.add(m.get('title', '').strip())
        if existing:
            print(f"⏭  Skipping {len(existing)} already-indexed persons")

    print(f"\n🚀 Indexing with {engine.model_info['recognizer']}...\n")

    start = time.time()
    total_indexed = 0
    total_skipped = 0
    person_count = 0

    # Batch buffers
    batch_embs = []
    batch_imgs = []
    batch_records = []

    def flush_batch():
        nonlocal total_indexed
        if batch_embs:
            db.add_batch(batch_embs, batch_imgs, batch_records)
            total_indexed += len(batch_embs)
            batch_embs.clear()
            batch_imgs.clear()
            batch_records.clear()

    for name, img_paths in persons.items():
        display = name.replace('_', ' ')

        if display in existing:
            total_skipped += 1
            continue

        person_count += 1
        indexed_this = 0

        for img_path in img_paths[:args.max_per_person]:
            img_rgb = load_image_rgb(img_path)
            if img_rgb is None:
                continue

            try:
                faces = engine.detect_faces(img_rgb)
                if not faces:
                    continue

                face = faces[0]
                emb = engine.extract_embedding(face['face_rgb'])

                batch_embs.append(emb)
                batch_imgs.append(face['face_rgb'])
                batch_records.append({
                    'source_url': f"lfw://dataset/{name}",
                    'title': display,
                    'description': f"LFW · {os.path.basename(img_path)}"
                })
                indexed_this += 1

                # Flush batch
                if len(batch_embs) >= args.batch_size:
                    flush_batch()

            except Exception as e:
                continue

        elapsed = time.time() - start
        rate = (total_indexed + len(batch_embs)) / elapsed if elapsed > 0 else 0
        eta = (total_imgs - total_indexed) / rate if rate > 0 else 0

        status = "✓" if indexed_this > 0 else "✗"
        print(f"  [{person_count:4d}/{total}] {status} {display[:30]:<30} | "
              f"{total_indexed+len(batch_embs):,} indexed | "
              f"{rate:.1f}/s | ETA {eta/60:.1f}min")

    # Final flush
    flush_batch()

    elapsed = time.time() - start
    print()
    print("=" * 65)
    print("  ✅ INDEXING COMPLETE")
    print("=" * 65)
    print(f"  Model used:     {engine.model_info['recognizer']}")
    print(f"  Embedding dim:  {engine.embedding_dim}")
    print(f"  Persons indexed:{person_count:,}")
    print(f"  Persons skipped:{total_skipped:,}")
    print(f"  Faces indexed:  {total_indexed:,}")
    print(f"  Time:           {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Rate:           {total_indexed/elapsed:.1f} faces/sec")
    print(f"  DB total:       {len(db.metadata):,} faces")
    print()
    print("  🚀 Now run: python start.py")
    print("=" * 65)


if __name__ == '__main__':
    main()
