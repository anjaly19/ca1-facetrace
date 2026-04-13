#!/usr/bin/env python3
"""FaceTrace V2 - Start Script"""
import os, sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 'backend')

print("=" * 65)
print("  FaceTrace V2 — Reverse Face Search")
print("  ArcFace (pretrained) + FAISS (fast search)")
print("=" * 65)
print()

os.system(f"{sys.executable} backend/app.py")
