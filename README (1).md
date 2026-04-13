# FaceTrace — AI-Powered Reverse Face Search System

> MSc in Data Science — Data Analytics & Algorithms  
> Continuous Assessment | April 2026

---

## What Is FaceTrace?

FaceTrace is a reverse face search web application built from scratch using a Convolutional Neural Network (CNN). Upload any photo containing a face — the system detects the face, extracts a 128-dimensional embedding vector using the CNN, and searches a locally-indexed database to find visually similar matches ranked by percentage similarity.

It replicates the core concept of commercial tools like PimEyes, running entirely on local hardware with no cloud dependency.

---

## Features

- Custom CNN built from scratch using NumPy only (no TensorFlow/PyTorch)
- Face detection using OpenCV Haar Cascade
- 128-dimensional L2-normalised face embeddings
- Cosine similarity search across indexed face database
- Person-grouped results view
- Adjustable similarity threshold slider
- Dark professional UI
- Flask REST API backend
- LFW dataset bulk indexer

---

## Dataset

**Labeled Faces in the Wild (LFW) — Top 50 Subset**

| Property | Details |
|----------|---------|
| Full dataset | 13,233 images, 5,749 people |
| Subset used | Top 50 most-photographed individuals |
| Image format | JPEG, 250×250 pixels |
| Source | University of Massachusetts Amherst |
| License | Free for academic use |

---

## CNN Architecture

```
Input:  3 × 64 × 64  (RGB face image)
        ↓
Conv2D(16, 3×3) → BatchNorm → ReLU → MaxPool    →  16 × 32 × 32
        ↓
Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool    →  32 × 16 × 16
        ↓
Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool    →  64 × 8 × 8
        ↓
Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool    →  64 × 4 × 4
        ↓
Flatten                                          →  1024
        ↓
Dense(1024 → 256) → ReLU → Dropout(0.3)
        ↓
Dense(256 → 128)
        ↓
L2 Normalise                                     →  128-dim unit vector
        ↓
Cosine Similarity Search
```

All layers (Conv2D, BatchNorm, MaxPool, Dense, Dropout) implemented manually in NumPy.

---

## Project Structure

```
face trace/
├── start.py                  # Start the server
├── index_lfw.py              # Index LFW dataset into database
├── requirements.txt          # Python dependencies
│
├── backend/
│   ├── app.py                # Flask REST API
│   ├── cnn_model.py          # CNN from scratch (NumPy)
│   ├── face_detector.py      # OpenCV face detection pipeline
│   └── face_database.py      # Embedding storage & search
│
├── frontend/
│   └── index.html            # Complete single-page UI
│
└── database/
    ├── embeddings.npy        # Stored face embeddings
    ├── metadata.json         # Face metadata
    └── faces/                # Saved face thumbnails
```

---

## Setup & Installation

### Step 1 — Install Python dependencies

```bash
pip install flask opencv-python numpy scipy scikit-learn pillow
```

### Step 2 — Navigate to project folder

```bash
cd "C:\Users\HP\OneDrive\Desktop\face trace"
```

### Step 3 — Index your LFW dataset

```bash
python index_lfw.py --lfw-path "C:\Users\HP\OneDrive\Desktop\face trace\lfw_top50"
```

This will detect faces in all LFW images, extract CNN embeddings, and store them in the database. Takes approximately 10–20 minutes.

### Step 4 — Start the server

```bash
python start.py
```

### Step 5 — Open in browser

```
http://localhost:5000
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/search` | Search for similar faces |
| POST | `/api/add-face` | Add a face to the database |
| GET | `/api/database` | List all indexed faces |
| DELETE | `/api/face/<id>` | Remove a face |
| GET | `/api/stats` | Database statistics |
| POST | `/api/clear` | Reset database |

### Example Search Request

```bash
curl -X POST http://localhost:5000/api/search \
  -F "image=@photo.jpg" \
  -F "threshold=0.35" \
  -F "top_k=20"
```

### Example Response

```json
{
  "success": true,
  "faces_detected": 1,
  "total_results": 8,
  "total_persons": 3,
  "search_time_ms": 412,
  "results": [
    {
      "id": "a3f2c1b0",
      "title": "George W Bush",
      "similarity": 0.87,
      "similarity_percent": 87.0,
      "source_url": "lfw://dataset/George_W_Bush"
    }
  ]
}
```

---

## How Similarity Works

1. Upload photo → OpenCV detects face region
2. Crop + resize to 64×64 + CLAHE enhancement
3. CNN forward pass → 128-dim L2-normalised embedding
4. Cosine similarity computed against every embedding in database
5. Results above threshold returned, sorted by similarity

**Similarity score guide:**

| Score | Meaning |
|-------|---------|
| 0.8 – 1.0 | Very likely same person |
| 0.5 – 0.8 | Possible match |
| 0.3 – 0.5 | Weak match |
| < 0.3 | Different person |

---

## Why It Cannot Replicate PimEyes

PimEyes crawls **billions of images** from the public internet and indexes them. Our system is limited to a local database for the following reasons:

| Barrier | Explanation |
|---------|-------------|
| No web-scale database | PimEyes has years of crawled data. We have LFW Top 50. |
| Legal restrictions | Scraping Facebook, Instagram etc. violates their Terms of Service |
| Infrastructure | Crawling the internet requires terabytes of storage and dozens of servers |
| GDPR | Collecting facial images of individuals without consent is illegal in the EU |
| Untrained model | Our CNN uses random weights — a trained model (ArcFace) is needed for real-world accuracy |

---

## Development Iterations

| # | Version | Key Change |
|---|---------|-----------|
| 1 | Setup | Dataset selected, project structure created |
| 2 | Basic UI | Upload and display working |
| 3 | CNN | Embedding extraction implemented from scratch |
| 4 | Database | Full search pipeline working end-to-end |
| 5 | UI v2 | Dark theme, similarity cards, person grouping |
| 6 | Internet attempt | Investigated web crawling — unsuccessful (see above) |
| 7 | V2 ArcFace | Pretrained ArcFace + FAISS fast search integrated |

---

## Sustainability

- Runs entirely offline — no cloud services or internet required
- Works on low-cost consumer hardware (no GPU needed)
- No biometric data transmitted externally — full privacy
- Open source stack — no vendor lock-in
- Can be used by humanitarian organisations for offline identity verification

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3.11 | Core language |
| NumPy | CNN implementation from scratch |
| OpenCV | Face detection & image processing |
| Flask | REST API backend |
| HTML / CSS / JavaScript | Frontend UI |
| scikit-learn | Utility functions |
| LFW Dataset | Face image database |

---

## Authors

> Fill in your names and student IDs here

- Student 1: _________________ (ID: _________)
- Student 2: _________________ (ID: _________)

MSc in Data Science — Data Analytics & Algorithms  
April 2026
