# Golf Lab

An AI-powered golf swing analysis platform. Upload a swing video and get instant biomechanical analysis, fault detection with targeted practice drills, pro swing comparisons, and progress tracking over time.

## Features

### Swing Analysis
- **Pose Estimation** — MediaPipe-powered keypoint extraction (13 body joints per frame with 3D world coordinates)
- **8-Phase Segmentation** — Automatic detection of address, takeaway, backswing, top, downswing, impact, follow-through, and finish
- **Biomechanical Metrics** — Hip/shoulder rotation, X-factor, spine tilt, lead knee flex, and tempo ratio
- **Per-Phase Scoring** — Letter grades (A–F) for each swing phase so you know exactly where you're losing points

### Fault Detection
Detects 8 common swing faults with severity scoring, descriptions, and correction guidance:
- Lateral Sway & Slide
- Early Extension
- Chicken Wing
- Casting / Early Release
- Over the Top
- Reverse Pivot
- Excessive Head Movement

### Practice & Drills
- **Dedicated training page** with targeted exercises for every detected fault
- 3 progressive drills per fault (beginner → intermediate → advanced)
- Step-by-step instructions with equipment lists, rep guidance, and "why it works" explanations
- Direct links from fault cards to relevant drills

### Pro Comparison
- Database of 16 PGA/LPGA tour pros (Tiger Woods, Rory McIlroy, Scottie Scheffler, Nelly Korda, and more)
- Weighted similarity scoring across 8 metrics
- Swing archetype classification (Modern Rotational, Stack & Tilt, Compact, etc.)
- Side-by-side skeleton overlays with pro angle annotations

### Progress Tracking
- **Score trend** — Line chart tracking overall swing score across sessions
- **Metrics trend** — Multi-line chart with toggleable metrics (tempo, X-factor, hip rotation, etc.)
- **Fault frequency** — Bar chart showing most recurring faults to prioritize practice
- Inline score badges on session history for quick comparison

### Shot Dispersion *(coming soon)*
- Predict shot shape and dispersion patterns from swing data

## Architecture

```
golf_lab/
├── frontend/         # React + TypeScript + Vite
│   ├── src/pages/    # Upload, Analysis, History, Training
│   ├── src/components/  # PhaseTimeline, FaultsList, TrendCharts, ProComparison, etc.
│   └── src/lib/      # API client, drill database
├── backend/          # FastAPI + Python ML backend
│   ├── app/          # API layer (routes, schemas, services)
│   └── ml/           # ML modules (pose, analysis, comparison, dispersion)
├── data/             # SQLite DB, model weights, uploads
├── notebooks/        # Research & experimentation
└── scripts/          # Training and data collection scripts
```

## Tech Stack

| Layer | Tech |
|---|---|
| Frontend | React 19, TypeScript, Vite, Tailwind CSS, Recharts |
| Backend API | FastAPI (Python) |
| Pose Estimation | MediaPipe PoseLandmarker (heavy model) |
| ML Framework | PyTorch |
| Database | SQLite |
| Swing Matching | Metrics-based similarity + learned embeddings (WIP) |

## Getting Started

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs available at `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

App available at `http://localhost:5173`

### Environment Variables

Copy `.env.example` to `.env` and fill in values:

```bash
cp backend/.env.example backend/.env
```

## Development Roadmap

- [x] Pose keypoint extraction pipeline
- [x] Swing phase segmentation (8 phases)
- [x] Biomechanical metrics computation
- [x] Fault detection (8 fault types)
- [x] Per-phase scoring with letter grades
- [x] Practice & drills page with targeted exercises
- [x] Pro swing database + similarity matching
- [x] Progress tracking with trend charts
- [x] Full-stack web app (React + FastAPI)
- [ ] Swing embedding model (Transformer-based)
- [ ] Side-by-side swing comparison
- [ ] Club type tagging + per-club thresholds
- [ ] Shot dispersion model
- [ ] React Native mobile app
