# Golf Lab

An AI-powered golf swing analysis platform. Upload a swing video and get instant coaching feedback, pro swing comparisons, and (coming soon) shot dispersion analytics.

## Features

- **Swing Analysis** — Pose estimation + AI coaching feedback on mechanics, positions, and sequencing
- **Pro Comparison** — Match your swing to similar PGA/LPGA tour pros using embedding similarity
- **Shot Dispersion** *(coming soon)* — Predict shot shape and dispersion patterns from swing data

## Architecture

```
golf_lab/
├── backend/          # FastAPI + Python ML backend
│   ├── app/          # API layer (routes, schemas, services)
│   └── ml/           # ML modules (pose, analysis, comparison, dispersion)
├── mobile/           # React Native app (coming soon)
├── data/             # Reference data (pro swings, model weights)
├── notebooks/        # Research & experimentation
└── scripts/          # Training and data collection scripts
```

## Tech Stack

| Layer | Tech |
|---|---|
| Backend API | FastAPI (Python) |
| Pose Estimation | MediaPipe / OpenCV |
| ML Framework | PyTorch |
| Swing Matching | DTW + learned embeddings |
| Mobile | React Native (Expo) |

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

### Environment Variables

Copy `.env.example` to `.env` and fill in values:

```bash
cp backend/.env.example backend/.env
```

## Development Roadmap

- [ ] Pose keypoint extraction pipeline
- [ ] Swing phase segmentation (address, backswing, top, downswing, impact, follow-through)
- [ ] Fault detection model
- [ ] Pro swing database + embedding index
- [ ] Swing similarity matching
- [ ] React Native mobile app
- [ ] Shot dispersion model
