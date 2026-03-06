from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import get_settings
from app.api.routes import analysis, comparison, dispersion

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="AI-powered golf swing analysis, pro comparison, and shot dispersion.",
    version="0.1.0",
    debug=settings.debug,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["Swing Analysis"])
app.include_router(comparison.router, prefix="/api/v1/comparison", tags=["Pro Comparison"])
app.include_router(dispersion.router, prefix="/api/v1/dispersion", tags=["Shot Dispersion"])


@app.get("/health")
def health_check():
    return {"status": "ok", "app": settings.app_name}
