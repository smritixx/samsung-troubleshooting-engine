from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import FRONTEND_DIR
from .logging_config import setup_logging
from .pipeline import TroubleshootPipeline

setup_logging()

app = FastAPI(
    title="Smart Guided Troubleshooting Engine",
    version="2.0.0",
    description="Samsung PRISM Theme 2 reference implementation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = TroubleshootPipeline()


class TroubleshootRequest(BaseModel):
    query: str
    sis_response: Optional[Any] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/troubleshoot")
def troubleshoot(req: TroubleshootRequest):
    if not req.query or not req.query.strip():
        return JSONResponse(status_code=400, content={
            "detail": "query must be a non-empty string",
        })
    return pipeline.run(req.query, req.sis_response)


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def root():
        return FileResponse(str(FRONTEND_DIR / "index.html"))