"""Entry point of the application"""

import pretty_errors  # type: ignore
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.assistant import router as assistant_router
from routers.items import router as items_router
from routers.stt import router as stt_router
from routers.tts import router as tts_router
from routers.user import router as user_router

pretty_errors.replace_stderr()

app = FastAPI()


@app.get("/health")
def health_check() -> dict:
    """health check of the server."""
    return {"status": "ok"}


app.include_router(items_router)
app.include_router(stt_router)
app.include_router(tts_router)
app.include_router(assistant_router)
app.include_router(user_router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
