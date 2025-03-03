"""Entry point of the application"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_pretty_errors import PrettyErrorsMiddleware  # type: ignore

from routers.assistant import router as assistant_router
from routers.items import router as items_router
from routers.stt import router as stt_router
from routers.tts import router as tts_router
from routers.users import router as users_router
from routers.languages import router as languages_router
from routers.situations import router as situations_router
from routers.evaluator import router as evaluator_router
from routers.learning_session import router as learning_session_router

from dotenv import load_dotenv

load_dotenv(".env", override=True)

app = FastAPI()

# middlewares
app.add_middleware(PrettyErrorsMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routes
@app.get("/")
def health_check() -> dict:
    """health check of the server."""
    return {"status": "ok"}


app.include_router(items_router)
app.include_router(stt_router)
app.include_router(tts_router)
app.include_router(assistant_router)
app.include_router(users_router)
app.include_router(languages_router)
app.include_router(situations_router)
app.include_router(evaluator_router)
app.include_router(learning_session_router)

# Allow uvicorn to be executed using `uv run`
if __name__ == "__main__":
    uvicorn.run("start_server:app", host="0.0.0.0", port=8000, reload=True)
