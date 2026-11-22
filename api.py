"""
api.py — Expone el planeador con el MISMO contrato REST del proyecto pruebachat.

Así el frontend del curso (copiado sin cambios en frontend/index.html) funciona
contra este backend en Python, igual que contra el backend en Spring AI.

Uso:
    python ejecutar.py web          (API + frontend)
    uvicorn api:app --port 8081     (solo la API)

El enunciado usa el puerto 8080 (el del frontend original del curso). Nuestro
frontend de pruebachat quedó apuntando a 8081 desde el taller de RAG, por eso
levantamos la API ahí. Con el frontend original, usen --port 8080.
"""
import json
import logging
import time
import urllib.request
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import OLLAMA_URL
from grafo import planear

app = FastAPI(title="Planeador de fin de semana - LangGraph")

# En Windows, al recargar la página el navegador corta conexiones que estaban abiertas y asyncio
# imprime "ConnectionResetError: [WinError 10054]" con una traza, aunque no afecta nada.
logging.getLogger("asyncio").addFilter(
    lambda registro: not (registro.exc_info and isinstance(registro.exc_info[1], ConnectionResetError)))

# Mismo permiso CORS que @CrossOrigin del ChatController en Spring.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    sessionId: str | None = None   # RETO (memoria): se usa como thread_id de la conversación


@app.post("/api/v1/chat")
def chat(req: ChatRequest):
    inicio = time.perf_counter()
    try:
        respuesta = planear(req.question, req.sessionId)
    except Exception as error:   # p. ej. Ollama apagado: se responde 502 como el backend de Spring
        print(f"   [api] error: {error}")
        respuesta = ""
    elapsed_ms = int((time.perf_counter() - inicio) * 1000)
    status = "ok" if respuesta and respuesta.strip() else "error"
    cuerpo = {"requestId": str(uuid.uuid4()), "answer": respuesta or "",
              "elapsedMs": elapsed_ms, "status": status}
    return JSONResponse(cuerpo, status_code=200 if status == "ok" else 502)


@app.get("/api/v1/plantillas")
def plantillas():
    """El frontend del curso pide las plantillas de sistema (feature 005 de pruebachat) para su
    selector. Aquí hay un solo comportamiento: el planeador multiagente."""
    return {"plantillas": ["planeador"], "porDefecto": "planeador"}


@app.get("/health/llm")
def health_llm():
    """Equivalente a LlmHealthController: consulta /api/tags de Ollama."""
    endpoint = OLLAMA_URL.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(endpoint, timeout=5) as r:
            return {"ok": True, "endpoint": endpoint, "result": json.load(r)}
    except Exception as error:
        return JSONResponse({"ok": False, "error": str(error)}, status_code=502)
