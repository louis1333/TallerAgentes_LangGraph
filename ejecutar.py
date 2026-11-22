"""
ejecutar.py — Punto de entrada único del proyecto.

Todo queda dentro de esta carpeta: el entorno virtual (.venv), el código, el frontend
del curso y la evidencia. Lo único que se instala aparte es Python y Ollama.
Este archivo solo usa la librería estándar, así que funciona antes de instalar nada.

Uso:  python ejecutar.py <comando>

  instalar   crea .venv, instala requirements.txt y descarga el modelo en Ollama si falta
  consola    chat por consola con el planeador (muestra la traza de los agentes).
             Con preguntas entre comillas las hace en orden y termina:
             python ejecutar.py consola "¿Cómo estará el clima en Suesca el sábado?" "¿Y el domingo?"
  web        API en http://localhost:8081 y frontend en http://localhost:4200 (abre el navegador;
             con --sin-navegador no lo abre)
  grafo      imprime el grafo en formato Mermaid
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import venv
import webbrowser
from pathlib import Path

from config import MODELO, OLLAMA_URL

RAIZ = Path(__file__).resolve().parent
VENV = RAIZ / ".venv"
PYTHON = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
PUERTO_API = 8081        # el frontend (frontend/index.html) llama a http://localhost:8081
PUERTO_FRONTEND = 4200


def _entorno() -> dict:
    return {**os.environ, "PYTHONUTF8": "1"}   # tildes bien en la consola de Windows


def _correr(*argumentos: str) -> int:
    """Ejecuta un comando con el Python del .venv, desde la raíz del proyecto."""
    return subprocess.run([str(PYTHON), *argumentos], cwd=RAIZ, env=_entorno()).returncode


def revisar_ollama(descargar: bool = False) -> bool:
    """Comprueba que Ollama responde y que el modelo está descargado."""
    try:
        with urllib.request.urlopen(OLLAMA_URL.rstrip("/") + "/api/tags", timeout=5) as r:
            modelos = [m["name"] for m in json.load(r)["models"]]
    except Exception:
        print(f"No encuentro Ollama en {OLLAMA_URL}. Abran la aplicación Ollama y repitan.")
        return False
    if MODELO in modelos:
        return True
    if descargar and shutil.which("ollama"):
        print(f"Descargando el modelo {MODELO} (~1,9 GB, solo la primera vez)...")
        return subprocess.run(["ollama", "pull", MODELO]).returncode == 0
    print(f"Falta el modelo en Ollama. Ejecuten:  ollama pull {MODELO}")
    return False


def instalar() -> None:
    if not PYTHON.exists():
        print("Creando el entorno virtual en .venv ...")
        venv.create(VENV, with_pip=True)
    print("Instalando dependencias de requirements.txt ...")
    subprocess.run([str(PYTHON), "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)
    subprocess.run([str(PYTHON), "-m", "pip", "install", "-q", "-r", str(RAIZ / "requirements.txt")], check=True)
    if revisar_ollama(descargar=True):
        print("Listo. Prueben:  python ejecutar.py consola   o   python ejecutar.py web")


def _asegurar_entorno() -> None:
    if not PYTHON.exists():   # primera vez: instala solo
        instalar()


def web() -> None:
    """Levanta la API y el frontend juntos. Ctrl+C detiene los dos."""
    if not revisar_ollama():
        return
    api = subprocess.Popen([str(PYTHON), "-m", "uvicorn", "api:app", "--port", str(PUERTO_API)],
                           cwd=RAIZ, env=_entorno())
    frontend = subprocess.Popen([str(PYTHON), "-m", "http.server", str(PUERTO_FRONTEND),
                                 "--bind", "127.0.0.1", "--directory", str(RAIZ / "frontend")],
                                cwd=RAIZ, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(30):   # espera a que la API conteste (127.0.0.1: en Windows "localhost"
            try:              # prueba primero IPv6 y eso tarda ~2 s en fallar)
                urllib.request.urlopen(f"http://127.0.0.1:{PUERTO_API}/docs", timeout=10)
                break
            except Exception:
                time.sleep(1)
        print(f"\n  Frontend:  http://localhost:{PUERTO_FRONTEND}"
              f"\n  API:       http://localhost:{PUERTO_API}/docs"
              "\n  La traza de los agentes aparece en esta terminal. Ctrl+C para terminar.\n")
        if "--sin-navegador" not in sys.argv:
            webbrowser.open(f"http://localhost:{PUERTO_FRONTEND}")
        api.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for proceso in (api, frontend):
            proceso.terminate()


COMANDOS = {
    "instalar": instalar,
    "consola": lambda: _correr("main.py", *sys.argv[2:]),
    "web": web,
    "grafo": lambda: _correr("main.py", "--grafo"),
}

if __name__ == "__main__":
    comando = sys.argv[1] if len(sys.argv) > 1 else ""
    if comando not in COMANDOS:
        print(__doc__)
        sys.exit(1)
    if comando != "instalar":
        _asegurar_entorno()
    COMANDOS[comando]()
