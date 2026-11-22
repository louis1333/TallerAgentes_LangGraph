# Planeador de fin de semana multiagente (LangGraph + Ollama)

Taller del Módulo 3 (Arquitecturas agénticas con LangGraph). Un asistente que responde
preguntas sobre salidas de fin de semana cerca de Bogotá (clima, costo y reservas) con varios
agentes coordinados por un supervisor. El modelo corre en el computador con Ollama
(`qwen2.5:3b`) y los datos son ficticios. Reto elegido (Parte 7): **memoria conversacional**.

**Integrantes:** Daniel Fernando Angarita, Louis Alejandro Beltrán Castaño y Cristian Santiago Cruz.

El proyecto es autocontenido: el código, el frontend del curso, el entorno virtual (`.venv`)
y la evidencia viven en esta carpeta. Fuera de ella solo se necesita Python y Ollama.

## Cómo funciona

Cada pregunta recorre un grafo de LangGraph. Todos los pasos leen y escriben un mismo
**estado** (la pregunta, los resultados de los agentes, la respuesta final…):

```
pregunta → reformulador → supervisor → clima / presupuesto / reservas → supervisor → … → sintetizador → respuesta
```

1. **Reformulador** (el reto): si es la segunda pregunta de una conversación, la completa con lo
   que ya se habló ("¿Y el domingo?" → "¿Cómo estará el clima en Suesca el domingo?"). En la
   primera pregunta no hace nada.
2. **Supervisor**: el modelo lee la pregunta y marca qué especialistas necesita. El código decide
   el orden (reservas siempre de último), que nadie se repita y el máximo de 3 (`MAX_PASOS`).
3. **Especialistas** (clima, presupuesto, reservas): cada uno es un agente con sus propias tools.
   El modelo solo decide qué tool pedir y con qué datos; las cifras las calculan las tools
   (`herramientas.py`) con los datos de `datos.py`, nunca el modelo.
4. **Guardas**: crear una reserva es la única acción con efectos, así que antes de llamar al agente
   de reservas el código revisa que el usuario la haya pedido y que se cumplan sus condiciones
   ("Sí alcanza", "poca lluvia").
5. **Sintetizador**: redacta la respuesta final usando solo los datos que trajeron las tools.

Ejemplo real (pregunta 4 de la Parte 5, traza recortada):

```
Tú: Somos 3 y tenemos 250000. Si no llueve mucho y nos alcanza, reserva Suesca para el sábado a nombre de Laura.
   [supervisor] plan: clima -> presupuesto -> reservas; reservar solo si: dinero, lluvia
      clima pide consultar_pronostico({'destino': 'Suesca', 'dia': 'sabado'})
      clima recibe: ... probabilidad de lluvia 15 % (poca lluvia).
      presupuesto pide calcular_saldo({'destino': 'Suesca', 'personas': 3, 'presupuesto': 250000})
      presupuesto recibe: ... Sí alcanza: presupuesto $250.000 - costo $210.000 = sobran $40.000.
      reservas pide crear_reserva({'destino': 'Suesca', 'dia': 'sábado', 'personas': 3, 'responsable': 'Laura'})
      reservas recibe: Reserva confirmada RES-001: Rocas de Suesca, sabado, 3 personas, a nombre de Laura.
Planeador: La reserva ... ha sido confirmada con el código RES-001. El costo total para 3 personas es $210.000 ...
```

## Requisitos

- Python 3.11 o superior (probado con 3.14.7).
- [Ollama](https://ollama.com) instalado y abierto. El modelo lo descarga `python ejecutar.py instalar`.

## Ejecutar

Todo se hace desde esta carpeta con `ejecutar.py`:

```bash
python ejecutar.py instalar     # una sola vez: crea .venv, instala dependencias y descarga qwen2.5:3b
python ejecutar.py web          # abre el frontend en http://localhost:4200 (API en el puerto 8081)
python ejecutar.py consola      # chat por consola con la traza de los agentes ('nueva', 'salir')
python ejecutar.py grafo        # el grafo en Mermaid (pegar en https://mermaid.live)
```

Con `web` las preguntas se hacen en el navegador y la traza de los agentes (qué decidió el
supervisor, qué tool pidió cada agente y qué le respondió) aparece en la terminal. A `consola`
también se le pueden pasar preguntas entre comillas; las hace en orden, en la misma conversación:
`python ejecutar.py consola "¿Cómo estará el clima en Suesca el sábado?" "¿Y el domingo?"`.

Preguntas para probar:

- ¿Cómo estará el clima en Guatavita el domingo?
- Somos 4 amigos con 350000 pesos. ¿Nos alcanza para ir a Zipaquirá?
- Somos 3 y tenemos 250000. Si no llueve mucho y nos alcanza, reserva Suesca para el sábado a nombre de Laura.
- ¿Cómo estará el clima en Suesca el sábado? → y después: ¿Y el domingo?

En un portátil sin GPU cada respuesta tarda entre 10 s y 1,5 min: es inferencia en CPU con
varias llamadas al modelo por pregunta.

`frontend/index.html` es una copia sin cambios del frontend de pruebachat
(`frontend/src/index.html`), que llama a `POST /api/v1/chat`, `GET /health/llm` y
`GET /api/v1/plantillas` en `http://localhost:8081`. La API (`api.py`) expone ese mismo contrato y
usa el `sessionId` de cada pestaña como `thread_id`: preguntas en la misma pestaña comparten
memoria, y al recargar la página empieza una conversación nueva.

Variables opcionales: `MODELO_OLLAMA` (por defecto `qwen2.5:3b`) y `OLLAMA_URL`
(por defecto `http://localhost:11434`). El resto de la configuración está en `config.py`.

<details>
<summary>Sin <code>ejecutar.py</code> (comandos del enunciado)</summary>

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1                 # Windows (PowerShell); macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:3b

python main.py                             # consola
uvicorn api:app --port 8081                # API (terminal 1)
python -m http.server 4200 --directory frontend   # frontend (terminal 2)
```

El enunciado usa el puerto 8080 para la API; el frontend de nuestro pruebachat quedó apuntando
a 8081 desde el taller de RAG.
</details>

## Ajustes respecto al código del enunciado

Con el código tal cual el enunciado, `qwen2.5:3b` acertó 1 de las 6 preguntas de prueba: el
supervisor repetía el último agente, el sintetizador inventaba cifras y hasta un código de reserva,
y el agente de presupuesto adivinaba el costo total al llamar las dos tools en paralelo. Los
cambios, con la traza de cada corrida en `docs/evidencia/historial/p5_v*`, fueron:

1. Los resultados de cada especialista llevan los **datos exactos de las tools**, no solo su resumen,
   y cada especialista **ve solo lo que necesita**: la pregunta y su tarea ("Tu parte: …"); reservas,
   además, los resultados de los otros (en una prueba el de presupuesto tomó una cifra que escribió
   el de clima como si fuera el presupuesto del grupo).
2. El **supervisor clasifica** una vez qué especialistas necesita la pregunta (salida estructurada
   con `true/false`) y el **código** aplica el orden, la no repetición y `MAX_PASOS`. Decidir
   "quién sigue" paso a paso fallaba con un modelo de 3B.
3. Tools que no obligan al modelo a comparar ni a copiar cifras: `consultar_pronostico` devuelve
   "(poca lluvia)" / "(mucha lluvia)", y `calcular_saldo(destino, personas, presupuesto)` calcula
   el costo ella misma (antes recibía `costo_total` y el modelo lo adivinaba).
4. **Límites de autonomía en código** para crear reservas: el agente de reservas solo entra si el
   usuario habló de reservar, y no entra si la reserva era condicional y las tools dicen
   "No alcanza" o "(mucha lluvia)".
5. Sin resultados (pregunta fuera de tema) la respuesta es un texto fijo, sin llamar al modelo.

El resultado de la versión final, con los pantallazos de cada pregunta hecha desde el navegador,
y las limitaciones que quedan están en la bitácora.

## Estructura

| Archivo | Contenido |
|---|---|
| `ejecutar.py` | Punto de entrada: instalar, web, consola y grafo |
| `config.py` | Modelo, URL de Ollama, límite de autonomía (`MAX_PASOS`) e historial del reto (`MAX_HISTORIAL`) |
| `datos.py` | Destinos, pronóstico y cupos ficticios; normalización de nombres y días |
| `herramientas.py` | Las seis tools (`@tool`) de los tres especialistas |
| `agentes.py` | Los tres agentes: modelo + tools + system prompt (`create_agent`) |
| `grafo.py` | Estado, reformulador, supervisor, especialistas, sintetizador, aristas y checkpointer |
| `main.py` | Chat por consola y exportación del grafo a Mermaid |
| `api.py` | FastAPI con el mismo contrato REST del backend de Spring |
| `frontend/` | Frontend del curso (copia de pruebachat) |
| `docs/` | Bitácora (PDF y Word), diagrama del grafo y evidencia |

## Entregables

- Bitácora: [`docs/Bitacora_Taller_LangGraph.pdf`](docs/Bitacora_Taller_LangGraph.pdf)
  (fuente editable en `docs/Bitacora_Taller_LangGraph.docx`).
- Diagrama del grafo: [`docs/grafo.mmd`](docs/grafo.mmd).
- Evidencia: `docs/evidencia/` tiene los pantallazos de pantalla completa (con la fecha y la hora de
  Windows) de la ejecución en el computador: entorno, tools, grafo y cada pregunta hecha en el
  navegador junto con la traza de la terminal. `docs/evidencia/historial/` guarda las trazas de las
  corridas anteriores, que documentan los fallos y sus soluciones.

Nota: `qwen2.5:3b` se distribuye bajo la Qwen Research License; revisen la licencia del modelo
antes de cualquier uso fuera del curso.
