"""
grafo.py — El sistema multiagente como un grafo de LangGraph.

      START ──► reformulador (RETO: memoria)
                       │
                  ┌────▼─────────┐
                  │  supervisor  │ ◄──────────────┐
                  └──────┬───────┘                │
          (arista condicional: ¿quién sigue?)     │
         ┌───────────┬───┴────────┬──────────┐    │
         ▼           ▼            ▼          ▼    │
       clima    presupuesto    reservas   sintetizador ──► END
         │           │            │               │
         └───────────┴────────────┴───────────────┘
                 (vuelven al supervisor)

AJUSTES respecto al código del enunciado (la bitácora muestra cada corrida):
  - Supervisor: en la primera visita de cada pregunta el LLM CLASIFICA qué
    especialistas necesita (salida estructurada con true/false) y el CÓDIGO aplica
    el orden (reservas siempre de último), la no repetición y MAX_PASOS. Con
    qwen2.5:3b, decidir "quién sigue" paso a paso fallaba: repetía agentes, se
    saltaba otros o reservaba antes de revisar el dinero.
  - Límites de autonomía para la única acción con efectos (crear una reserva): el
    agente de reservas solo entra si el usuario habló de reservar, y si la reserva era
    condicional y los datos de las tools dicen que la condición no se cumple, el
    código no deja pasar a reservas.
  - Los resultados llevan los datos exactos de las tools, no solo el resumen del agente.
  - Cada especialista ve solo lo que necesita: la pregunta y su tarea; reservas, además,
    los resultados de los otros (para revisar las condiciones).
  - El sintetizador copia las cifras de las tools y, sin resultados, responde un texto fijo.

RETO ELEGIDO (Parte 7): memoria conversacional.
  - El grafo se compila con un CHECKPOINTER (InMemorySaver): al final de cada
    paso LangGraph guarda una copia del estado, separada por thread_id.
    Un thread_id = una conversación.
  - El estado tiene un campo "historial" con las preguntas del usuario y las
    respuestas del planeador. Así el usuario puede preguntar "¿y el domingo?"
    sin repetir el destino.
  - Nodo "reformulador": si hay conversación previa, REESCRIBE la pregunta para que
    se entienda sola ("¿Y el domingo?" -> "¿Cómo estará el clima en Suesca el
    domingo?"). Desde ahí el grafo trabaja igual que en un primer turno. En el
    primer turno no hace nada (no llama al modelo).
  - En cada turno nuevo, "resultados", "consultados" y "plan" se reinician: el
    límite de autonomía (MAX_PASOS) es por pregunta.
"""
import operator
import uuid
from typing import Annotated

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Overwrite
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from agentes import agente_clima, agente_presupuesto, agente_reservas, crear_modelo
from config import MAX_HISTORIAL, MAX_PASOS
from datos import normalizar

# Orden en que se llaman los especialistas. "reservas" va de último para que, si la
# reserva está condicionada (clima, dinero), esos resultados ya existan cuando reserve.
AGENTES = ("clima", "presupuesto", "reservas")


# ----------------------------------------------------------------------------
# 1. ESTADO: la "memoria compartida" que viaja por todos los nodos.
#    Annotated[list, operator.add] = REDUCER: lo que devuelve un nodo se
#    AGREGA a la lista en lugar de reemplazarla.
# ----------------------------------------------------------------------------
class EstadoPlaneador(TypedDict):
    pregunta: str
    resultados: Annotated[list[str], operator.add]
    consultados: Annotated[list[str], operator.add]
    siguiente: str
    respuesta_final: str
    # Especialistas que necesita la pregunta, en orden. None = el supervisor aún no decide.
    plan: list[str] | None
    # Condiciones que puso el usuario para reservar: "dinero" y/o "lluvia".
    condiciones: list[str]
    # RETO (memoria): mensajes de la conversación. add_messages es el reducer de
    # LangGraph para mensajes: agrega los nuevos al final de la lista.
    historial: Annotated[list[AnyMessage], add_messages]


# ----------------------------------------------------------------------------
# 2. SALIDA ESTRUCTURADA del supervisor: en vez de texto libre, el modelo
#    debe devolver un objeto con estos campos (Ollama lo garantiza con un
#    JSON Schema). Así la decisión de enrutamiento es confiable.
# ----------------------------------------------------------------------------
class Decision(BaseModel):
    clima: bool = Field(description="true si pregunta por el clima, la lluvia o el pronóstico, "
                                    "o si pone una condición de clima ('si no llueve'). "
                                    "false si no menciona nada del tiempo.")
    presupuesto: bool = Field(description="true si pregunta por costos o precios, o si pone una "
                                          "condición de dinero ('si nos alcanza').")
    reservas: bool = Field(description="true solo si pide reservar o pregunta por cupos.")
    reserva_si_alcanza: bool = Field(description="true si pide reservar SOLO si les alcanza el dinero.")
    reserva_si_no_llueve: bool = Field(description="true si pide reservar SOLO si no llueve o hace buen tiempo.")
    motivo: str = Field(description="Explicación breve de la decisión.")


PROMPT_SUPERVISOR = """Eres el SUPERVISOR de un planeador de salidas de fin de semana cerca de Bogotá.
Coordinas a tres agentes especialistas:
- clima: pronóstico del tiempo de un destino para el sábado o el domingo.
- presupuesto: costo del plan para un grupo y si el dinero alcanza.
- reservas: cupos disponibles y creación de reservas.

Lee la pregunta del usuario y marca qué especialistas necesita (true o false).
Reglas:
1. Marca solo los especialistas que la pregunta realmente necesita.
2. Si el usuario quiere reservar con una condición (clima, dinero), marca también los especialistas
   de esa condición: el sistema los consulta antes que a reservas.
3. Si la pregunta no trata de salidas de fin de semana (deportes, noticias, etc.), todo es false.
4. Si hay conversación anterior, úsala para entender preguntas cortas como "¿y el domingo?"
   (mismo destino y mismo tema que antes).

Ejemplos:
- "¿Va a llover en Chicaque el sábado?" -> clima.
- "Somos 5, ¿cuánto cuesta ir a Guatavita?" -> presupuesto.
- "¿Qué clima hará en Zipaquirá y cuánto cuesta para 2?" -> clima y presupuesto.
- "Si hace buen tiempo y nos alcanza con 400000, reserva Chicaque a nombre de Ana"
  -> clima, presupuesto y reservas; reserva_si_no_llueve y reserva_si_alcanza.
- "Si nos alcanza con 100000, reserva Guatavita a nombre de Ana" -> presupuesto y reservas; reserva_si_alcanza.
- "Reserva Zipaquirá el domingo para 2 a nombre de Ana" -> reservas.
- "¿Cuál es la capital de Francia?" -> ninguno."""

PROMPT_SINTETIZADOR = """Eres el asistente final de un planeador de salidas de fin de semana.
Escribe la respuesta para el usuario en español, con tono amable y en máximo 4 frases, usando SOLO
la información de "Resultados de otros agentes".
- Las cuentas ya están hechas en "Datos de las herramientas": copia esas cifras tal cual, sin recalcular.
- Incluye las cifras de cada agente que trabajó: temperaturas y % de lluvia, costo total y cuánto sobra o falta.
- Menciona una reserva solo si el agente de reservas la creó, con su código exacto; si el usuario pidió
  reservar y no se hizo, di por qué.
- Si hay conversación anterior, responde solo la pregunta actual."""

# Sin resultados (pregunta fuera de tema) no hay nada que sintetizar: respuesta fija, sin llamar
# al modelo. En la primera versión el modelo respondía la pregunta con su propio conocimiento.
RESPUESTA_FUERA_DE_TEMA = ("Solo puedo ayudarte a planear salidas de fin de semana cerca de Bogotá: "
                           "clima, presupuesto y reservas para Guatavita, Zipaquirá, Villa de Leyva, "
                           "Suesca y Chicaque. ¿Qué plan tienen en mente?")

# RETO (memoria): una sola tarea y salida en texto plano. Se probó pedirle al supervisor
# la pregunta reescrita dentro de su JSON y qwen2.5:3b escribía cosas como "clima".
PROMPT_REFORMULADOR = """Reescribe la última pregunta del usuario para que se entienda sola, sin la conversación.
- Agrega el destino, el día, el número de personas y el presupuesto que el usuario ya dijo antes
  y que la última pregunta da por sabidos.
- No respondas la pregunta. Escribe solo la pregunta reescrita, en una línea.

Ejemplo 1
Conversación:
Usuario: ¿Va a llover en Chicaque el sábado?
Planeador: En Chicaque el sábado habrá neblina en la mañana, con 50 % de lluvia.
Última pregunta: ¿y el domingo?
Pregunta reescrita: ¿Cómo estará el clima en Chicaque el domingo?

Ejemplo 2
Conversación:
Usuario: Somos 3 y queremos ir a Zipaquirá. ¿Cuánto cuesta?
Planeador: El costo total para 3 personas es $300.000.
Última pregunta: ¿Y si somos 5?
Pregunta reescrita: Somos 5 y queremos ir a Zipaquirá. ¿Cuánto cuesta?"""

modelo_supervisor = crear_modelo().with_structured_output(Decision)
modelo_reformulador = crear_modelo()
modelo_sintetizador = crear_modelo()


def _conversacion_previa(estado: EstadoPlaneador) -> str:
    """RETO (memoria): los últimos turnos de la conversación, sin la pregunta actual."""
    previos = estado.get("historial", [])[:-1][-MAX_HISTORIAL:]   # el último es la pregunta actual
    return "\n".join(f"{'Usuario' if m.type == 'human' else 'Planeador'}: {m.content}" for m in previos)


def _contexto(estado: EstadoPlaneador) -> str:
    """Texto con la pregunta y lo que ya respondieron los agentes."""
    previos = "\n".join(estado["resultados"]) or "(todavía no hay resultados)"
    return f"Pregunta del usuario: {estado['pregunta']}\n\nResultados de otros agentes:\n{previos}"


# ----------------------------------------------------------------------------
# 3. NODOS: funciones que reciben el estado y devuelven SOLO lo que cambian.
# ----------------------------------------------------------------------------
def nodo_reformulador(estado: EstadoPlaneador) -> dict:
    """RETO (memoria): convierte "¿Y el domingo?" en una pregunta que se entiende sola."""
    conversacion = _conversacion_previa(estado)
    if not conversacion:                         # primer turno: nada que resolver
        return {}
    mensaje = f"Conversación:\n{conversacion}\nÚltima pregunta: {estado['pregunta']}\nPregunta reescrita:"
    try:
        texto = modelo_reformulador.invoke([("system", PROMPT_REFORMULADOR), ("human", mensaje)]).content
    except Exception as error:
        print(f"   [reformulador] error ({error}); se usa la pregunta tal cual")
        return {}
    completa = texto.strip().removeprefix("Pregunta reescrita:").strip().strip('"').splitlines()[0]
    print(f"   [reformulador] {estado['pregunta']} -> {completa}")
    return {"pregunta": completa} if completa else {}


def _texto_del_usuario(estado: EstadoPlaneador) -> str:
    """Lo que el usuario escribió en este turno (la pregunta puede venir reescrita)."""
    historial = estado.get("historial") or []
    return historial[-1].content if historial else estado["pregunta"]


def _decidir_plan(estado: EstadoPlaneador) -> dict:
    """El LLM clasifica la pregunta; el código convierte esa clasificación en un plan ordenado."""
    try:
        d = modelo_supervisor.invoke([("system", PROMPT_SUPERVISOR), ("human", _contexto(estado))])
    except Exception as error:                   # el modelo no devolvió un JSON válido
        print(f"   [supervisor] decisión inválida ({error}); se pasa a la respuesta final")
        return {"plan": [], "condiciones": []}
    # Límite de autonomía: reservar es la única acción con efectos, así que "reservas" solo
    # entra al plan si el usuario ESCRIBIÓ sobre reservar o cupos en este turno (no la versión
    # reescrita). En una prueba sin contexto ("¿Y el domingo?") el modelo marcó reservas y se
    # inventó la reserva.
    pide_reserva = d.reservas and any(p in normalizar(_texto_del_usuario(estado)) for p in ("reserv", "cupo"))
    condiciones = (["dinero"] * d.reserva_si_alcanza + ["lluvia"] * d.reserva_si_no_llueve
                   if pide_reserva else [])
    necesita = {"clima": d.clima or "lluvia" in condiciones,              # una condición implica
                "presupuesto": d.presupuesto or "dinero" in condiciones,  # consultar a ese agente
                "reservas": pide_reserva}
    plan = [agente for agente in AGENTES if necesita[agente]]       # orden fijo: reservas al final
    print(f"   [supervisor] plan: {' -> '.join(plan) or 'ninguno'}"
          f"{'; reservar solo si: ' + ', '.join(condiciones) if condiciones else ''} ({d.motivo})")
    return {"plan": plan, "condiciones": condiciones}


def _condicion_incumplida(estado: EstadoPlaneador) -> str | None:
    """LÍMITE DE AUTONOMÍA para crear reservas (la única acción con efectos).

    Revisa las frases EXACTAS que producen nuestras tools, no la opinión del modelo.
    En una corrida el agente de reservas reservó aunque el presupuesto decía "No alcanza".
    """
    datos = "\n".join(estado["resultados"])
    if "dinero" in estado["condiciones"] and "No alcanza:" in datos:
        return "no alcanza el dinero"
    if "lluvia" in estado["condiciones"] and "(mucha lluvia)" in datos:
        return "el pronóstico es de mucha lluvia"
    return None


def nodo_supervisor(estado: EstadoPlaneador) -> dict:
    consultados = estado["consultados"]
    if len(consultados) >= MAX_PASOS:            # límite de autonomía
        return {"siguiente": "fin"}

    cambios = {}
    if estado.get("plan") is None:               # primera visita de esta pregunta: decide el LLM
        cambios = _decidir_plan(estado)
        estado = {**estado, **cambios}

    pendientes = [agente for agente in estado["plan"] if agente not in consultados]   # nunca repite
    siguiente = pendientes[0] if pendientes else "fin"

    if siguiente == "reservas" and (falla := _condicion_incumplida(estado)):
        print(f"   [supervisor] no llama a reservas: la reserva era condicional y {falla}")
        return {**cambios, "siguiente": "fin",
                "resultados": [f"[supervisor] No se hizo la reserva: era condicional y {falla}."]}

    print(f"   [supervisor] -> {siguiente}")
    return {**cambios, "siguiente": siguiente}


# Lo que le toca a cada especialista. El supervisor se lo entrega junto con la pregunta: con
# solo la pregunta "¿Cómo estará el clima y cuánto nos cuesta?" el agente de presupuesto
# respondía que no era de su tema y no usaba ninguna tool.
TAREAS = {
    "clima": "el pronóstico del clima del destino y el día que pregunta.",
    "presupuesto": "cuánto cuesta el plan y, si el usuario dijo su presupuesto, si les alcanza el dinero.",
    "reservas": "consultar cupos y crear la reserva si el usuario la pidió y se cumplen sus condiciones.",
}


def crear_nodo_especialista(nombre: str, agente, ve_resultados: bool = False):
    """Fábrica de nodos: envuelve un agente (create_agent) como nodo del grafo.

    ve_resultados: si el agente recibe lo que respondieron los otros agentes. Solo reservas
    lo necesita (para revisar las condiciones). Clima y presupuesto reciben solo la pregunta
    y su tarea: en una prueba el de clima escribió "El costo será $1500" y el de presupuesto
    tomó esa cifra como el presupuesto del grupo.
    """
    def nodo(estado: EstadoPlaneador) -> dict:
        partes = [f"Pregunta del usuario: {estado['pregunta']}", f"Tu parte: {TAREAS[nombre]}"]
        if ve_resultados:
            partes.append("Resultados de otros agentes:\n" +
                          ("\n".join(estado["resultados"]) or "(todavía no hay resultados)"))
        resultado = agente.invoke({"messages": [{"role": "user", "content": "\n\n".join(partes)}]})
        # Traza de las tools que usó el agente dentro de su ciclo ReAct.
        datos_tools = []
        for m in resultado["messages"]:
            for llamada in getattr(m, "tool_calls", None) or []:
                print(f"      {nombre} pide {llamada['name']}({llamada['args']})")
            if m.type == "tool":
                print(f"      {nombre} recibe: {m.content}")
                datos_tools.append(m.content)
        texto = resultado["messages"][-1].content
        print(f"   [{nombre}] {texto}")
        # Además del resumen del agente se guardan los datos EXACTOS de las tools: en la
        # primera corrida un resumen omitió el % de lluvia y el sintetizador inventó cifras.
        registro = f"[{nombre}] {texto}"
        if datos_tools:
            registro += "\n   Datos de las herramientas: " + " | ".join(datos_tools)
        return {"resultados": [registro], "consultados": [nombre]}
    return nodo


def nodo_sintetizador(estado: EstadoPlaneador) -> dict:
    if estado["resultados"]:
        texto = modelo_sintetizador.invoke(
            [("system", PROMPT_SINTETIZADOR), ("human", _contexto(estado))]).content
    else:
        texto = RESPUESTA_FUERA_DE_TEMA
    # RETO (memoria): la respuesta final queda en el historial de la conversación.
    return {"respuesta_final": texto, "historial": [AIMessage(texto)]}


# ----------------------------------------------------------------------------
# 4. ARISTA CONDICIONAL: lee el estado y dice a qué nodo ir.
# ----------------------------------------------------------------------------
def enrutar(estado: EstadoPlaneador) -> str:
    return estado["siguiente"]


# ----------------------------------------------------------------------------
# 5. CONSTRUCCIÓN Y COMPILACIÓN DEL GRAFO
# ----------------------------------------------------------------------------
def construir_grafo():
    grafo = StateGraph(EstadoPlaneador)

    grafo.add_node("reformulador", nodo_reformulador)   # RETO (memoria)
    grafo.add_node("supervisor", nodo_supervisor)
    grafo.add_node("clima", crear_nodo_especialista("clima", agente_clima))
    grafo.add_node("presupuesto", crear_nodo_especialista("presupuesto", agente_presupuesto))
    grafo.add_node("reservas", crear_nodo_especialista("reservas", agente_reservas, ve_resultados=True))
    grafo.add_node("sintetizador", nodo_sintetizador)

    grafo.add_edge(START, "reformulador")
    grafo.add_edge("reformulador", "supervisor")
    grafo.add_conditional_edges("supervisor", enrutar, {
        "clima": "clima",
        "presupuesto": "presupuesto",
        "reservas": "reservas",
        "fin": "sintetizador",
    })
    for especialista in AGENTES:
        grafo.add_edge(especialista, "supervisor")
    grafo.add_edge("sintetizador", END)

    # RETO (memoria): el checkpointer guarda el estado de cada conversación (thread_id).
    # InMemorySaver lo guarda en la RAM del proceso: se pierde al reiniciar el programa.
    return grafo.compile(checkpointer=InMemorySaver())


def estado_inicial(pregunta: str) -> dict:
    # RETO (memoria): con checkpointer el estado del turno anterior sigue ahí. Overwrite
    # salta el reducer operator.add y deja resultados y consultados vacíos para la nueva
    # pregunta; sin esto el supervisor creería que ya consultó a los agentes del turno anterior.
    return {"pregunta": pregunta, "resultados": Overwrite([]), "consultados": Overwrite([]),
            "plan": None, "condiciones": [], "historial": [HumanMessage(pregunta)]}


app = construir_grafo()


def planear(pregunta: str, sesion: str | None = None) -> str:
    """Punto de entrada usado por main.py y api.py.

    sesion: identificador de la conversación (thread_id). Las preguntas con la misma
    sesión comparten memoria; sin sesión, cada pregunta es una conversación nueva.
    """
    config = {"configurable": {"thread_id": sesion or str(uuid.uuid4())}}
    estado_final = app.invoke(estado_inicial(pregunta), config)
    return estado_final["respuesta_final"]
