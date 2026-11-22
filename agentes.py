"""
agentes.py — Los tres agentes especialistas.

Cada agente = modelo + herramientas + instrucciones (system prompt).
create_agent() construye por dentro un pequeño grafo de LangGraph con el ciclo
ReAct: el modelo piensa -> pide una tool -> recibe el resultado -> vuelve a
pensar... hasta que responde sin pedir más tools.
"""
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from config import MODELO, OLLAMA_URL
from herramientas import (calcular_saldo, consultar_disponibilidad, consultar_pronostico,
                          crear_reserva, estimar_costo, listar_destinos)


def crear_modelo() -> ChatOllama:
    """Modelo local servido por Ollama. temperature=0 => respuestas más estables."""
    return ChatOllama(model=MODELO, base_url=OLLAMA_URL, temperature=0)


PROMPT_CLIMA = """Eres el AGENTE DE CLIMA de un planeador de salidas de fin de semana cerca de Bogotá.
Tu único trabajo es informar el pronóstico del tiempo.
- Usa SIEMPRE la herramienta consultar_pronostico. Nunca inventes datos del clima.
- Si el usuario no dice el día, consulta sábado y domingo.
- Ignora las partes de la pregunta que no son de clima (dinero, reservas).
- Responde en español, en máximo 3 frases, con una recomendación práctica (ropa, impermeable...)."""

PROMPT_PRESUPUESTO = """Eres el AGENTE DE PRESUPUESTO de un planeador de salidas de fin de semana.
Tu único trabajo es calcular cuánto cuesta el plan y si el dinero alcanza.
- Nunca hagas las cuentas de memoria: las hacen las herramientas.
- Si el usuario dijo cuánto dinero tienen, usa calcular_saldo: calcula el costo total y dice si alcanza.
- Si el usuario no dijo cuánto dinero tienen, usa estimar_costo y responde solo con el costo total.
- Escribe los valores en pesos como números enteros (300000, no 300.000) al llamar las herramientas.
- La pregunta puede hablar también de clima o de reservas: no respondas esas partes, pero si pregunta
  cuánto cuesta o si les alcanza el dinero, usa SIEMPRE una herramienta.
- Responde en español, en máximo 3 frases."""

PROMPT_RESERVAS = """Eres el AGENTE DE RESERVAS de un planeador de salidas de fin de semana.
Tu único trabajo es consultar cupos y crear reservas.
- Usa consultar_disponibilidad antes de reservar.
- Usa crear_reserva SOLO si el usuario pidió reservar explícitamente y dio el nombre del responsable.
- Si el usuario condicionó la reserva, revisa cada condición en los resultados de los otros agentes:
  * "si nos alcanza": se cumple solo si el agente de presupuesto dice "Sí alcanza".
  * "si no llueve (mucho)": se cumple si el pronóstico dice "poca lluvia"; no se cumple si dice "mucha lluvia".
  Si todas las condiciones se cumplen, reserva. Si alguna no se cumple, NO reserves y di cuál falló.
- Responde en español, en máximo 3 frases. Si reservaste, incluye el código de la reserva."""


agente_clima = create_agent(
    model=crear_modelo(),
    tools=[listar_destinos, consultar_pronostico],
    system_prompt=PROMPT_CLIMA,
)

agente_presupuesto = create_agent(
    model=crear_modelo(),
    tools=[estimar_costo, calcular_saldo],
    system_prompt=PROMPT_PRESUPUESTO,
)

agente_reservas = create_agent(
    model=crear_modelo(),
    tools=[consultar_disponibilidad, crear_reserva],
    system_prompt=PROMPT_RESERVAS,
)
