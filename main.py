"""
main.py — Chat por consola con el planeador multiagente.

Uso:
    python main.py                          -> conversa con el planeador
    python main.py "pregunta" ["pregunta"]  -> hace esas preguntas en orden, en la misma
                                               conversación, y termina
    python main.py --grafo                  -> imprime el grafo en formato Mermaid

Dentro del chat:
    nueva   -> empieza otra conversación (el planeador olvida lo anterior)
    salir   -> termina
"""
import sys
import time
import uuid

from config import MODELO


def responder(planear, pregunta: str, sesion: str) -> None:
    inicio = time.perf_counter()
    respuesta = planear(pregunta, sesion)
    segundos = time.perf_counter() - inicio
    print(f"\nPlaneador ({segundos:.0f} s):\n{respuesta}")


def main() -> None:
    from grafo import app, planear   # se importa aquí para mostrar primero el mensaje de carga

    if "--grafo" in sys.argv:
        # Pega la salida en https://mermaid.live para ver el dibujo del grafo.
        print(app.get_graph().draw_mermaid())
        return

    sesion = str(uuid.uuid4())   # RETO (memoria): un thread_id por conversación
    print(f"Planeador de fin de semana (modelo: {MODELO}).")

    preguntas = sys.argv[1:]
    if preguntas:                # preguntas pasadas como argumentos: misma conversación
        for pregunta in preguntas:
            print(f"\nTú: {pregunta}")
            responder(planear, pregunta, sesion)
        return

    print("Escribe 'nueva' para empezar otra conversación o 'salir' para terminar.")
    while True:
        pregunta = input("\nTú: ").strip()
        if pregunta.lower() in {"salir", "exit", "q"}:
            break
        if pregunta.lower() == "nueva":
            sesion = str(uuid.uuid4())
            print("(conversación nueva: el planeador ya no recuerda lo anterior)")
            continue
        if pregunta:
            responder(planear, pregunta, sesion)


if __name__ == "__main__":
    print("Cargando agentes...")
    main()
