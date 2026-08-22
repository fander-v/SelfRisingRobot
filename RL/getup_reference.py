"""Trayectorias de referencia ("datos maestros") para cada postura caída.

Este módulo es el puente entre la búsqueda y el aprendizaje. Contiene, para cada
postura, una secuencia corta de waypoints: pares de ángulos objetivo de los dos
servos por los que debe pasar el robot para levantarse.

De dónde salen estos números: `search_all_getup.py` prueba miles de secuencias
candidatas en simulación y va imprimiendo la mejor (`BEST_SEQ`). Esos valores se
copian aquí a mano. Después `pretrain_robo1_from_scripted.py` los usa como
demostraciones de experto para preentrenar la política por imitación, y
`scripted_getup.py` permite verlas ejecutarse en el visor.

Por qué existe este paso intermedio: aprender la maniobra desde cero solo con
exploración es muy lento, porque la recompensa por levantarse casi nunca se
alcanza por azar. Partiendo de una trayectoria que sí funciona, PPO ya no tiene
que descubrirla, solo pulirla y generalizarla.

Todos los ángulos están en radianes y acotados a ±1.55 (~±88.8 grados), que es
el recorrido útil de los servos. El último waypoint es siempre [0, 0]: el robot
termina con los servos en posición neutra, es decir, de pie y erguido.

NOTA DE TRADUCCIÓN: comentarios y docstrings en español añadidos sobre el código
original de HomeMadeGarbage. Los valores numéricos no fueron modificados.
"""

import numpy as np


# Secuencia genérica de reserva. Solo se usa a través de POSE_SEQUENCE_SIGNS,
# como respaldo para posturas que no tengan una secuencia propia más abajo.
GETUP_SEQUENCE = np.array(
    [
        [1.532655, -0.450321],
        [1.432846, 0.075935],
        [-0.570885, 1.072732],
        [-0.713748, 0.631464],
        [0.0, 0.0],
    ],
    dtype=np.float64,
)


# Signos para reflejar la secuencia genérica según el lado hacia el que cayó el
# robot: caer sobre un costado o sobre el otro es el mismo movimiento en espejo.
POSE_SEQUENCE_SIGNS = {
    "roll_pos": np.array([1.0, 1.0], dtype=np.float64),
    "roll_neg": np.array([-1.0, 1.0], dtype=np.float64),
}


def getup_sequence_for_pose(pose):
    """Devuelve la secuencia de waypoints de levantado para una postura dada.

    Args:
        pose: "roll_pos", "roll_neg", "pitch_pos" o "pitch_neg".

    Returns:
        Un array (N, 2) de float64: N waypoints con los ángulos objetivo de
        servo1 y servo2, en radianes y en el orden en que deben recorrerse.

    Nota: cada rama devuelve valores hallados empíricamente con
    `search_all_getup.py`. Las caídas laterales (roll) se resuelven en 3
    waypoints; las frontales y traseras (pitch) necesitan 5, porque el robot
    tiene que balancearse antes de poder empujar.
    """
    if pose == "roll_pos":
        return np.array(
            [
                [1.524535, 1.483570],
                [1.550000, 0.069879],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        )
    if pose == "roll_neg":
        # Espejo de roll_pos, en los topes del recorrido de los servos.
        return np.array(
            [
                [-1.55, -1.55],
                [-1.55, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        )
    if pose == "pitch_pos":
        return np.array(
            [
                [0.930212, 1.488389],
                [1.489351, 1.473381],
                [1.394132, 1.327853],
                [0.224632, 0.997530],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        )
    if pose == "pitch_neg":
        return np.array(
            [
                [0.249033, -1.151603],
                [1.462818, -1.341321],
                [1.042669, 1.015654],
                [0.024410, 1.435544],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        )
    # Respaldo: secuencia genérica reflejada según la postura.
    return GETUP_SEQUENCE * POSE_SEQUENCE_SIGNS[pose]
