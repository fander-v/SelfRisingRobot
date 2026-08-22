"""Búsqueda de trayectorias de levantado para cada postura caída.

Este es el primer paso de la cadena: antes de que exista cualquier política
aprendida, hay que encontrar al menos una secuencia de ángulos que consiga
levantar al robot. El script prueba miles de secuencias candidatas en simulación
y se queda con la mejor de cada postura.

La estrategia de búsqueda combina dos fuentes de candidatas:

1. Rejilla sistemática: todas las combinaciones de una lista de ángulos base,
   en tres patrones de secuencia distintos.
2. Muestreo aleatorio: secuencias de 5 waypoints con valores uniformes.

Uso:

    python search_all_getup.py
    python search_all_getup.py --pose pitch_pos --random-count 3000

El bloque `BEST_SEQ` / `REFERENCE_CANDIDATES` que se imprime al final se copia a
mano dentro de `getup_sequence_for_pose()` en `getup_reference.py`.

NOTA DE TRADUCCIÓN: comentarios y docstrings en español añadidos sobre el código
original de HomeMadeGarbage. La lógica no fue modificada.
"""

import argparse

import numpy as np


FALLEN_POSES = ("roll_pos", "roll_neg", "pitch_pos", "pitch_neg")


def expert_target_schedule(env, sequence):
    """Convierte una secuencia de waypoints en objetivos paso a paso de la política.

    Interpola 350 pasos de física entre waypoints, pero solo conserva un objetivo
    de cada `frame_skip`: así el calendario resultante queda en la misma escala
    temporal en la que actúa la política (un valor por llamada a `env.step`).

    Al final añade 150 pasos con objetivo cero, para que el robot tenga tiempo de
    asentarse en la postura erguida y se pueda comprobar si aguanta.
    """
    targets = []
    prev = np.zeros(2, dtype=np.float64)
    for waypoint in sequence:
        for i in range(350):
            t = (i + 1) / 350
            ctrl = (1.0 - t) * prev + t * waypoint
            # Se toma la última submuestra de cada bloque de frame_skip.
            if i % env.frame_skip == env.frame_skip - 1:
                targets.append(ctrl.copy())
        prev = waypoint

    for _ in range(150):
        targets.append(np.zeros(2, dtype=np.float64))
    return targets


def scripted_action(env, waypoint):
    """Traduce un objetivo absoluto a la acción incremental que espera el entorno.

    El entorno no acepta posiciones, sino incrementos de como mucho `target_delta`.
    Dividir el error por `target_delta` y recortar a [-1, 1] da la acción que más
    se acerca al objetivo deseado en un solo paso. Esta inversión es lo que permite
    reutilizar el mismo entorno tanto para control programado como para aprendizaje.
    """
    error = waypoint - env.target
    return np.clip(error / env.target_delta, -1.0, 1.0).astype(np.float32)


def eval_sequence(pose, sequence):
    """Simula una secuencia candidata y la puntúa.

    Returns:
        (best, final): el mejor estado según la puntuación y el `info` del
        último paso.

    La puntuación premia la verticalidad y penaliza ligeramente que los servos y
    los objetivos queden lejos de cero. Es decir, no basta con levantarse: se
    prefiere terminar en una postura neutra y sostenible.
    """
    # Importación local a propósito: así construir las candidatas no obliga a
    # cargar MuJoCo, que es lo pesado del proceso.
    from robo1_env import Robo1GetupEnv

    env = Robo1GetupEnv("robo1.xml", fallen_poses=(pose,))
    obs, _ = env.reset(options={"pose": pose})
    best = {"score": -1e9}
    final = {}

    for waypoint in expert_target_schedule(env, sequence):
        obs, _, terminated, truncated, info = env.step(scripted_action(env, waypoint))
        servo = info["servo_angles"]
        score = (
            info["upright"]
            - 0.05 * float(np.sum(servo * servo))
            - 0.02 * float(np.sum(info["target"] * info["target"]))
        )
        if score > best["score"]:
            best = {
                "score": score,
                "upright": info["upright"],
                "servo_angles": servo.copy(),
                "target": info["target"].copy(),
                "goal_pose": info["goal_pose"],
            }
        final = info
        if terminated or truncated:
            break

    return best, final


def build_candidates(seed, random_count):
    """Construye la lista completa de secuencias candidatas a evaluar.

    Args:
        seed: semilla del generador aleatorio, para que la búsqueda sea reproducible.
        random_count: cuántas secuencias aleatorias añadir a las de la rejilla.

    Returns:
        Lista de arrays (N, 2) con los waypoints de cada candidata.
    """
    rng = np.random.default_rng(seed)
    # Nueve ángulos base que cubren todo el recorrido del servo, de tope a tope.
    base_vals = [-1.55, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.55]
    candidates = []

    # Rejilla: por cada par (a, b) se prueban tres formas de encadenarlo, que
    # corresponden a tres maneras distintas de repartir el movimiento entre los
    # dos servos. Son 9 x 9 x 3 = 243 candidatas sistemáticas.
    for a in base_vals:
        for b in base_vals:
            candidates.append(np.array([[a, b], [0.0, b], [0.0, 0.0]], dtype=np.float64))
            candidates.append(np.array([[a, b], [a, 0.0], [0.0, 0.0]], dtype=np.float64))
            candidates.append(np.array([[0.0, b], [a, b], [0.0, 0.0]], dtype=np.float64))

    # Muestreo aleatorio: secuencias más largas (5 waypoints) que la rejilla no
    # cubre. El último waypoint se fuerza a cero para terminar siempre en neutro.
    for _ in range(random_count):
        seq = rng.uniform(-1.55, 1.55, size=(5, 2))
        seq[-1] = 0.0
        candidates.append(seq)

    return candidates


def print_reference_block(results):
    """Imprime las mejores secuencias con el formato listo para copiar y pegar.

    La salida se pasa a `getup_reference.py`, en `getup_sequence_for_pose()`.
    """
    print()
    print("REFERENCE_CANDIDATES")
    for pose, seq in results.items():
        print(pose)
        print(np.array2string(seq, precision=6, separator=", "))


def main():
    """Recorre las posturas pedidas y busca la mejor secuencia para cada una."""
    parser = argparse.ArgumentParser(
        description="Search scripted get-up target-angle sequences for each fallen pose."
    )
    parser.add_argument(
        "--pose",
        choices=FALLEN_POSES,
        action="append",
        help="pose to search; can be passed multiple times. Defaults to all poses.",
    )
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--random-count", type=int, default=1200)
    parser.add_argument(
        "--no-stop-on-goal",
        action="store_true",
        help="keep searching even after a candidate reaches goal_pose",
    )
    args = parser.parse_args()

    poses = tuple(args.pose) if args.pose else FALLEN_POSES
    candidates = build_candidates(args.seed, args.random_count)
    results = {}

    for pose in poses:
        best = None
        best_seq = None
        print("POSE", pose)
        for i, seq in enumerate(candidates):
            detail, final = eval_sequence(pose, seq)
            # Solo se informa cuando se supera el récord anterior: la traza queda
            # legible aunque se evalúen miles de candidatas.
            if best is None or detail["score"] > best["score"]:
                best = detail
                best_seq = seq
                print(
                    "new_best",
                    i,
                    "score",
                    round(best["score"], 4),
                    "upright",
                    round(best["upright"], 4),
                    "goal",
                    best["goal_pose"],
                    "servo",
                    np.round(best["servo_angles"], 4),
                    "target",
                    np.round(best["target"], 4),
                    "seq",
                    np.round(best_seq, 4),
                    "final_up",
                    round(final["upright"], 4),
                )
                # Por defecto la búsqueda se detiene en cuanto una candidata
                # alcanza de verdad la postura objetivo: basta con una que sirva.
                # Con --no-stop-on-goal se sigue buscando por si hay una mejor.
                if best["goal_pose"] and not args.no_stop_on_goal:
                    break

        print("BEST_SEQ", pose, np.round(best_seq, 6))
        print("BEST", best)
        results[pose] = best_seq

    print_reference_block(results)


if __name__ == "__main__":
    main()
