"""Reproduce en el visor de MuJoCo la trayectoria de referencia, sin red neuronal.

Sirve para comprobar a ojo que los waypoints de `getup_reference.py` realmente
levantan al robot antes de usarlos como datos de imitación. Aquí no interviene
ninguna política aprendida: los ángulos objetivo se interpolan directamente
entre waypoints.

Uso:

    python scripted_getup.py --pose roll_pos

Terminada la secuencia, el script deja los servos en cero y sigue imprimiendo la
verticalidad, el roll, el pitch y los ángulos de los servos, para verificar que
la postura final es estable y no se derrumba pasados unos segundos.

NOTA DE TRADUCCIÓN: comentarios y docstrings en español añadidos sobre el código
original de HomeMadeGarbage. La lógica no fue modificada.
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from getup_reference import getup_sequence_for_pose
from robo1_env import Robo1GetupEnv, quat_to_roll_pitch


# Copia local de la secuencia genérica. La que realmente se ejecuta viene de
# getup_reference.getup_sequence_for_pose(); esta queda como referencia histórica.
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


def ramp(a, b, n):
    """Genera una interpolación lineal de `a` a `b` en `n` pasos.

    Los waypoints son posiciones sueltas; enviarlos de golpe provocaría un salto
    imposible para un servo real. Esta rampa los convierte en un barrido suave.

    Yields:
        Vectores intermedios; el último es exactamente `b`.
    """
    for i in range(n):
        t = (i + 1) / n
        yield (1.0 - t) * a + t * b


def main():
    """Ejecuta la trayectoria de referencia de una postura dentro del visor."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pose",
        choices=("roll_pos", "roll_neg", "pitch_pos", "pitch_neg"),
        default="roll_pos",
    )
    args = parser.parse_args()

    env = Robo1GetupEnv("robo1.xml", fallen_poses=(args.pose,))
    env.reset(options={"pose": args.pose})
    sequence = getup_sequence_for_pose(args.pose)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        prev = env.target.copy()
        for target in sequence:
            # 350 pasos de física por tramo: un movimiento deliberadamente lento,
            # para que la inercia ayude a levantar en vez de hacer rebotar al robot.
            for ctrl in ramp(prev, target, 350):
                if not viewer.is_running():
                    return
                # Se escribe directamente en data.ctrl, saltándose env.step():
                # esto es control programado, no aprendido.
                env.data.ctrl[:] = ctrl
                mujoco.mj_step(env.model, env.data)
                viewer.sync()
                time.sleep(env.model.opt.timestep)
            prev = target

        # Fase de comprobación: servos en neutro y monitoreo del estado final.
        while viewer.is_running():
            env.data.ctrl[:] = 0.0
            mujoco.mj_step(env.model, env.data)
            viewer.sync()
            roll, pitch = quat_to_roll_pitch(env.data.qpos[3:7])
            print(
                "upright",
                round(env._upright(), 4),
                "roll",
                round(roll, 4),
                "pitch",
                round(pitch, 4),
                "servo",
                np.round(env._servo_angles(), 4),
            )
            time.sleep(env.model.opt.timestep * 20)


if __name__ == "__main__":
    main()
