"""Reproduce visualmente la política entrenada en el visor de MuJoCo.

Es la comprobación cualitativa del resultado: se ve al robot levantarse en
tiempo real. Al terminar un episodio (por éxito o por corte de tiempo) se
reinicia automáticamente, así que el visor queda en bucle.

Uso:

    python play_robo1_policy.py                  # postura inicial aleatoria
    python play_robo1_policy.py --pose roll_pos  # siempre la misma postura

NOTA DE TRADUCCIÓN: comentarios y docstrings en español añadidos sobre el código
original de HomeMadeGarbage. La lógica no fue modificada.
"""

import argparse
import time

import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

from robo1_env import DEFAULT_FALLEN_POSES, Robo1GetupEnv


def main():
    """Carga el modelo entrenado y lo ejecuta en bucle dentro del visor."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose", choices=DEFAULT_FALLEN_POSES, default=None)
    args = parser.parse_args()

    # Si se fija una postura, el entorno se restringe solo a ella; si no, se usan
    # las cuatro y cada reinicio elige una al azar.
    fallen_poses = (args.pose,) if args.pose is not None else DEFAULT_FALLEN_POSES
    env = Robo1GetupEnv("robo1.xml", fallen_poses=fallen_poses)
    model = PPO.load("robo1_getup_ppo")
    reset_options = {"pose": args.pose} if args.pose is not None else None
    obs, info = env.reset(options=reset_options)
    print("initial pose:", info["pose"])

    # `launch_passive` abre el visor sin tomar el control del bucle de simulación:
    # somos nosotros quienes avanzamos la física y luego sincronizamos la vista.
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            # deterministic=True usa la media de la distribución de la política en
            # lugar de muestrear. En evaluación interesa el comportamiento
            # aprendido, no la exploración.
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            viewer.sync()
            # Espera igual al tiempo simulado en ese paso, para que la animación
            # transcurra aproximadamente a velocidad real.
            time.sleep(env.model.opt.timestep * env.frame_skip)
            if terminated or truncated:
                obs, info = env.reset(options=reset_options)
                print("initial pose:", info["pose"])


if __name__ == "__main__":
    main()
