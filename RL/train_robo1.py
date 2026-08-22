"""Entrenamiento con PPO de la política de levantado.

Se puede usar de dos formas:

1. Continuando desde el modelo preentrenado por imitación
   (`pretrain_robo1_from_scripted.py`), pasando `--model-in`. Es lo recomendado:
   el preentrenamiento ya deja una política que se levanta, y PPO solo la afina.
2. Desde cero, omitiendo `--model-in`. Funciona, pero converge mucho más lento
   porque el agente debe descubrir la maniobra por exploración pura.

Ejemplo:

    python train_robo1.py --model-in robo1_getup_ppo.zip --timesteps 200000 \
        --n-envs 6 --model-out robo1_getup_ppo

NOTA DE TRADUCCIÓN: comentarios y docstrings en español añadidos sobre el código
original de HomeMadeGarbage. La lógica no fue modificada.
"""

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from robo1_env import Robo1GetupEnv


# Número de entornos que se simulan en paralelo por defecto (uno por proceso).
# Conviene ajustarlo al número de núcleos disponibles de la CPU.
N_ENVS = 6


def make_env():
    """Devuelve una fábrica de entornos, como exige SubprocVecEnv.

    SubprocVecEnv necesita funciones que construyan el entorno *dentro* de cada
    proceso hijo, no instancias ya creadas: un modelo de MuJoCo no se puede
    compartir entre procesos.
    """
    def _init():
        return Robo1GetupEnv("robo1.xml")

    return _init


def main():
    """Analiza los argumentos, entrena con PPO y guarda el modelo resultante."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model-in",
        default=None,
        help="existing PPO model zip to continue training from",
    )
    parser.add_argument(
        "--model-out",
        default="robo1_getup_ppo",
        help="output model name or zip path",
    )
    args = parser.parse_args()

    # Comprueba que el entorno cumple la especificación de Gymnasium (formas de
    # los espacios, tipos devueltos, etc.) antes de gastar tiempo entrenando.
    check_env(Robo1GetupEnv("robo1.xml"), warn=True)
    # Varios entornos en procesos separados: más muestras por segundo y también
    # más diversidad de experiencia, porque cada uno arranca en una postura distinta.
    env = SubprocVecEnv([make_env() for _ in range(args.n_envs)])
    env.seed(args.seed)

    if args.model_in:
        # Entrenamiento incremental sobre una política existente.
        model = PPO.load(args.model_in, env=env, device="cpu")
    else:
        # Política nueva. Se usa "MlpPolicy": la red por defecto de SB3, un
        # perceptrón multicapa pequeño; es justo lo que después se puede exportar
        # a C y ejecutar dentro del M5Atom.
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=512,        # pasos recogidos por entorno antes de cada actualización
            batch_size=256,     # tamaño del minilote de optimización
            learning_rate=3e-4,  # tasa de aprendizaje típica de PPO
            gamma=0.99,         # factor de descuento: horizonte relativamente largo
            seed=args.seed,
            # CPU en vez de GPU a propósito: la red es diminuta y el cuello de
            # botella es la simulación física, no el cálculo de la red.
            device="cpu",
        )

    model.learn(total_timesteps=args.timesteps)
    model.save(args.model_out)
    env.close()


if __name__ == "__main__":
    main()
