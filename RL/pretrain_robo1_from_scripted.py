"""Preentrenamiento de la política PPO por imitación de las trayectorias de referencia.

Este es el paso que hace viable todo el proyecto. Aprender a levantarse solo con
exploración aleatoria es muy difícil: la recompensa por conseguirlo casi nunca
aparece por azar, así que PPO puede pasarse millones de pasos sin señal útil.

La solución aquí es clonación de comportamiento (behavior cloning): se generan
pares (observación, acción) siguiendo las trayectorias programadas de
`getup_reference.py` y se entrena la red de la política por regresión para que
reproduzca esas acciones. El resultado ya se levanta; después `train_robo1.py`
lo afina con PPO.

Detalle importante: el entrenamiento es supervisado puro (Adam sobre el error
cuadrático medio), no hay recompensas de por medio. Se aprovecha el objeto PPO
solo porque así la red resultante queda en el formato que espera Stable-Baselines3
y se puede guardar como `robo1_getup_ppo.zip`.

Salida: `robo1_getup_ppo.zip`.

NOTA DE TRADUCCIÓN: comentarios y docstrings en español añadidos sobre el código
original de HomeMadeGarbage. La lógica no fue modificada.
"""

import numpy as np
import torch
import mujoco
from stable_baselines3 import PPO

from getup_reference import getup_sequence_for_pose
from robo1_env import FALLEN_POSES, Robo1GetupEnv, roll_pitch_to_quat


# Variantes de cada postura inicial, como desviaciones en grados sobre el
# (roll, pitch) base. Es aumento de datos: al robot real nunca se le cae
# exactamente a 90 grados, así que entrenar con perturbaciones hace la política
# más robusta. `pitch_pos` recibe 13 variantes porque es la caída más difícil;
# las demás solo usan la postura nominal.
POSE_VARIANTS_DEG = {
    "roll_pos": [(0.0, 0.0)],
    "roll_neg": [(0.0, 0.0)],
    "pitch_pos": [
        (0.0, 0.0),
        (-20.0, -20.0),
        (-20.0, 0.0),
        (-20.0, 20.0),
        (-10.0, -10.0),
        (-10.0, 10.0),
        (0.0, -20.0),
        (0.0, 20.0),
        (10.0, -10.0),
        (10.0, 10.0),
        (20.0, -20.0),
        (20.0, 0.0),
        (20.0, 20.0),
    ],
    "pitch_neg": [(0.0, 0.0)],
}


def expert_target_schedule(env, sequence):
    """Convierte waypoints en una lista de objetivos, uno por paso de política.

    Idéntica a la de `search_all_getup.py`: interpola 350 pasos de física entre
    waypoints, se queda con una muestra de cada `frame_skip` y añade 150 pasos
    finales en cero para dejar que el robot se estabilice de pie.
    """
    targets = []
    prev = np.zeros(2, dtype=np.float64)
    for waypoint in sequence:
        for i in range(350):
            t = (i + 1) / 350
            ctrl = (1.0 - t) * prev + t * waypoint
            if i % env.frame_skip == env.frame_skip - 1:
                targets.append(ctrl.copy())
        prev = waypoint

    for _ in range(150):
        targets.append(np.zeros(2, dtype=np.float64))
    return targets


def scripted_action(env, waypoint):
    """Calcula la acción del "experto": el incremento que acerca al objetivo.

    Estas son precisamente las etiquetas que la red aprenderá a predecir a partir
    de la observación.
    """
    error = waypoint - env.target
    return np.clip(error / env.target_delta, -1.0, 1.0).astype(np.float32)


def reset_variant(env, pose, roll_offset_deg, pitch_offset_deg):
    """Reinicia el entorno en una variante perturbada de una postura caída.

    Replica lo que hace `Robo1GetupEnv._set_fixed_fallen_pose()`, pero sumando
    desviaciones en roll y pitch. Se manipula el entorno desde fuera porque el
    `reset()` estándar solo admite las cuatro posturas nominales.

    Args:
        env: entorno a reiniciar (se modifica en el sitio).
        pose: postura base cuyo (roll, pitch) se toma como referencia.
        roll_offset_deg, pitch_offset_deg: desviaciones en grados.

    Returns:
        (observación inicial, dict con el nombre de la postura), igual que reset().
    """
    base_roll, base_pitch = FALLEN_POSES[pose]
    roll = base_roll + np.deg2rad(roll_offset_deg)
    pitch = base_pitch + np.deg2rad(pitch_offset_deg)

    env.data.qpos[:] = 0.0
    env.data.qvel[:] = 0.0
    env.data.qpos[0:3] = [0.0, 0.0, 0.08]
    env.data.qpos[3:7] = roll_pitch_to_quat(roll, pitch)
    env.target[:] = 0.0
    env.data.ctrl[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    for _ in range(env.settle_steps):
        env.data.ctrl[:] = 0.0
        mujoco.mj_step(env.model, env.data)

    env.target[:] = 0.0
    env.data.ctrl[:] = 0.0
    # Se reinicia también el estado interno del episodio, ya que no pasamos por reset().
    env.step_count = 0
    env.success_count = 0
    env.prev_upright = env._upright()
    env.current_pose_name = pose
    return env._get_obs(), {"pose": pose}


def collect_expert_data():
    """Genera el conjunto de datos de imitación recorriendo posturas y variantes.

    Returns:
        (observations, actions): dos arrays float32 alineados, donde `actions[i]`
        es la acción que el experto tomó al observar `observations[i]`.

    Se ejecuta la trayectoria dentro del entorno real, así que las observaciones
    corresponden a estados físicamente alcanzables, no a estados inventados.
    """
    observations = []
    actions = []

    for pose in ("roll_pos", "roll_neg", "pitch_pos", "pitch_neg"):
        for roll_offset, pitch_offset in POSE_VARIANTS_DEG[pose]:
            # Un entorno nuevo por variante, para no arrastrar estado entre ellas.
            env = Robo1GetupEnv("robo1.xml", fallen_poses=(pose,))
            obs, _ = reset_variant(env, pose, roll_offset, pitch_offset)
            sequence = getup_sequence_for_pose(pose)

            for waypoint in expert_target_schedule(env, sequence):
                action = scripted_action(env, waypoint)
                # Se guarda el par ANTES de aplicar la acción: la etiqueta debe
                # corresponder a la observación sobre la que se decidió.
                observations.append(obs.copy())
                actions.append(action.copy())

                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def evaluate_policy(model):
    """Comprueba la política preentrenada en las cuatro posturas nominales.

    Returns:
        Diccionario por postura con la mejor verticalidad alcanzada y el `info`
        final. Sirve para saber si el preentrenamiento ya funciona antes de
        invertir tiempo en el entrenamiento con PPO.
    """
    results = {}
    for pose in ("roll_pos", "roll_neg", "pitch_pos", "pitch_neg"):
        env = Robo1GetupEnv("robo1.xml", fallen_poses=(pose,))
        obs, _ = env.reset(options={"pose": pose})
        best_upright = -1.0
        final_info = {}

        for _ in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            best_upright = max(best_upright, info["upright"])
            final_info = info
            if terminated or truncated:
                break

        results[pose] = {"best_upright": best_upright, "final_info": final_info}
    return results


def main():
    """Recoge las demostraciones, entrena por regresión y guarda el modelo."""
    observations, actions = collect_expert_data()
    env = Robo1GetupEnv("robo1.xml")
    # El objeto PPO se crea solo para obtener la red de política con la
    # arquitectura correcta y poder guardarla en formato Stable-Baselines3.
    # En este script no se usa el algoritmo PPO en sí.
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        verbose=0,
        device="cpu",
    )

    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=model.device)
    act_tensor = torch.as_tensor(actions, dtype=torch.float32, device=model.device)
    # Optimizador propio, independiente del que usaría PPO, con tasa más alta
    # (1e-3) porque esto es regresión supervisada y converge rápido.
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)

    # Entrenamiento a lote completo: el conjunto es pequeño y cabe entero en
    # memoria, así que cada época es una sola actualización sobre todos los datos.
    for epoch in range(5000):
        dist = model.policy.get_distribution(obs_tensor)
        # Se toma la media de la distribución (la acción determinista) y se
        # ajusta contra la acción del experto por error cuadrático medio.
        pred = dist.distribution.mean
        loss = torch.mean((pred - act_tensor) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print("epoch", epoch, "loss", float(loss.detach().cpu()))

    print("expert_samples", len(observations))
    results = evaluate_policy(model)
    for pose, result in results.items():
        print(pose, result)
    # Se guarda con el nombre que espera train_robo1.py como --model-in.
    model.save("robo1_getup_ppo")


if __name__ == "__main__":
    main()
