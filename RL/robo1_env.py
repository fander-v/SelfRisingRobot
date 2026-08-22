"""Entorno de Gymnasium/MuJoCo para el robot de 2 grados de libertad `robo1`.

Este módulo define el problema de aprendizaje por refuerzo: un robot de dos servos
que parte de una postura caída y debe volver a quedar de pie por sí solo.

El entorno sigue la interfaz estándar de Gymnasium (`reset` / `step`), por lo que
puede entrenarse directamente con Stable-Baselines3 (PPO en este proyecto).

Resumen del problema:

- Observación (4 valores): roll y pitch del cuerpo, más los dos ángulos objetivo
  actuales de los servos.
- Acción (2 valores en [-1, 1]): incremento relativo que se suma a cada ángulo
  objetivo. La política no ordena posiciones absolutas, sino cambios pequeños;
  eso suaviza el movimiento y limita la velocidad de los servos.
- Recompensa: mezcla de un término principal de verticalidad y varias
  penalizaciones que evitan soluciones bruscas o poses forzadas.
- Fin de episodio: éxito si el robot se mantiene 50 pasos seguidos en la postura
  objetivo; corte por tiempo a los 700 pasos.

NOTA DE TRADUCCIÓN: los comentarios y docstrings en español fueron añadidos sobre
el código original de HomeMadeGarbage
(https://github.com/homemadegarbage/SelfRisingRobot). La lógica no fue modificada.
"""

import math
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


# Las cuatro posturas caídas desde las que se entrena y evalúa, expresadas como
# el par (roll, pitch) en radianes que se aplica al cuerpo al inicio del episodio.
# pi/2 = 90 grados, es decir, el robot tumbado completamente de costado o de frente.
FALLEN_POSES = {
    "roll_pos": (math.pi / 2.0, 0.0),    # tumbado sobre un costado
    "roll_neg": (-math.pi / 2.0, 0.0),   # tumbado sobre el costado opuesto
    "pitch_pos": (0.0, math.pi / 2.0),   # tumbado hacia adelante
    "pitch_neg": (0.0, -math.pi / 2.0),  # tumbado hacia atrás
}
DEFAULT_FALLEN_POSES = tuple(FALLEN_POSES.keys())


def roll_pitch_to_quat(roll, pitch):
    """Convierte un par (roll, pitch) en radianes al cuaternión que usa MuJoCo.

    MuJoCo representa la orientación del cuerpo libre como un cuaternión
    [w, x, y, z] en `qpos[3:7]`, así que esta función traduce los ángulos de
    Euler cómodos para nosotros al formato que espera el simulador.
    """
    cr = math.cos(roll / 2.0)
    sr = math.sin(roll / 2.0)
    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)
    return np.array([cr * cp, sr * cp, cr * sp, -sr * sp], dtype=np.float64)


def quat_to_roll_pitch(q):
    """Operación inversa: del cuaternión de MuJoCo a los ángulos (roll, pitch).

    Se usa en cada paso para construir la observación, porque la política razona
    en términos de inclinación y no de cuaterniones. El yaw se descarta: al robot
    le da igual hacia dónde está orientado, solo importa cuánto está inclinado.
    """
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    # Se recorta a [-1, 1] antes del asin para evitar un NaN por error numérico
    # cuando el valor se pasa mínimamente del rango válido.
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    return roll, pitch


class Robo1GetupEnv(gym.Env):
    """Entorno de levantado (`get-up`) para el robot de dos servos.

    Cada episodio empieza con el robot tumbado en una de las posturas de
    `FALLEN_POSES` y termina cuando consigue mantenerse de pie o cuando se agota
    el número máximo de pasos.
    """

    metadata = {"render_modes": []}

    def __init__(self, xml_path="robo1.xml", fallen_poses=None):
        """Carga el modelo de MuJoCo y define los espacios de acción y observación.

        Args:
            xml_path: ruta al modelo MuJoCo (`robo1.xml`). Necesita las mallas STL
                de `assets/`, por eso los scripts deben ejecutarse desde `RL/`.
            fallen_poses: subconjunto de posturas iniciales a usar. Si es None se
                entrena con las cuatro; pasar una sola sirve para evaluar caso a caso.
        """
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(Path(xml_path)))
        self.data = mujoco.MjData(self.model)

        # --- Hiperparámetros del entorno -----------------------------------
        # Pasos de simulación de MuJoCo por cada paso de la política. Con esto la
        # política decide a ~1/20 de la frecuencia física: acciones más estables
        # y episodios más cortos de simular.
        self.frame_skip = 20
        # Corte por tiempo del episodio (en pasos de política, no de simulación).
        self.max_steps = 700
        # Incremento máximo del ángulo objetivo por paso, en radianes (~4.6 grados).
        # Es lo que convierte la acción en un cambio suave en vez de un salto.
        self.target_delta = 0.08
        # Tope absoluto del ángulo objetivo (~88.8 grados), acorde al recorrido
        # físico de los servos del robot real.
        self.target_limit = 1.55
        # Pasos de simulación que se dejan correr para que el robot "se asiente"
        # en el suelo antes de empezar a contar el episodio.
        self.settle_steps = 200
        self.fallen_poses = list(fallen_poses or DEFAULT_FALLEN_POSES)
        self.current_pose_name = self.fallen_poses[0]

        # --- Índices del modelo ---------------------------------------------
        # MuJoCo trabaja con índices numéricos; aquí se resuelven una sola vez
        # los nombres definidos en robo1.xml para no buscarlos en cada paso.
        self.foot_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "foot"
        )
        # Posición (qpos) de cada articulación de servo.
        self.servo1_qpos_id = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "servo1_joint")
        ]
        self.servo2_qpos_id = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "servo2_joint")
        ]
        # Velocidad (qvel) de cada articulación de servo.
        self.servo1_qvel_id = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "servo1_joint")
        ]
        self.servo2_qvel_id = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "servo2_joint")
        ]
        # Altura de referencia del pie estando de pie: se mide una sola vez al
        # construir el entorno y sirve como objetivo en la recompensa.
        self.standing_height = self._compute_standing_height()

        # --- Espacios de acción y observación --------------------------------
        # Acción: dos valores continuos en [-1, 1], uno por servo. Se multiplican
        # por target_delta, así que representan "cuánto mover el objetivo".
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        # Observación: [roll, pitch, objetivo_servo1, objetivo_servo2].
        # Es deliberadamente pequeña porque esta misma política debe correr
        # después dentro de un M5Atom, con memoria y cómputo muy limitados.
        self.observation_space = spaces.Box(
            low=np.array(
                [-math.pi, -math.pi, -1.55, -1.55],
                dtype=np.float32,
            ),
            high=np.array(
                [math.pi, math.pi, 1.55, 1.55],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # --- Estado interno del episodio -------------------------------------
        self.step_count = 0
        # Ángulos objetivo actuales de los servos; es el estado que la acción modifica.
        self.target = np.zeros(2, dtype=np.float64)
        # Pasos consecutivos cumpliendo la postura objetivo (para terminar con éxito).
        self.success_count = 0
        # Verticalidad del paso anterior, usada para premiar el progreso.
        self.prev_upright = 0.0

    def _compute_standing_height(self):
        """Mide la altura del pie con el robot de pie y en reposo.

        Se simula en una copia limpia de los datos: se coloca el robot vertical,
        con los servos a cero, y se deja caer y asentar. La altura resultante es
        la referencia contra la que después se penaliza la diferencia de altura.
        """
        data = mujoco.MjData(self.model)
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qpos[0:3] = [0.0, 0.0, 0.08]      # posición inicial (x, y, z)
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # cuaternión identidad = sin inclinación
        data.qpos[self.servo1_qpos_id] = 0.0
        data.qpos[self.servo2_qpos_id] = 0.0
        data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, data)

        # Se deja asentar para que la altura medida sea la de equilibrio real y
        # no la de la posición impuesta a mano.
        for _ in range(self.settle_steps):
            data.ctrl[:] = 0.0
            mujoco.mj_step(self.model, data)

        return float(data.xpos[self.foot_body_id, 2])

    def _set_fixed_fallen_pose(self, pose_name=None):
        """Coloca el robot en la postura caída indicada y lo deja asentarse.

        Es determinista: la misma postura produce siempre el mismo estado inicial,
        lo que hace que las evaluaciones sean reproducibles.
        """
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0

        self.data.qpos[0:3] = [0.0, 0.0, 0.08]
        self.current_pose_name = pose_name or self.current_pose_name
        roll, pitch = FALLEN_POSES[self.current_pose_name]
        self.data.qpos[3:7] = roll_pitch_to_quat(roll, pitch)
        self.data.qpos[self.servo1_qpos_id] = 0.0
        self.data.qpos[self.servo2_qpos_id] = 0.0

        self.target[:] = 0.0
        self.data.ctrl[:] = self.target
        mujoco.mj_forward(self.model, self.data)

        # Se simula sin actuar sobre los servos para que el robot termine de caer
        # y quede realmente apoyado en el suelo antes de empezar el episodio.
        for _ in range(self.settle_steps):
            self.data.ctrl[:] = self.target
            mujoco.mj_step(self.model, self.data)

        self.target[:] = 0.0
        self.data.ctrl[:] = self.target

    def _get_obs(self):
        """Construye el vector de observación de 4 elementos.

        Importante para el robot real: roll y pitch son justamente lo que el
        M5Atom puede estimar con su IMU (filtro de Kalman en el sketch), y los
        dos objetivos son variables internas. Es decir, la observación es
        reproducible en hardware sin sensores adicionales.
        """
        roll, pitch = quat_to_roll_pitch(self.data.qpos[3:7])
        return np.array(
            [
                roll,
                pitch,
                self.target[0],
                self.target[1],
            ],
            dtype=np.float32,
        )

    def _upright(self):
        """Mide la verticalidad del cuerpo: 1.0 = perfectamente de pie, -1.0 = invertido.

        Es el elemento [2][2] de la matriz de rotación del cuerpo `foot`, o sea el
        coseno del ángulo entre el eje Z del robot y el eje Z del mundo.
        """
        xmat = self.data.xmat[self.foot_body_id].reshape(3, 3)
        return float(xmat[2, 2])

    def _foot_height(self):
        """Altura actual (coordenada Z) del cuerpo `foot` en el mundo."""
        return float(self.data.xpos[self.foot_body_id, 2])

    def _servo_angles(self):
        """Ángulos reales de los dos servos, que no siempre igualan al objetivo."""
        return np.array(
            [
                self.data.qpos[self.servo1_qpos_id],
                self.data.qpos[self.servo2_qpos_id],
            ],
            dtype=np.float64,
        )

    def _servo_velocities(self):
        """Velocidades angulares de los dos servos, usadas para penalizar brusquedad."""
        return np.array(
            [
                self.data.qvel[self.servo1_qvel_id],
                self.data.qvel[self.servo2_qvel_id],
            ],
            dtype=np.float64,
        )

    def reset(self, seed=None, options=None):
        """Inicia un episodio nuevo.

        Args:
            seed: semilla para el generador aleatorio de Gymnasium.
            options: si trae la clave "pose", fuerza esa postura inicial; si no,
                se elige una al azar entre las disponibles. Forzarla es lo que
                permite evaluar cada caída por separado.

        Returns:
            (observación inicial, dict con el nombre de la postura usada)
        """
        super().reset(seed=seed)
        self.step_count = 0
        self.success_count = 0
        pose_name = None
        if options is not None:
            pose_name = options.get("pose")
        if pose_name is None:
            pose_name = self.np_random.choice(self.fallen_poses)
        self._set_fixed_fallen_pose(pose_name)
        self.prev_upright = self._upright()
        return self._get_obs(), {"pose": self.current_pose_name}

    def step(self, action):
        """Aplica una acción, avanza la simulación y devuelve la recompensa.

        El flujo es: la acción modifica los ángulos objetivo, esos objetivos se
        mantienen durante `frame_skip` pasos de física, y sobre el estado
        resultante se calcula la recompensa.
        """
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)

        # La acción es incremental: desplaza el objetivo en vez de fijarlo.
        old_target = self.target.copy()
        self.target += action * self.target_delta
        self.target = np.clip(self.target, -self.target_limit, self.target_limit)

        # Se mantiene el mismo objetivo durante varios pasos de física.
        for _ in range(self.frame_skip):
            self.data.ctrl[:] = self.target
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        obs = self._get_obs()

        # --- Magnitudes con las que se arma la recompensa ---------------------
        upright = self._upright()
        # Cuánto mejoró la verticalidad respecto al paso anterior: premia el
        # progreso, no solo el estado final. Es clave para que el agente aprenda
        # algo desde el principio, cuando todavía nunca ha llegado a ponerse de pie.
        upright_progress = upright - self.prev_upright
        self.prev_upright = upright
        roll, pitch = float(obs[0]), float(obs[1])
        servo_angles = self._servo_angles()
        servo_velocities = self._servo_velocities()
        foot_height_error = self._foot_height() - self.standing_height

        # --- Penalizaciones ---------------------------------------------------
        # Inclinación residual del cuerpo.
        tilt_cost = 0.2 * (roll * roll + pitch * pitch)
        # "Puerta" que activa progresivamente las penalizaciones de postura final:
        # vale 0 mientras el robot está tumbado (upright <= 0.65) y sube a 1 cuando
        # ya está casi vertical. Así no se le castiga por doblar los servos durante
        # la maniobra, solo por quedarse en una pose forzada una vez levantado.
        stand_gate = float(np.clip((upright - 0.65) / 0.35, 0.0, 1.0))
        # Ya de pie, se busca que los servos vuelvan a su posición neutra...
        servo_zero_cost = stand_gate * 0.5 * float(np.sum(servo_angles * servo_angles))
        # ...y que los objetivos comandados también estén cerca de cero.
        target_zero_cost = stand_gate * 0.2 * float(np.sum(self.target * self.target))
        # Diferencia respecto a la altura de referencia estando de pie.
        height_cost = stand_gate * 5.0 * foot_height_error * foot_height_error
        # Velocidad del cuerpo libre (6 grados de libertad): penaliza el balanceo.
        body_velocity_cost = 0.01 * float(np.sum(self.data.qvel[0:6] * self.data.qvel[0:6]))
        # Velocidad de los servos: evita movimientos violentos que el servo real
        # no podría seguir.
        servo_velocity_cost = 0.005 * float(np.sum(servo_velocities * servo_velocities))
        # Magnitud de la acción: favorece control suave.
        action_cost = 0.005 * float(np.sum(action * action))
        # Cambio del objetivo entre pasos: penaliza el temblor o "chattering".
        servo_motion_cost = 0.02 * float(np.sum((self.target - old_target) ** 2))
        # Recompensa total. El peso 8.0 del progreso es mayor que el 2.0 del estado
        # absoluto: el diseño prioriza que el robot avance hacia arriba.
        reward = (
            2.0 * upright
            + 8.0 * upright_progress
            - tilt_cost
            - servo_zero_cost
            - target_zero_cost
            - height_cost
            - body_velocity_cost
            - servo_velocity_cost
            - action_cost
            - servo_motion_cost
        )

        # --- Condición de éxito ------------------------------------------------
        # Todas las condiciones deben cumplirse a la vez: bien vertical, poco
        # inclinado, servos casi neutros, a la altura correcta y prácticamente
        # quieto. Lo último evita contar como éxito un paso "de suerte" mientras
        # el robot todavía va cayendo.
        goal_pose = (
            upright > 0.95
            and abs(roll) < 0.25
            and abs(pitch) < 0.25
            and np.max(np.abs(servo_angles)) < 0.12
            and abs(foot_height_error) < 0.02
            and np.linalg.norm(self.data.qvel[0:6]) < 0.25
        )

        if goal_pose:
            self.success_count += 1
            reward += 5.0  # bonificación por cada paso en la postura objetivo
        else:
            self.success_count = 0  # el contador exige pasos CONSECUTIVOS

        # Éxito real: mantenerse 50 pasos seguidos de pie, no solo tocarlo una vez.
        terminated = self.success_count >= 50
        truncated = self.step_count >= self.max_steps
        # `info` no interviene en el entrenamiento, pero es lo que consultan los
        # scripts de evaluación y búsqueda para medir la calidad del resultado.
        info = {
            "pose": self.current_pose_name,
            "upright": upright,
            "foot_height_error": foot_height_error,
            "servo_angles": servo_angles.copy(),
            "goal_pose": goal_pose,
            "target": self.target.copy(),
            "success_count": self.success_count,
        }
        return obs, reward, terminated, truncated, info
