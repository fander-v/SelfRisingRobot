# robo1 getup PPO

> **Traducción al español** del README original de
> [homemadegarbage/SelfRisingRobot](https://github.com/homemadegarbage/SelfRisingRobot).
> Autor original: HomeMadeGarbage.

Modelo PPO entrenado para que el robot de 2 grados de libertad `robo1`, simulado en
MuJoCo, se levante desde una postura caída.

## Archivos

Estos son los archivos necesarios para ejecutar el repositorio:

- `robo1_getup_ppo.zip` - modelo PPO entrenado con Stable-Baselines3
- `robo1_env.py` - entorno Gymnasium/MuJoCo
- `robo1.xml` - definición del modelo de MuJoCo
- `assets/` - mallas STL que referencia `robo1.xml`
- `play_robo1_policy.py` - reproducción del modelo entrenado
- `eval_robo1_policy.py` - evaluación desde cada postura caída
- `search_all_getup.py` - busca trayectorias candidatas de levantado para cada postura
  caída y emite el `BEST_SEQ` que luego se lleva a `getup_reference.py`
- `getup_reference.py` - secuencias de waypoints con los ángulos objetivo de los servos
  para cada postura caída, usadas para generar los datos de referencia (datos maestros)
- `scripted_getup.py` - reproduce en el visor de MuJoCo la trayectoria de levantado
  definida en `getup_reference.py` para comprobarla
- `pretrain_robo1_from_scripted.py` - genera pares de observación y acción de referencia
  a partir de las trayectorias de `getup_reference.py` y preentrena la política PPO
- `train_robo1.py` - entrenamiento adicional con PPO
- `export_policy_header.py` - genera la cabecera de C para Arduino a partir del modelo
  entrenado
- `requirements.txt` - dependencias de Python

## Preparación del entorno

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Reproducción

```powershell
python play_robo1_policy.py
```

Para reproducir desde una postura inicial concreta:

```powershell
python play_robo1_policy.py --pose roll_pos
python play_robo1_policy.py --pose roll_neg
python play_robo1_policy.py --pose pitch_pos
python play_robo1_policy.py --pose pitch_neg
```

## Evaluación

```powershell
python eval_robo1_policy.py
```

## Entrenamiento

Se preentrena una política inicial con los datos de referencia de levantado definidos en
`getup_reference.py`.

Si quieres volver a buscar las trayectorias de levantado, ejecuta:

```powershell
python search_all_getup.py
```

Después lleva el arreglo `BEST_SEQ` (o `REFERENCE_CANDIDATES`) que se imprime a la
función `getup_sequence_for_pose()` de `getup_reference.py`.


Preentrenamiento de la política inicial a partir de los datos de referencia:

```powershell
python pretrain_robo1_from_scripted.py
```

Con esto se genera `robo1_getup_ppo.zip`.

A continuación se sigue entrenando con PPO:

```powershell
python train_robo1.py --model-in robo1_getup_ppo.zip --timesteps 200000 --n-envs 6 --model-out robo1_getup_ppo
```

Para entrenar con PPO desde cero, omite `--model-in`:

```powershell
python train_robo1.py --timesteps 200000 --n-envs 6 --model-out robo1_getup_ppo
```

## Exportación para Arduino

Genera `policy_network.h` a partir del modelo entrenado:

```powershell
python export_policy_header.py robo1_getup_ppo.zip -o policy_network.h
```

Coloca el `policy_network.h` generado junto al sketch de Arduino para usarlo.

## Notas

- `robo1_getup_ppo.zip` depende de la definición del entorno en `robo1.xml` y
  `robo1_env.py`.
- Sin los archivos STL de `assets/` no se puede cargar el modelo de MuJoCo.
- Ejecuta los scripts desde la raíz del repositorio.
