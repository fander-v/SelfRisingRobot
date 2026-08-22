# SelfRisingRobot

> **Traducción al español.** Este es un fork traducido del repositorio original
> [homemadegarbage/SelfRisingRobot](https://github.com/homemadegarbage/SelfRisingRobot),
> creado por HomeMadeGarbage. Todo el crédito del diseño, el código y el trabajo
> original corresponde al autor original. Aquí solo se tradujo la documentación
> y los comentarios del japonés al español; el código y los modelos no se modificaron.

"poco" es un robot de dos ejes con servos capaz de levantarse por sí solo.

El movimiento de levantarse se aprende con aprendizaje por refuerzo en MuJoCo y luego
se ejecuta en el robot físico, controlado por un M5Atom.

Artículo detallado (en japonés):

https://homemadegarbage.com/rl13

## Contenido

- `3Dmodel/` - modelos para impresión 3D del robot físico
- `RL/` - modelo de MuJoCo, entorno de aprendizaje por refuerzo y modelo PPO entrenado
- `Arduino/` - sketch para M5Atom y la red de política ya exportada

Para instrucciones detalladas de uso, consulta el README dentro de cada carpeta.

## Modelo 3D

En `3Dmodel/` están los archivos STL para fabricar el robot físico.

- `footP.stl`
- `arm1P.stl`
- `arm2P.stl`
- `armhornP.stl`

## Aprendizaje por refuerzo

En `RL/` están la simulación y el modelo entrenado.

Contenido principal:

- modelo de MuJoCo
- entorno de Gymnasium
- modelo PPO entrenado con Stable-Baselines3
- scripts de reproducción y evaluación

## Arduino

En `Arduino/` está el sketch para M5Atom que controla el robot físico.

Contenido principal:

- `robo03.ino`
- `policy_network.h`

`policy_network.h` es la política entrenada convertida a una cabecera de C, y se usa
para ejecutar el movimiento de levantarse en el M5Atom.
