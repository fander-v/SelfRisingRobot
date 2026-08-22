# robo03

> **Traducción al español** del README original de
> [homemadegarbage/SelfRisingRobot](https://github.com/homemadegarbage/SelfRisingRobot).
> Autor original: HomeMadeGarbage.

Sketch de Arduino que controla un robot de 2 grados de libertad con un M5Atom y
ejecuta el movimiento de levantarse usando la política entrenada.

## Archivos

- `robo03/robo03.ino` - sketch de control para M5Atom
- `robo03/policy_network.h` - la política entrenada convertida a cabecera de C

`policy_network.h` se incluye desde `robo03.ino`, así que debe estar siempre dentro de
la misma carpeta `robo03`.

## Librerías necesarias

Instálalas, por ejemplo, desde el Gestor de Librerías del IDE de Arduino:

- M5Atom
- Kalman
- ESP32Servo

También hace falta el entorno de placas ESP32.

## Carga al dispositivo

Abre `robo03/robo03.ino` en el IDE de Arduino, compila para M5Atom y carga el programa.

## Control por Wi-Fi

Al encenderse, el M5Atom crea un punto de acceso Wi-Fi.

- SSID: `robo1`
- Contraseña: `password`
- URL: `http://192.168.42.1`

Al entrar desde el navegador puedes ajustar los servos manualmente, iniciar y detener
el movimiento de levantarse, y activar o desactivar el levantado automático.

## Botón

El movimiento de levantarse también se puede iniciar con el botón del propio M5Atom.

Si pulsas el botón mientras el robot se está levantando, el movimiento se detiene y se
vuelve al modo manual.

## Notas

- Se asume que el servo 1 está conectado al GPIO 26 y el servo 2 al GPIO 32.
- El offset y el ancho de pulso de los servos se ajustan desde la página web y se
  guardan en Preferences.
- `policy_network.h` es una red de política ya generada. Si quieres usar otro resultado
  de entrenamiento, reemplaza este archivo.
