"""Exporta la política PPO entrenada a una cabecera de C para el M5Atom.

Este script cierra el ciclo del proyecto: convierte la red neuronal entrenada en
PyTorch/Stable-Baselines3 en un archivo `policy_network.h` que se compila dentro
del sketch de Arduino y se ejecuta en el microcontrolador, sin PyTorch, sin
Python y sin conexión a nada.

Cómo lo hace:

1. Carga el `.zip` del modelo y saca su `state_dict`.
2. Se queda solo con las capas de la política (`policy_net` y `action_net`),
   descartando la red de valor (`value_net`), que solo hace falta al entrenar.
   Esto es lo que reduce a la mitad la memoria necesaria en el microcontrolador.
3. Escribe los pesos y sesgos como arrays `static const float`.
4. Genera el código C de la pasada hacia adelante, desenrollado con las
   dimensiones exactas de cada capa.

La activación es tanh en todas las capas ocultas y lineal en la de salida, que es
justo lo que usa la "MlpPolicy" por defecto de Stable-Baselines3.

Uso:

    python export_policy_header.py robo1_getup_ppo.zip -o policy_network.h

NOTA DE TRADUCCIÓN: comentarios y docstrings en español añadidos sobre el código
original de HomeMadeGarbage. La lógica no fue modificada.
"""

import argparse
import re

from stable_baselines3 import PPO


MODEL_PATH = "robo1_getup_ppo.zip"
OUTPUT_HEADER = "policy_network.h"


def is_policy_key(key: str) -> bool:
    """Indica si una clave del state_dict pertenece a la red de política.

    Se excluye todo lo demás (sobre todo `value_net`): el crítico solo sirve
    durante el entrenamiento y no tiene por qué ocupar espacio en el M5Atom.
    """
    return ("policy_net" in key) or ("action_net" in key)


def layer_order_index(key: str) -> int:
    """Deduce el orden de una capa a partir del nombre de su parámetro.

    El state_dict es un diccionario sin orden garantizado, pero al generar código
    C el orden importa. Se extrae el número de `policy_net.N` y se le asigna a
    `action_net` un índice enorme (10^6) para forzar que quede siempre al final,
    que es donde va la capa de salida.
    """
    if "policy_net" in key:
        match = re.search(r"policy_net\.(\d+)", key)
        if match:
            return int(match.group(1))
    if "action_net" in key:
        return 10**6

    # Respaldo para nombres inesperados: se usa el último número que aparezca, y
    # si no hay ninguno, un índice todavía mayor para dejarlo al final.
    numbers = re.findall(r"\d+", key)
    return int(numbers[-1]) if numbers else 10**9


def sanitize(key: str) -> str:
    """Convierte una clave de PyTorch en un identificador válido de C.

    Los puntos y corchetes de nombres como `mlp_extractor.policy_net.0.weight`
    no son legales en C, así que se sustituyen por guiones bajos.
    """
    return key.replace(".", "_").replace("[", "_").replace("]", "_")


def extract_policy_layers(state_dict):
    """Agrupa pesos y sesgos por capa y los devuelve en orden de ejecución.

    Args:
        state_dict: diccionario de parámetros de `model.policy`.

    Returns:
        Lista de dicts con los nombres C y los arrays numpy de cada capa
        (`w_name`, `b_name`, `W`, `b`), ordenados de la entrada a la salida.
    """
    layers_by_index = {}

    for key, value in state_dict.items():
        if not is_policy_key(key):
            continue

        if key.endswith(".weight") or key.endswith(".bias"):
            index = layer_order_index(key)
            if index not in layers_by_index:
                layers_by_index[index] = {"W": None, "b": None}

            # detach().cpu().numpy() saca el tensor del grafo de autograd y lo
            # trae a memoria de CPU como array de numpy.
            if key.endswith(".weight"):
                layers_by_index[index]["W"] = value.detach().cpu().numpy()
                layers_by_index[index]["w_raw_key"] = key
            else:
                layers_by_index[index]["b"] = value.detach().cpu().numpy()
                layers_by_index[index]["b_raw_key"] = key

    layers = []
    for index in sorted(layers_by_index.keys()):
        entry = layers_by_index[index]
        weight = entry["W"]
        bias = entry["b"]
        # Se descartan capas incompletas: sin peso y sesgo no se puede generar
        # código válido.
        if weight is None or bias is None:
            continue

        layers.append(
            {
                "w_name": sanitize(entry["w_raw_key"]) + "_weight",
                "b_name": sanitize(entry["b_raw_key"]) + "_bias",
                "W": weight,
                "b": bias,
            }
        )

    return layers


def write_array(file, name, arr):
    """Escribe un array numpy como array C `static const float`.

    Los pesos salen como matriz 2D `[salidas][entradas]` y los sesgos como vector
    1D. `static const` importa: en ESP32 permite que los datos vivan en flash y no
    consuman la escasa RAM.
    """
    if arr.ndim == 2:
        rows, cols = arr.shape
        file.write(f"static const float {name}[{rows}][{cols}] = {{\n")
        for row in arr:
            # 8 decimales: suficiente para no perder precisión útil en float32.
            file.write("    {" + ", ".join(f"{x:.8f}" for x in row) + "},\n")
        file.write("};\n\n")
    else:
        file.write(f"static const float {name}[{arr.shape[0]}] = {{\n")
        file.write(", ".join(f"{x:.8f}" for x in arr))
        file.write("};\n\n")


def generate_forward(layers):
    """Genera el código C de la pasada hacia adelante de la red.

    En vez de escribir un intérprete genérico de redes, se emite código con las
    dimensiones ya fijadas: bucles con límites constantes que el compilador puede
    optimizar bien y sin ninguna reserva dinámica de memoria.

    Returns:
        El código C como una sola cadena.
    """
    code = []
    code.append("// Policy-only MLP forward pass generated from Stable-Baselines3 PPO.\n")
    code.append("#include <math.h>\n")
    code.append("static inline float tanhf_fast(float x) { return tanhf(x); }\n")
    code.append("static inline void forward_policy(const float input[], float output[]) {")
    code.append("    const float *curr = input;")

    for index, layer in enumerate(layers):
        w_name = layer["w_name"]
        b_name = layer["b_name"]
        weight = layer["W"]
        # PyTorch guarda los pesos como [salidas][entradas].
        out_dim, in_dim = weight.shape

        # `static` en el buffer de cada capa: evita reservar en la pila, que en un
        # microcontrolador es muy limitada.
        code.append(f"    static float layer{index}[{out_dim}];")
        code.append(f"    for (int i = 0; i < {out_dim}; i++) {{")
        code.append(f"        float s = {b_name}[i];")
        code.append(f"        for (int j = 0; j < {in_dim}; j++) {{")
        code.append(f"            s += {w_name}[i][j] * curr[j];")
        code.append("        }")

        # tanh en las capas ocultas; la de salida queda lineal, igual que en SB3.
        if index < len(layers) - 1:
            code.append(f"        layer{index}[i] = tanhf_fast(s);")
        else:
            code.append(f"        layer{index}[i] = s;")

        code.append("    }")
        # La salida de esta capa es la entrada de la siguiente.
        code.append(f"    curr = layer{index};")

    final_dim = layers[-1]["W"].shape[0]
    code.append(f"    for (int i = 0; i < {final_dim}; i++) output[i] = curr[i];")
    code.append("}\n")
    return "\n".join(code)


def parse_args():
    """Define y analiza los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Export a Stable-Baselines3 PPO policy to a C header."
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        default=MODEL_PATH,
        help=f"source Stable-Baselines3 model zip file (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=OUTPUT_HEADER,
        help=f"output header file (default: {OUTPUT_HEADER})",
    )
    return parser.parse_args()


def main():
    """Carga el modelo, extrae la política y escribe la cabecera de C."""
    args = parse_args()

    print("Loading model:", args.model_path)
    model = PPO.load(args.model_path)
    layers = extract_policy_layers(model.policy.state_dict())
    if not layers:
        raise RuntimeError("No policy_net/action_net layers were found.")

    # Dimensiones extremo a extremo: entradas de la primera capa y salidas de la
    # última. Deben coincidir con la observación (4) y la acción (2) del entorno.
    input_dim = layers[0]["W"].shape[1]
    output_dim = layers[-1]["W"].shape[0]

    with open(args.output, "w") as file:
        file.write("// Auto-generated policy-only MLP\n\n")
        # #pragma once evita la doble inclusión de la cabecera.
        file.write("#pragma once\n\n")
        # OBS_DIM y ACTION_DIM los usa robo03.ino para dimensionar sus buffers.
        file.write(f"#define OBS_DIM {input_dim}\n")
        file.write(f"#define ACTION_DIM {output_dim}\n\n")

        for layer in layers:
            write_array(file, layer["w_name"], layer["W"])
            write_array(file, layer["b_name"], layer["b"])

        file.write(generate_forward(layers))

    print("Generated:", args.output)


if __name__ == "__main__":
    main()
