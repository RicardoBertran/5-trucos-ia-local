# Guía paso a paso: cinco trucos para tu IA local

Esta guía explica, uno a uno, los trucos del tutorial. Cada apartado empieza por el problema que resuelve, sigue con lo que hay que hacer y termina con lo que puedes esperar.

Lo único que necesitas para empezar:

- [llama.cpp](https://github.com/ggml-org/llama.cpp) con CUDA (o el backend que corresponda a tu GPU).
- Un modelo en formato GGUF. En nuestras pruebas usamos Qwen3.8-27B en Q5_K_M.
- Un poco de paciencia para ajustar valores: lo que funciona en nuestro equipo puede variar en el tuyo.

> **Sobre los ejemplos:** todas las rutas (`ruta/a/tu/modelo.gguf`) son de ejemplo. Sustitúyelas por la ruta real de tu modelo, y adapta también el reparto de GPU y el tamaño de contexto a tu equipo.

## 1. Aprovechar varias tarjetas gráficas

### El problema

Un modelo grande ocupa más memoria de la que cabe en una sola tarjeta gráfica. Si no hay sitio, llama.cpp no puede cargarlo completo en la GPU: o falla al arrancar, o recurre a la RAM, que es mucho más lenta.

### La solución

llama.cpp puede repartir el modelo entre varias tarjetas con la opción `--tensor-split`. Los números son pesos relativos: `45,55` significa que la primera GPU se lleva 45 partes del modelo y la segunda 55. El primer número corresponde a la GPU que llama.cpp detecta primero (GPU 0) y el segundo a la siguiente.

En nuestras pruebas usamos un reparto **45/55** entre la RTX 5070 (12 GB) y la RTX 5060 Ti (16 GB):

```bash
llama-cli -m ruta/a/tu/modelo.gguf --tensor-split 45,55
```

### Por qué no repartir al 50/50

Dos razones. La primera: cada tarjeta tiene una memoria distinta, y conviene que la que tiene más VRAM se lleve la parte mayor. La segunda, y más importante: **hay que reservar memoria para Windows**. El sistema operativo, el escritorio y los controladores consumen VRAM aunque no estés usando la GPU. Si llenas la tarjeta hasta el final, el sistema puede quedarse sin memoria y dar problemas.

Por eso el 45/55: la tarjeta de 16 GB se lleva la parte mayor y en las dos dejamos un margen. **Cada ordenador necesita un reparto diferente**: si tus tarjetas son iguales, un 50/50 suele ir bien; si son distintas, da más peso a la que tenga más memoria. Empieza con un reparto aproximado y ajústalo si ves que una tarjeta se queda sin sitio.

### Qué esperar

El modelo carga completo en las GPUs y la generación se ejecuta a velocidad de GPU. El reparto exacto no cambia mucho la velocidad, pero si una tarjeta se queda sin memoria el resultado es muy distinto: errores al arrancar o un rendimiento muy inferior.

## 2. Activar Flash Attention

### El problema

La memoria de la GPU se gasta en dos cosas: el modelo y la caché KV (la verás en el truco 3). Flash Attention es una forma de calcular las atenciones del modelo que necesita menos memoria intermedia y, en muchos casos, la aprovecha mejor.

### Qué es

Flash Attention no cambia lo que el modelo calcula, cambia cómo lo calcula: procesa la atención en bloques para no tener que guardar en memoria matrices enormes. El resultado es prácticamente el mismo, pero con menos VRAM ocupada y, a menudo, más rápido.

### Cómo activarlo

```bash
llama-cli -m ruta/a/tu/modelo.gguf --flash-attn on
```

En algunas versiones de llama.cpp la opción es simplemente `--flash-attn`, sin el `on`. Si la opción no aparece, actualiza llama.cpp.

### Qué esperar

Menos memoria ocupada y, en muchos modelos, algo más de velocidad. **No es un aumento garantizado**: los resultados dependen del modelo, del tamaño de contexto y del equipo. En algunos casos apenas se nota; en otros, bastante. Actívalo y compara tú mismo.

## 3. Comprimir la caché KV

### El problema

Cuando hablas con un modelo, este necesita recordar lo que ha pasado en la conversación para seguir coherente. Esa memoria de trabajo es la **caché KV**: una tabla donde el modelo guarda, para cada token de la conversación, la información que necesita para generar el siguiente.

Piénsalo como las notas que llevas de una conversación larga: si no las anotas, al cabo de un rato pierdes el hilo. El problema es que esas notas ocupan sitio, y mucho: la caché crece con el tamaño del contexto y con el número de capas del modelo.

### La solución

llama.cpp permite guardar la caché KV en un formato comprimido. Por defecto se guarda en **F16** (16 bits por valor, el formato estándar). También puede guardarse en formatos cuantizados como **Q4_0** (4 bits por valor, con un pequeño error de redondeo).

- **F16**: máxima fidelidad, ocupa más.
- **Q4_0**: ocupa mucho menos, con una pérdida de precisión pequeña.

Las opciones `--cache-type-k` y `--cache-type-v` configuran por separado la caché de las claves (K) y la de los valores (V):

```bash
# Caché KV en F16 (el comportamiento por defecto)
llama-cli -m ruta/a/tu/modelo.gguf --ctx-size 16384

# Caché KV comprimida en Q4_0
llama-cli -m ruta/a/tu/modelo.gguf --ctx-size 16384 --cache-type-k Q4_0 --cache-type-v Q4_0
```

### Qué esperar

En nuestro equipo, con el contexto a 16.384 tokens y Flash Attention activado, la ocupación total de VRAM (sumando las dos tarjetas) fue de **unos 23,4 GB con F16** y **unos 22,6 GB con Q4_0**. Son lecturas del monitor de GPU, que incluyen también otros consumos del sistema, así que la diferencia real de la caché es un poco menor.

Con contextos más largos la diferencia crece, porque la caché KV es lo que más crece. Si te sobra VRAM, F16 es la opción más segura; si el contexto te aprieta, Q4_0 te da margen. Prueba ambas con tu modelo: en la mayoría de casos la diferencia de calidad es mínima, pero depende del modelo.

## 4. Acelerar la generación con MTP

### El problema

La generación de texto es secuencial: el modelo produce un token, lo lee y produce el siguiente. Cada token cuesta un paso completo de cálculo.

### La solución

**MTP (Multi-Token Prediction)** es una técnica en la que el modelo intenta predecir varios tokens de golpe y luego comprueba esas predicciones. Si las acepta, esos tokens cuestan mucho menos que generarlos uno a uno. Es como cuando escribes y el autocorrector propone la palabra siguiente: si la propuesta es buena, la aceptas y avanzas sin escribirla.

En las versiones de llama.cpp que lo soportan, MTP se activa con las opciones de decodificación especulativa: `--draft` apunta al propio modelo (las capas MTP van dentro del GGUF) y `--draft-max` limita cuántos tokens se especulan:

```bash
llama-cli -m ruta/a/tu/modelo.gguf --draft ruta/a/tu/modelo.gguf --draft-max 1
```

> **Ojo:** la sintaxis exacta depende de la versión de llama.cpp y del modelo. Comprueba las opciones disponibles con `llama-cli --help` (busca `--draft`) y consulta la documentación del modelo.

### Qué esperar

En nuestra prueba con el mismo modelo (Qwen3.8-27B Q5_K_M) generamos **22,38 tokens por segundo sin MTP** y **57,56 con MTP activado**.

Dos matices importantes:

- Ese resultado sale de un **benchmark de texto repetitivo**, donde las predicciones aciertan a menudo. En tareas reales (conversación, código, redacción) la aceleración será distinta, a veces mucho menor.
- **MTP no funciona con cualquier modelo**: necesita que el modelo tenga capas MTP y que tu versión de llama.cpp sepa usarlas. Si tu modelo no lo soporta, este truco no existe para ti, y no pasa nada: los otros siguen valiendo.

## 5. Conectar herramientas con MCP

### El problema

Un modelo local solo puede hablar: no puede leer tu repositorio, crear issues ni hacer nada fuera de la conversación. **MCP (Model Context Protocol)** es un protocolo estándar para conectar el modelo a herramientas externas. En palabras simples: es un enchufe universal para darle al modelo manos.

### Ejemplo: conectar GitHub

GitHub ofrece un servidor MCP. Conectado, el modelo puede ver tus repositorios, leer archivos o crear issues, siempre dentro de lo que el token de acceso permita.

### Cómo añadirlo desde la interfaz de llama.cpp

1. Arranca el servidor web de llama.cpp (`llama-server`) y abre la interfaz en el navegador.
2. Busca la sección de **MCP** (en las versiones que la incluyen; la ubicación exacta puede variar entre versiones).
3. Añade el servidor de GitHub. Según cómo lo declares, te pedirá el comando de arranque y, normalmente, un **token de acceso personal** de GitHub.
4. Autoriza la conexión cuando la interfaz lo pida.
5. Comprueba la lista de herramientas disponibles: ahí verás qué puede hacer el modelo (por ejemplo, listar repositorios o leer archivos).

Un ejemplo de cómo se suele declarar un servidor MCP, con un token de ejemplo (nunca uses un token real en la documentación):

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "TU_TOKEN_AQUI"
      }
    }
  }
}
```

> El formato exacto de la configuración depende de la versión de llama.cpp. Este ejemplo sirve para entender qué datos pide un servidor MCP: cómo arrancarlo y qué credenciales necesita.

### Precauciones básicas

- **Usa un token con los permisos mínimos** que necesites. Si solo quieres que el modelo lea repositorios, no le des permisos de escritura.
- **No compartas tu token** en capturas, documentación ni chats. Si lo has expuesto, revócalo y crea otro.
- **Revisa qué herramientas se han conectado** antes de usarlas. Un servidor MCP con permisos amplios puede hacer cosas que no esperas.
- El modelo puede equivocarse al usar las herramientas: revisa lo que hace antes de darle permisos sensibles.

## Todo junto: un ejemplo completo

Así se vería una orden con los cuatro trucos de llama.cpp aplicados a la vez (MTP se añade aparte, según la versión):

```bash
llama-cli -m ruta/a/tu/modelo.gguf \
  --tensor-split 45,55 \
  --flash-attn on \
  --cache-type-k Q4_0 --cache-type-v Q4_0 \
  --ctx-size 16384
```

Valores que tienes que adaptar:

- `ruta/a/tu/modelo.gguf`: la ruta real de tu modelo.
- `--tensor-split 45,55`: el reparto que convenga a tus tarjetas, dejando margen para el sistema.
- `--ctx-size 16384`: el tamaño de contexto que necesites.

¿Tienes alguna duda? Mira las [preguntas frecuentes](faq.md).
