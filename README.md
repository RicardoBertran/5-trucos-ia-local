# 5 trucos de IA local

Cinco trucos prácticos para sacar más partido a los modelos de inteligencia artificial que ejecutamos en nuestro ordenador con [llama.cpp](https://github.com/ggml-org/llama.cpp).

Este repositorio acompaña a un tutorial de YouTube y sirve como punto de partida para cualquier persona que quiera empezar con la inteligencia artificial local.

## Los cinco trucos

| # | Truco | Qué resuelve |
|---|-------|--------------|
| 1 | [Aprovechar varias tarjetas gráficas](guia.md#1-aprovechar-varias-tarjetas-gráficas) | Cargar un modelo que no cabe en una sola GPU |
| 2 | [Activar Flash Attention](guia.md#2-activar-flash-attention) | Gestionar mejor la memoria de la GPU |
| 3 | [Comprimir la caché KV](guia.md#3-comprimir-la-caché-kv) | Ahorrar VRAM con contextos largos |
| 4 | [Acelerar la generación con MTP](guia.md#4-acelerar-la-generación-con-mtp) | Generar más rápido cuando el modelo lo soporta |
| 5 | [Conectar herramientas con MCP](guia.md#5-conectar-herramientas-con-mcp) | Dar al modelo acceso a herramientas externas, como GitHub |

## Nuestro equipo de pruebas

Los resultados que citamos en la guía salen de este equipo:

- AMD Ryzen 9 5900X
- 64 GB de RAM
- NVIDIA RTX 5070 de 12 GB
- NVIDIA RTX 5060 Ti de 16 GB
- llama.cpp con CUDA
- Qwen3.8-27B en formato GGUF Q5_K_M

No hace falta tener exactamente este hardware para aprovechar los trucos: algunos funcionan con una sola tarjeta gráfica y otros requieren soporte específico del modelo o de la versión de llama.cpp.

## Contenido del repositorio

- [Guía paso a paso](guia.md) — la explicación completa de cada truco, con ejemplos de configuración.
- [Preguntas frecuentes](faq.md) — dudas habituales y problemas comunes.

## Antes de empezar

Los ejemplos de configuración usan rutas de ejemplo (`ruta/a/tu/modelo.gguf`) y valores de nuestro equipo. Sustitúyelos por los tuyos antes de ejecutar nada.
