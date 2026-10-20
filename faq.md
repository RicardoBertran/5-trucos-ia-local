# Preguntas frecuentes

Dudas habituales y problemas comunes. Si no encuentras la tuya, vuelve a la [guía paso a paso](guia.md).

## ¿Necesito dos tarjetas gráficas para todos los trucos?

No. Solo el truco 1 (repartir el modelo entre GPUs) necesita varias tarjetas. Flash Attention, la caché KV comprimida, MTP y MCP funcionan con una sola GPU.

## ¿Puedo usar estos trucos con una GPU que no sea NVIDIA?

La guía está escrita para GPUs NVIDIA con CUDA, que es lo que usamos en las pruebas. llama.cpp también soporta otras arquitecturas (Vulkan, SYCL, entre otras), y varios de los trucos funcionan en ellas, pero los resultados y algunas opciones pueden variar.

## ¿La caché KV en Q4_0 empeora las respuestas?

Puede haber una pequeña pérdida de precisión, porque los valores se redondean. En la práctica, para la mayoría de modelos y usos, la diferencia es mínima. Si notas que el modelo «olvida» cosas en conversaciones largas, vuelve a F16.

## ¿Por qué mi reparto de GPU no funciona?

Comprueba tres cosas: que llama.cpp detecta las dos tarjetas (aparecen al arrancar), que el orden de `--tensor-split` coincide con el orden de detección, y que has dejado margen de memoria para el sistema. Si una tarjeta se queda sin VRAM, llama.cpp avisará o el sistema se pondrá inestable.

## ¿MTP no aparece en mi versión de llama.cpp?

MTP necesita dos cosas: que el modelo tenga capas MTP y que la versión de llama.cpp sepa usarlas. Si tu modelo no las tiene, no hay forma de activarlo. Si las tiene pero la opción no aparece, actualiza llama.cpp.

## ¿Flash Attention no me ha cambiado nada?

Es normal. Flash Attention optimiza el cálculo de la atención: en algunos modelos y contextos se nota, en otros no. No es un interruptor mágico de velocidad; actívalo porque ayuda a gestionar la memoria y, si de paso acelera, mejor.

## ¿Dónde consigo modelos en GGUF?

En repositorios de la comunidad, como Hugging Face, donde los modelos se publican ya cuantizados en varios formatos (Q4_K_M, Q5_K_M, etc.). Elige el tamaño que quepa en tu memoria con margen.

## ¿Es seguro conectar un servidor MCP con un token de GitHub?

Si usas un token con permisos mínimos y no lo compartes, el riesgo es bajo. Aun así, revisa qué herramientas se han conectado y qué puede hacer el modelo con ellas. Nunca pongas un token real en documentación, capturas o repositorios públicos.
