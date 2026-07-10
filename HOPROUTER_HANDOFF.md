# HopRouter — Estado del proyecto (Handoff completo)
**AMD Developer Hackathon: ACT II — Track 1 (General-Purpose AI Agent)**
Última actualización: 10 de julio, 2026

---

## 1. Qué es el proyecto

**HopRouter**: agente de IA que resuelve 8 categorías de tareas (factual, matemática,
sentimiento, resumen, NER, debug de código, lógica, generación de código),
decidiendo por tarea si responde con un modelo local (gratis) o llama a
Fireworks AI (paga tokens). Empaquetado en Docker, cumple el contrato exacto
del harness: lee `/input/tasks.json`, escribe `/output/results.json`.

## 2. Links clave

- **Repo GitHub (público):** https://github.com/Gonzalo8707/hoprouter
- **Imagen Docker (pública):** `ghcr.io/gonzalo8707/hoprouter:latest`
- **Página del paquete:** https://github.com/users/Gonzalo8707/packages/container/package/hoprouter
- **Cuenta Fireworks:** app.fireworks.ai (crédito: ~$51, del cupón del hackathon)
- **Notebook AMD (GPU):** notebooks.amd.com/hackathon (8h/día, equipo "team-4284")
- **Deadline del hackathon:** 11 de julio, 12:00 PM hora Chile (GMT-4)

## 3. Modelos permitidos (ALLOWED_MODELS, publicados día del lanzamiento)

```
minimax-m3, kimi-k2p7-code, gemma-4-31b-it, gemma-4-26b-a4b-it, gemma-4-31b-it-nvfp4
```

**IMPORTANTE:** los 3 modelos Gemma dan **404** en el endpoint serverless de
Fireworks — requieren despliegue "on-demand" separado (confirmado por el
organizador Steve en Discord: *"Gemma is allowed, but it's on-demand: deploy
it at app.fireworks.ai/models first (a 404 means 'not deployed', not
'banned')"*). Costo de desplegar Gemma 4 E4B: **~$7/hora, incluso inactivo**.
Decisión tomada: **no perseguir Gemma**, no es necesario para pasar el gate
("*You don't need Gemma to pass the gate*" — cita textual del organizador).
El premio bonus de Gemma ($1,000 extra) se dejó de lado a propósito.

Modelos usados en el router: **`minimax-m3`** (general) y **`kimi-k2p7-code`**
(código), ambos confirmados funcionando.

## 4. Reglas del harness (del Participant Guide PDF oficial)

- Leer `/input/tasks.json` → escribir `/output/results.json` antes de salir
- Exit code 0 en éxito
- Runtime máximo total: 10 minutos
- Máximo 30s por respuesta individual
- Contenedor listo en <60s
- Todas las respuestas en inglés
- Todo llamado a Fireworks DEBE pasar por `FIREWORKS_BASE_URL` (si no, no cuenta)
- No hardcodear modelos — leer de `ALLOWED_MODELS` en runtime
- Imagen máx 10GB, manifest `linux/amd64` obligatorio
- No cachear/hardcodear respuestas (usan variantes nunca vistas)
- Rate limit: 10 submissions/hora

## 5. Scoring (confirmado por el organizador en Discord, 9 julio)

- **Accuracy gate: 80%** — hay exactamente **19 tareas fijas**, así que el
  score siempre es n/19. Necesitas ≥16/19 (84.2%) para aparecer arriba del
  leaderboard. **15/19 = 78.9% ya NO pasa** (es menor a 80%).
- Bajo el gate → no importa cuántos tokens ahorraste, no rankeas.
- Sobre el gate → ranking por MENOS tokens totales (medidos por el proxy de
  Fireworks, usando la key que el harness inyecta, no la tuya).
- Las pruebas que tú corres con tu propia key NO cuentan para el score real.
- Se puede resometer varias veces (10/hora); el leaderboard se actualiza
  cada ~2 minutos.
- **El `docker push` a ghcr.io NO dispara re-evaluación por sí solo** — hay
  que volver a la plataforma lablab.ai y darle **"Submit Project"** de nuevo
  para que se re-evalúe la imagen `:latest` actualizada.

## 6. Resultado de la evaluación real (dato crítico)

Primera submission evaluada — checked **Jul 10, 04:00 GMT-4**:

```
HopRouter: Smart Local/Remote AI Router
ACCURACY_GATE_FAILED — 73.7% (14/19)
submitted Jul 9, 14:50 GMT-4
```

**No pasó el gate del 80%.** Esa evaluación corresponde a la versión que
tenía: NER movido a remoto, sin Gemma, con parseo robusto — PERO todavía
con 2 categorías (sentiment, factual) yendo al modelo local chico
(Qwen2.5-0.5B). Nuestro set de pruebas casero (21/21 con 21 casos
inventados por nosotros) fue demasiado optimista comparado con las
variantes ocultas reales del harness.

**Contexto del leaderboard real (Track 1, 40 submissions):** primer lugar
tiene **0 tokens y 100% accuracy** (alguien logró resolver TODO gratis,
local). Otros equipos en top 5 rondan 1,800–2,700 tokens con 84–100%
accuracy. Confirma que el diseño ideal SÍ es posible, pero requiere un
modelo local mucho más confiable del que usamos, o cero riesgo (todo remoto).

## 7. Cambio de estrategia aplicado (el más reciente, AÚN NO subido/resometido)

Dado que fallamos el gate, la prioridad cambió 100%: **ya no importa ahorrar
tokens, importa pasar el 80%**. Se hicieron estos cambios en el código local
(en `/home/claude/hoprouter`, sandbox de Claude — el usuario tiene que
replicarlos en su copia real):

### `app/router.py`
```python
LOCAL_CAPABLE = set()  # antes: {Category.SENTIMENT, Category.FACTUAL}
```
Ahora **TODAS las 8 categorías van a Fireworks** (remoto). El modelo local
queda solo como red de emergencia en `main.py` (si Fireworks falla del
todo, intenta local antes que devolver un error vacío).

### `app/fireworks_client.py`
- `max_tokens` subido de 280 → **450** (para no arriesgar respuestas cortadas)
- System prompt reescrito para priorizar **completitud** sobre brevedad:
  > "You are a precise assistant. Answer in English... make sure your
  > answer is COMPLETE - do not cut off mid-sentence or mid-structure...
  > never sacrifice correctness or completeness for brevity."

**Este cambio TODAVÍA NO se ha:**
1. Bajado por el usuario (el zip fue generado y compartido, pendiente de
   que el usuario lo baje y reemplace su carpeta local `app/`)
2. Pusheado a GitHub
3. Reconstruido en Docker (`docker buildx build --platform linux/amd64...`)
4. Subido a ghcr.io (`docker push`)
5. **Resometido en la plataforma lablab.ai** (paso obligatorio, el push solo
   no dispara re-evaluación)

## 8. Historial de bugs encontrados y arreglados (cronológico)

1. **404 en TODOS los modelos** — el ID del modelo necesitaba el prefijo
   completo `accounts/fireworks/models/<slug>`, no el nombre corto. Se
   arregló con auto-prefijo en `fireworks_client.py::_resolve_model()`.
   Confirmó que Gemma sigue fallando incluso con el prefijo correcto (ver
   punto 3) — es un problema de despliegue, no de formato de ID.

2. **Modelo local demasiado lento** (100+ segundos por respuesta, sobre el
   límite de 30s) — se arregló con `max_time=22s` + `max_new_tokens` bajo
   en `local_model.py`.

3. **NER perdía entidades** (JSON válido pero incompleto — le faltaba
   fecha o ubicación) — el modelo local de 0.5B no es confiable para
   extracción completa. Se movió NER a remoto (y ahora, con el cambio más
   reciente, TODO está en remoto).

4. **KeyError('content')** al parsear respuestas de Fireworks — algunos
   modelos devuelven la respuesta en formato distinto (bloques
   estructurados, `reasoning_content`). Se arregló con parseo defensivo en
   `_extract_content()`.

5. **~2GB de librerías CUDA innecesarias** en la imagen Docker — el
   `requirements.txt` traía PyTorch con soporte NVIDIA por defecto, sin
   necesidad (todo corre en CPU). Se instala aparte con
   `--index-url https://download.pytorch.org/whl/cpu`.

6. **Crashes de WSL2/Docker Desktop en Windows** (`0xc00000fd`) durante
   build y push — se resolvió con `wsl --update` + `wsl --shutdown` +
   reabrir Docker Desktop. Recurrente, puede volver a pasar.

7. **Secreto expuesto (.env con Personal Access Token)** casi se sube a
   GitHub — bloqueado automáticamente por GitHub Secret Scanning. Se
   arregló: `git rm --cached .env`, se creó `.gitignore`, se **revocó el
   token viejo** y se generó uno nuevo. Lección aprendida y ya resuelta.

8. **Fallo real de accuracy (73.7%, este handoff)** — en investigación /
   recién mitigado con el cambio del punto 7 de esta sección (todo a
   remoto). Pendiente confirmar si esto sí sube el score arriba del 80%.

## 9. Estructura del proyecto

```
hoprouter/
├── app/
│   ├── main.py            → entrypoint: lee tasks.json, escribe results.json
│   ├── router.py           → clasifica categoría + decide local/remoto
│   ├── fireworks_client.py → llama Fireworks, parseo robusto, auto-prefijo modelo
│   ├── local_model.py      → Qwen2.5-0.5B, ahora solo fallback de emergencia
│   └── validators.py       → chequeos de formato (ya no se usa activamente,
│                              LOCAL_CAPABLE vacío significa que casi nunca
│                              se llega a esta validación)
├── eval/
│   ├── eval_tasks.json     → 21 casos de prueba caseros (más generosos que
│   │                          los reales del harness, ojo con sobreconfiar)
│   └── run_eval.py         → script de evaluación local
├── input/tasks.json        → 8 tareas de ejemplo para probar el contenedor
├── Dockerfile               → build con torch CPU-only, modelo pre-descargado
├── .gitignore                → protege .env de subirse
├── .env.example              → plantilla de variables (sin secretos reales)
├── requirements.txt
└── README.md
```

## 10. Variables de entorno (para pruebas locales)

```bash
FIREWORKS_API_KEY=<tu key real, en tu .env local, NUNCA en git>
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
ALLOWED_MODELS=minimax-m3,kimi-k2p7-code,gemma-4-31b-it,gemma-4-26b-a4b-it,gemma-4-31b-it-nvfp4
```

## 11. Comandos de referencia rápida

**Build + push (Windows, Docker Desktop):**
```powershell
docker buildx build --platform linux/amd64 --tag ghcr.io/gonzalo8707/hoprouter:latest --load .
docker push ghcr.io/gonzalo8707/hoprouter:latest
```

**Prueba end-to-end local (sin exponer la key en pantalla):**
```powershell
docker run --rm --env-file .env -v ${PWD}/input:/input -v ${PWD}/output:/output ghcr.io/gonzalo8707/hoprouter:latest
cat output/results.json
```

**Si WSL/Docker crashea (0xc00000fd):**
```powershell
wsl --update
wsl --shutdown
# reabrir Docker Desktop, esperar que la ballena esté lista, reintentar
```

**Si git da error SSL en el notebook AMD:**
```bash
git config --global http.sslVerify false
git pull
git config --global http.sslVerify true
```

## 12. Próximos pasos inmediatos (en orden)

1. ⬜ Bajar el zip con los cambios de emergencia (todo remoto) — pedirle a
   Claude que lo regenere si se perdió, dado que el sandbox se reinicia
   entre sesiones
2. ⬜ Reemplazar `app/router.py` y `app/fireworks_client.py` en la carpeta local
3. ⬜ Commit + push a GitHub
4. ⬜ Rebuild + push de la imagen Docker
5. ⬜ Probar localmente con `docker run` (confirmar que las 8 respuestas
   siguen siendo correctas y completas)
6. ⬜ **Volver a lablab.ai y hacer clic en "Submit Project" de nuevo**
   (paso que se olvida fácil, pero es obligatorio para re-evaluar)
7. ⬜ Esperar ~2-10 min, revisar el leaderboard de nuevo
   (`AMD Judging → Automated Scoring Leaderboard → T1`)
8. ⬜ Si sigue sin pasar 80%, considerar: revisar qué categorías podrían
   estar fallando (no hay visibilidad directa de cuáles de las 19 tareas
   fallaron), reforzar prompts por categoría, o aumentar aún más el
   margen de tokens en categorías con mayor riesgo (lógica, código)

## 13. Cosas ya completadas y no requieren más acción

- ✅ Repo GitHub público con README completo
- ✅ Imagen Docker pública, verificada con `docker pull` sin login
- ✅ `.gitignore` protegiendo secretos (token viejo ya revocado)
- ✅ Pitch deck (PDF) de 7 slides ya generado y subido al formulario
- ✅ Video de presentación ya grabado y subido
- ✅ Formulario de submission (3 pasos) ya completado una vez
- ✅ Cupón de $50 de Fireworks ya canjeado (~$51 de crédito disponible)
- ✅ Acceso GPU del notebook AMD ya activo (8h/día, no es necesario para
  Docker ya que Docker corre en la compu local del usuario, no en el
  notebook — el notebook no tiene Docker instalado)

## 14. Notas de contexto personal (para que Claude en el chat nuevo entienda al usuario)

- Usuario: Gonzalo Aravena Muñiz, técnico electrónico (15 años en Canon),
  ahora data engineer/backend dev, estudiando agentes de IA/MCP
- Prefiere respuestas directas, sin rodeos, ir al grano
- Usa Windows con Docker Desktop + WSL2 (propenso a crashes recurrentes,
  ya se sabe el fix)
- Usa GitHub Desktop (no línea de comandos de git para todo)
- Objetivo explícito: **ganar el hackathon**, no solo participar
