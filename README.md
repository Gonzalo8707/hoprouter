# HopRouter — General-Purpose AI Agent (Track 1)

A token-efficient routing agent for the AMD Developer Hackathon: ACT II.
For each incoming task, it decides whether to solve it with a **local
model (free, 0 tokens)** or route it to the **best-suited Fireworks AI
model** for that category, aiming to stay above the accuracy threshold
while minimizing total tokens billed.

## Categories handled
Factual knowledge, mathematical reasoning, sentiment classification, text
summarisation, named entity recognition, code debugging, logical
reasoning, code generation.

## How it works
1. `router.py` classifies the prompt into one of the 8 categories using
   cheap keyword/structure heuristics (no model call, so this step is free).
2. Categories that a small local model handles reliably (sentiment, NER,
   short factual questions) are routed **local**.
3. Everything else is routed to the Fireworks model best suited for that
   category (from `ALLOWED_MODELS`), via `fireworks_client.py`.
4. `main.py` reads `/input/tasks.json`, runs each task through the router,
   and writes `/output/results.json`.

## Local development

```bash
cp .env.example .env
# edit .env with your own Fireworks dev key for local testing
pip install -r requirements.txt
export $(cat .env | xargs)
python app/main.py
```

Results will be written to `/output/results.json` (make sure `/input` and
`/output` exist locally, or adjust the paths for testing).

## Build for submission

The judging VM runs `linux/amd64`. If you're building on Apple Silicon
(M1/M2/M3), you MUST use `buildx` with the explicit platform flag:

```bash
docker buildx build --platform linux/amd64 --tag ghcr.io/<your-user>/hoprouter:latest --push .
```

On a standard Intel/AMD machine or GitHub Actions runner, a normal build
already produces the right manifest:

```bash
docker build --tag ghcr.io/<your-user>/hoprouter:latest .
docker push ghcr.io/<your-user>/hoprouter:latest
```

## Environment variables (injected by the harness at evaluation time)

| Variable | Description |
|---|---|
| `FIREWORKS_API_KEY` | Provided by the harness. Never hardcode your own. |
| `FIREWORKS_BASE_URL` | All Fireworks calls must go through this URL. |
| `ALLOWED_MODELS` | Comma-separated list of permitted model IDs, published on launch day. |

## Notes on scoring
- All local inference counts as 0 tokens.
- All calls to Fireworks must go through `FIREWORKS_BASE_URL`; anything
  else is not recorded and scores zero.
- No answers are hardcoded or cached — evaluation uses unseen prompt
  variants.
