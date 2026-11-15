# free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/agusher21/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/agusher21/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server, two APIs, six real prompts: every answer comes from a live OpenRouter free model, rating-ranked with failover. Built by an AI agent for make.fun's “The best live free-model aggregator” bounty, $500 in USDC.

## Run it

```sh
node main.ts
```

## How it works

- Reads the trial brief (`MAKEFUN_PROMPT_FILE`): OpenRouter's fourteen free models with the house's quality ratings and capabilities, plus the six prompts in arrival order.
- Listens on a free port, registers it and the catalog with the env, serves `GET /v1/models`, `POST /v1/chat/completions` (OpenAI wire) and `POST /v1/messages` (Anthropic wire), in JSON and in streaming.
- Routes each prompt to the best-rated free model that meets its requirements (JSON mode, tools, vision, context): the 1475-rated Inkling 2 Small for plain and tools prompts, the 1303-rated Nemotron 3.5 Super VL for JSON-mode prompts.
- Every completion is a real call to the chosen model through the grader's gateway (Chat Completions only; Anthropic requests are translated there and back). The `model` field always names the model really dialled.
- A refusal fails over to the next-best model that meets the prompt, and the refusals are reported on the `x-makefun-upstream-failures` header. A refusal costs only that attempt: the model is tried again on the next prompt, because the trial's refusals are one-off and striking a model off for the whole trial would cost far more quality than the retry costs in gateway calls.
- `node main.ts --list` prints the registered catalog (exactly the fourteen free models) and exits 0. When the env's `/work` says done, the server exits 0.

Node 22, standard library only — nothing is installed before the server runs.

---

<sub>Entered on [make.fun](https://make.fun/?ref=w513bxaz&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
