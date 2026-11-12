# OpenRA VLM Agent

A constrained VLM agent for Red Alert using [OpenRA-RL](https://openra-rl.dev/). The game runs in the official headless Docker server; the agent receives OpenRA-RL observations over its WebSocket/MCP API and sends native OpenRA orders through the gRPC bridge.

A deterministic bootstrap controller deploys the MCV, builds a minimal base, and trains a small army. The bundled `vlm-singles.oramap` makes the two player slots hostile; the VLM then controls combat and scouting. Its decision space remains intentionally constrained:

- `ATTACK` — attack a visible enemy actor
- `MOVE` — move selected units to a map cell
- `RETREAT` — move selected units to the friendly base
- `HOLD` — stop selected units

No custom OpenRA patch, screenshot automation, or HTTP bridge is used.

## Setup

Prerequisites:

- Python 3.10+
- Docker with a running engine (Docker Desktop or Colima)
- An RCD-compatible API key for VLM mode

```bash
cd starcraft-vlm-agent
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Credentials are read from `RCD_LLM_API_KEY`, `COURSE_LLM_API_KEY`, or `LLM_API_KEY`. Set `VLM_ENV_FILE` to load a separate dotenv file. Keys are never written to this project or run logs.

## Run

Start the official OpenRA-RL server:

```bash
.venv/bin/openra-rl server start
```

Then run the constrained VLM agent against the default Normal AI:

```bash
.venv/bin/python agent.py --max-seconds 600
```

Or start the Docker server automatically:

```bash
.venv/bin/python agent.py --start-server --difficulty normal --max-seconds 600
```

`run_openra.py` remains a compatibility entry point for the same agent command.

For a local deterministic smoke test without an API call:

```bash
.venv/bin/python agent.py --no-vlm --max-seconds 60
```

Useful options:

```text
--server-url URL       OpenRA-RL server, default http://127.0.0.1:8000
--difficulty NAME      beginner/easy/medium/hard/brutal or OpenRA style
--map NAME             OpenRA-RL map, default bundled vlm-singles.oramap
--advance-ticks N      ticks between decisions, 1–50, default 25
--max-turns N          stop after N decisions; 0 means time limit only
--no-bootstrap         disable automatic MCV/base/army setup
--model NAME           OpenAI-compatible model ID
--base-url URL         OpenAI-compatible chat-completions endpoint
--output-dir PATH      JSONL output directory
```

Stop the server when finished:

```bash
.venv/bin/openra-rl server stop
```

## Architecture

```text
agent.py
  ├── OpenRAMCPClient (OpenRA-RL WebSocket/OpenEnv endpoint)
  ├── deterministic bootstrap (deploy/build/train)
  ├── VLM decision client (OpenAI-compatible API)
  └── OpenRA-RL MCP tools
        ├── observations: get_game_state
        ├── orders: attack_target / move_units / stop_units
        └── simulation: advance
              │
              ▼
        OpenRA-RL FastAPI + gRPC bridge + headless OpenRA
```

OpenRA-RL keeps game ticks and agent decisions decoupled. The agent does not block the OpenRA game thread while the VLM is thinking.

## Outputs

Each run writes `runs/<timestamp>/events.jsonl` containing game states, validated actions, order results, model latency, and errors. The file contains no API credentials. The default bundled map is uploaded to the OpenRA-RL server at reset, so it works with a clean Docker image.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```
