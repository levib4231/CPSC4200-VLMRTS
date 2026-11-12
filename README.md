# OpenRA VLM Agent

A constrained VLM agent for Red Alert using [OpenRA-RL](https://openra-rl.dev/). The game runs in the official headless Docker server; the agent receives OpenRA-RL observations over its WebSocket/MCP API and sends native OpenRA orders through the gRPC bridge.

A deterministic bootstrap controller deploys the MCV, builds a minimal base, and trains a small army. The bundled `vlm-singles.oramap` makes the two player slots hostile; the VLM then controls combat and scouting. `bootstrap` uses that controller and disallows VLM economy orders; `hybrid` runs the controller first and permits validated VLM economy orders when it has no bootstrap task; `vlm` disables bootstrap and permits validated VLM economy orders. Its decision space remains intentionally constrained:

- `ATTACK` — attack a visible enemy actor
- `MOVE` — move selected units to a map cell
- `RETREAT` — move selected units to the friendly base
- `HOLD` — stop selected units
- `BUILD_UNIT`, `BUILD_STRUCTURE`, `DEPLOY` — validated economy orders

### Personalities and economy safety

The `selfish`, `helpful`, and `balanced` profiles affect both prompts and deterministic fallback. Base defense takes priority within 18 cells. Helpful bots assist explicitly observed allies threatened within 12 cells; balanced bots use 8 cells. Selfish bots prioritize their own forces instead.

Spending requires verified cash, item costs, queued commitments, and a 500-credit reserve. Unknown costs fail closed. Production is capped at 32 projected owned units and five per model order; structures are not subject to the unit cap. Explicit `MOVE`, `RETREAT`, and `HOLD` orders may select owned noncombat units; implicit selections exclude them.

### Team play

Team fallback consumes explicit `allies_summary` or `allied_units` actor lists and owner/team metadata. Missing allies are treated as unavailable, not inferred from hidden game state.

Installed package evidence: `GameObservation` in `openra_env/generated/rl_bridge_pb2.py` exposes one `units` list, visible enemies, economy and map, but no allies, teams, player slot, or multi-player commands. `GameState.player_count` reports a count only. `openra_env/models.py` mirrors the same single-player observation shape. The stock server therefore cannot supply these team features. The project now includes a [native server extension](server_extensions/README.md), a fixed-team 2v2 map, and `run_team.py`. One coordinator routes separate policies to two explicitly controlled allied players in the same game; allied observations come from the engine's visibility rules. The existing single-player MCP workflow remains separate.

```bash
.venv/bin/pip install -r server_extensions/requirements.txt
.venv/bin/python server_extensions/build.py --start
.venv/bin/python run_team.py --no-vlm --max-seconds 180
```

A deterministic 2v2 completed with both controlled players losing at tick 15775. Native cross-owner/allied-attack rejection and shared observation ticks were tested. This is not a VLM performance result. See the extension documentation for model mode, lifecycle, security, and opt-in live tests.

The single-player path uses the stock OpenRA-RL server. Team play uses the bundled, reproducible C# engine patch and native gRPC APIs. Neither path uses screenshot or mouse automation.

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

For personality-guided VLM production with deterministic startup:

```bash
.venv/bin/python agent.py --personality helpful --economy-control hybrid --max-seconds 600
```

Use `--economy-control vlm` to give the model deployment and production responsibility from the start. This mode still validates all orders; it does not guarantee a viable opening.

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
--personality NAME     selfish/helpful/balanced prompt guidance
--economy-control MODE bootstrap/hybrid/vlm; default bootstrap
--model NAME           OpenAI-compatible model ID
--base-url URL         OpenAI-compatible chat-completions endpoint
--output-dir PATH      JSONL output directory
--probe-capabilities   inspect live MCP tools/observation fields without starting a match
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

Each run writes `runs/<timestamp>/events.jsonl` containing game states, proposed and sanitized actions, decision source (`vlm`, `vlm_rejected`, or `fallback`), observation counts, order results, model latency, and errors. A model decision is not proof that its order executed: check `action_deferred` and `step.action_result` too. Fallback-completed games are not successful VLM-control tests. The file contains no API credentials. The default bundled map is installed in the local Docker server before reset; newer servers also receive map data at reset.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Opt-in live integration (creates a new single-player match on a running server):

```bash
OPENRA_LIVE_TEST=1 .venv/bin/python -m unittest discover -s tests -p test_live_openra.py -v
```

This checks native deployment, structure construction, and unit production through the validated executor. It does not call a model or verify 2v2 support.
