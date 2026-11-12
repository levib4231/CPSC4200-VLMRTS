# Native OpenRA team extension

This is a local source patch and Docker build, not a hosted service. It extends the C# game engine's existing `openra.rl.RLBridge` gRPC service. No screenshots, mouse automation, or inferred allies are used.

## Build and run

Requires ARM64 Docker with a running engine, internet access for build dependencies, and the project's Python environment. The pinned upstream runtime currently has only an ARM64 image; the helper explicitly targets `linux/arm64`. Emulation on x86-64 hosts is not validated. No image is pushed.

```bash
.venv/bin/pip install -r server_extensions/requirements.txt
.venv/bin/python server_extensions/build.py --start
.venv/bin/python run_team.py --no-vlm --max-seconds 180
```

The build helper regenerates the bundled Python stubs. The Docker build applies `engine-team.patch` to OpenRA commit `8a5d224223e0498e006a7350a9767a87bd45a708`; the SDK and existing OpenRA-RL runtime are pinned by digest in `Dockerfile`. Debian's native protoc avoids a crash in Grpc.Tools' ARM64 protoc. The matching native C# gRPC plugin is used instead of emulating the entire SDK.

The default container is `openra-team-server`, exposing gRPC only at `127.0.0.1:9999`. It does not replace `openra-rl-server` or its port 8000 API. The stock MCP probe does not inspect this new native endpoint.

For model decisions:

```bash
export RCD_LLM_API_KEY='your-key'  # Or set VLM_ENV_FILE externally.
.venv/bin/python run_team.py \
  --personality helpful --personality selfish \
  --economy-control hybrid --max-seconds 600
```

`--personality` entries correspond to `--players` order, default `Multi1,Multi2`. One coordinator gathers each player's decision, submits one batch, and advances the shared world once. `--no-vlm` uses the personality-specific deterministic policies; it is not evidence of successful model calls.

## Match lifecycle

Each engine container hosts one match. `run_team.py` attaches to that match rather than resetting it. The engine exits after game over, finalizing its replay. For a fresh match, use `docker restart openra-team-server`, wait for startup, then run the coordinator again. The build helper refuses a duplicate container name rather than deleting an existing match.

```bash
docker logs openra-team-server
docker cp openra-team-server:/root/.config/openra/Replays ./runs/team-replays
# Stop without deleting replays:
docker stop openra-team-server
```

## Protocol

`TeamState(StateRequest)` returns `TeamAdvanceResponse` containing a full observation for each configured player. `TeamAdvance(TeamAdvanceRequest)` accepts a repeated `TeamPlayerAction` with a player ID and native `AgentAction` commands. All returned observations have the same world tick. There can be a dispatch tick in addition to requested simulation ticks; clients must use returned ticks, not infer elapsed time.

Additive `GameObservation` fields:

- `controlled_player_id`
- `player_roster`: stable IDs and self/ally/enemy/neutral relationships
- `allied_units` and `allied_buildings`, filtered through the observing player's shroud just like existing enemy observations

The roster contains identities/relationships, not hidden actor positions or opponent economy. Own actors remain separate from visible allied actors. OpenRA's normal shared team vision rules determine visibility.

Control is deliberately centralized: exactly one `rl-agent` bridge owns the game, while additional explicitly allowlisted `dummy` players receive orders through it. This supports two distinct policies in one team, not two independent OpenEnv/WebSocket sessions. `RL_CONTROLLED_PLAYERS` names permitted slots; non-allied, non-playable, or non-dummy extra slots fail validation after world load, and team RPCs return `FAILED_PRECONDITION` with the configuration error.

The included `vlm-team-doubles.oramap` derives from OpenRA's **Doubles** by **Nuke'm Bro.** It locks Multi1/Multi2 into one team and Multi0/Multi3 into the other, with fixed adjacent spawn pairs. The two enemies use native Normal AI.

## Safety and synchronization

- The game thread validates the complete batch before issuing any orders.
- Every actor subject must belong to the requested player.
- Attack targets must be live visible enemies, not allies or neutrals.
- Unknown/unconfigured players and duplicate action batches are rejected.
- Unary operations are serialized per bridge; the world pauses after each team batch.
- Legacy streaming or FastAdvance cannot be mixed with active team control.
- A stale actor/target can invalidate a batch on its dispatch tick. The runner logs that rejection, refreshes observations, and replans without replaying the rejected batch.
- Transport failures/timeouts are ambiguous and stop the runner; they are not automatically retried, avoiding duplicate production/orders.

The endpoint is unauthenticated and intended for trusted local use only. Do not expose port 9999 publicly. The build currently inherits upstream's NuGet.CommandLine 6.12.2 low-severity advisory (NU1901); this patch does not upgrade upstream dependencies.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -v
# Start/restart the team container before these opt-in tests:
OPENRA_TEAM_LIVE_TEST=1 .venv/bin/python -m unittest discover -s tests -p test_live_team.py -v
```

Verified locally on ARM64 Docker:

- Native engine build and protobuf generation.
- Both controlled players deployed MCVs, built bases, and trained armies.
- Equal observation ticks, visible allies, and idle world pause.
- Server-side cross-owner, allied-attack, and unconfigured-slot rejection.
- A deterministic 2v2 reached terminal `lose` for both controlled players at tick **15775**; one helpful and one selfish policy fought two Normal AIs. Stale-target rejections were recovered during the game. This is transport/team integration evidence, not a VLM performance result.

Source patch licensing follows OpenRA's GPL-3.0-or-later; see `COPYING`. Full corresponding source is reproducible from the pinned upstream commit plus this patch.
