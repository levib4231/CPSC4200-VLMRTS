"""Compatibility entry point for the OpenRA-RL agent.

OpenRA is now managed by the official OpenRA-RL Docker/gRPC server.
Use `agent.py` directly for the canonical command.
"""

from agent import main


if __name__ == "__main__":
    raise SystemExit(main())
