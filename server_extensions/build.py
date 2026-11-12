"""Build the pinned, locally patched native team server. Never pushes an image."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def generate_stubs() -> None:
    output = ROOT / "server_extensions" / "generated"
    output.mkdir(exist_ok=True)
    subprocess.run([
        sys.executable, "-m", "grpc_tools.protoc",
        f"-I{ROOT / 'server_extensions'}",
        f"--python_out={output}", f"--grpc_python_out={output}",
        str(ROOT / "server_extensions" / "rl_bridge.proto"),
    ], check=True)
    # The installed legacy client defines the same wire messages without the
    # additive team fields. Isolate descriptors so both versions can coexist.
    messages = output / "rl_bridge_pb2.py"
    messages.write_text(messages.read_text().replace(
        "_descriptor_pool.Default().AddSerializedFile(",
        "_descriptor_pool.DescriptorPool().AddSerializedFile(",
    ))
    grpc_file = output / "rl_bridge_pb2_grpc.py"
    content = grpc_file.read_text()
    content = content.replace("import rl_bridge_pb2 as rl__bridge__pb2", "from . import rl_bridge_pb2 as rl__bridge__pb2")
    grpc_file.write_text(content)
    (output / "__init__.py").touch()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--start", action="store_true", help="start a separate local container after building")
    parser.add_argument("--image", default="openra-team-server:local")
    args = parser.parse_args()
    generate_stubs()
    if args.generate_only:
        return
    subprocess.run(["docker", "build", "--platform", "linux/arm64", "-f", "server_extensions/Dockerfile", "-t", args.image, "."], cwd=ROOT, check=True)
    if args.start:
        subprocess.run([
            "docker", "run", "-d", "--name", "openra-team-server",
            "-p", "127.0.0.1:9999:9999", args.image,
        ], check=True)
        import grpc
        with grpc.insecure_channel("127.0.0.1:9999") as channel:
            grpc.channel_ready_future(channel).result(timeout=60)
        print("Native team server ready at 127.0.0.1:9999")


if __name__ == "__main__":
    main()
