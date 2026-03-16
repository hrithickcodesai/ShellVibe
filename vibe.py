import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

SERVER_URL = "http://127.0.0.1:7070"
DEFAULT_CHECKPOINT = "checkpoints/best_edit_distance.pt"
HISTORY_FILE = Path.home() / ".vibe_history.sh"


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
GRAY = "\033[90m"


def c(text: str, *codes: str) -> str:
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + RESET




def server_is_alive() -> bool:
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=1) as r:
            return r.status == 200
    except Exception:
        return False


def start_server(checkpoint: str) -> None:
    """Spawn vibe_server.py as a background daemon process."""
    script = Path(__file__).parent / "vibe_server.py"
    cmd = [sys.executable, str(script), "--checkpoint", checkpoint]

    subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    print(c("  Starting model server (first-time cold start)…", GRAY))
    for i in range(60):
        time.sleep(1)
        if server_is_alive():
            print(c("  Server ready.                    ", GRAY))
            return
        if i % 5 == 4:
            dots = "." * ((i // 5 + 1) % 4 + 1)
            print(c(f"  Still loading{dots}          ", GRAY), end="\r", flush=True)

    print(c("\n  Server didn't start in time. Try running manually:", YELLOW))
    print(f"    uv run python vibe_server.py --checkpoint {checkpoint}")
    sys.exit(1)


def ensure_server(checkpoint: str) -> None:
    if not server_is_alive():
        start_server(checkpoint)


def generate(instruction: str, max_new_tokens: int) -> str:
    payload = json.dumps(
        {"instruction": instruction, "max_new_tokens": max_new_tokens}
    ).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["command"]


def append_to_history(instruction: str, command: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n# [{timestamp}] vibe: {instruction}\n{command}\n"
    with open(HISTORY_FILE, "a") as f:
        f.write(entry)


def prompt_accept() -> bool:
    options = c("[y]es", BOLD, GREEN) + " / " + c("[n]o", BOLD, RED)
    while True:
        try:
            raw = input(f"\n  Accept? {options}: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if raw in ("y", "yes", ""):
            return True
        if raw in ("n", "no", "q", "quit"):
            return False
        print(c("  Please enter y or n.", DIM))


def run(instruction: str, max_new_tokens: int) -> None:
    print(c("\n  ⚡ ShellVibe", BOLD, CYAN))
    print(c("  ─────────────────────────────", DIM))
    print(c("  Instruction: ", DIM) + instruction)
    print(c("  Generating…\n", GRAY))

    try:
        command = generate(instruction, max_new_tokens)
    except urllib.error.URLError as e:
        print(c(f"\n  Error contacting server: {e}", RED))
        sys.exit(1)

    print(c("  Command:", DIM))
    print(f"\n    {c(command, BOLD, CYAN)}\n")

    if prompt_accept():
        append_to_history(instruction, command)
        print(
            c("\n  ✓ Saved to ", GREEN)
            + c(str(HISTORY_FILE), BOLD)
            + c(" — not executed.", DIM)
        )
        print(f"\n  Run it:\n\n    {c(command, YELLOW)}\n")
    else:
        print(c("\n  ✗ Rejected. Nothing saved.\n", RED))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="vibe: natural language → shell command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python vibe.py list all python files under src
  python vibe.py show disk usage in human readable format
  vibe find large files modified this week
        """,
    )
    parser.add_argument(
        "instruction",
        nargs="+",
        help="Natural language instruction (no quotes needed)",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"Checkpoint to load when starting the server (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        dest="max_tokens",
    )
    args = parser.parse_args()

    instruction = " ".join(args.instruction)
    ensure_server(args.checkpoint)
    run(instruction, args.max_tokens)


if __name__ == "__main__":
    main()
