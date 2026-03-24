import argparse
import platform
import time
from llama_cpp import Llama
from rich.console import Console
from rich.text import Text
from rich.rule import Rule
from rich.theme import Theme

SYSTEM_PROMPT = (
    "You are a helpful assistant that converts natural language instructions "
    "into shell commands. Output only the shell command, nothing else."
)

MODEL_PATHS = {
    "0.5b": "gguf-models/qwen2.5-0.5b-inst-q8_0.gguf",
    "1.5b": "gguf-models/qwen2.5-1.5b-inst-q8_0.gguf",
    "3b": "gguf-models/qwen2.5-3b-inst-q8_0.gguf",
}

BANNER_LINES = [
    r"  ███████╗██╗  ██╗███████╗██╗     ██╗      ██╗   ██╗██╗██████╗ ███████╗",
    r"  ██╔════╝██║  ██║██╔════╝██║     ██║      ██║   ██║██║██╔══██╗██╔════╝",
    r"  ███████╗███████║█████╗  ██║     ██║      ██║   ██║██║██████╔╝█████╗  ",
    r"  ╚════██║██╔══██║██╔══╝  ██║     ██║      ╚██╗ ██╔╝██║██╔══██╗██╔══╝  ",
    r"  ███████║██║  ██║███████╗███████╗███████╗  ╚████╔╝ ██║██████╔╝███████╗",
    r"  ╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝   ╚═══╝  ╚═╝╚═════╝ ╚══════╝",
]

GRADIENT = [
    "bold color(129)",
    "bold color(135)",
    "bold color(141)",
    "bold color(147)",
    "bold color(153)",
    "bold color(159)",
]

CUSTOM_THEME = Theme(
    {
        "subtitle": "bold color(141)",
        "prompt.arrow": "bold color(129)",
        "command": "bold bright_green",
        "meta": "dim color(147)",
        "goodbye": "dim color(141)",
    }
)


def detect_backend() -> tuple[int, str]:
    if platform.system() == "Darwin":
        return -1, "metal"
    return 0, "cpu"


def build_console() -> Console:
    return Console(theme=CUSTOM_THEME, highlight=False)


def print_banner(console: Console, model_size: str, backend: str) -> None:
    for line, color in zip(BANNER_LINES, GRADIENT):
        console.print(line, style=color)

    console.print()
    console.print(Text("  speak naturally  ·  get the command", style="subtitle"))
    console.print(
        Text(
            f"  model · qwen2.5-{model_size}  ·  backend · {backend}  ·  type exit to quit",
            style="meta",
        )
    )
    console.print(Rule(style="color(57)"))


def load_model(model_size: str, n_gpu_layers: int, console: Console) -> Llama:
    path = MODEL_PATHS[model_size]
    with console.status(f"[color(135)]loading[/] [dim]{path}[/]", spinner="dots"):
        model = Llama(
            model_path=path,
            n_ctx=512,
            n_gpu_layers=n_gpu_layers,
            n_threads=8,
            verbose=False,
        )
    console.print("  [bold color(129)]◆[/] [dim]model ready[/]\n")
    return model


def predict(
    nl_instruction: str, model: Llama, max_new_tokens: int = 128
) -> tuple[str, float]:
    start = time.perf_counter()
    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": nl_instruction.strip()},
        ],
        max_tokens=max_new_tokens,
        temperature=0.0,
        repeat_penalty=1.0,
        stop=["<|im_end|>"],
    )
    elapsed = time.perf_counter() - start
    text = response["choices"][0]["message"]["content"].strip()
    completion_tokens = response.get("usage", {}).get("completion_tokens", 1)
    tps = completion_tokens / elapsed if elapsed > 0 else 0.0
    return text, tps


def interactive_mode(
    model: Llama, console: Console, model_size: str, backend: str
) -> None:
    print_banner(console, model_size, backend)

    while True:
        try:
            console.print("[prompt.arrow]❯[/] ", end="")
            nl = input().strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[goodbye]  stay in the vibe ✦[/]", highlight=False)
            break

        if not nl:
            continue

        if nl.lower() in ("quit", "exit"):
            console.print("[goodbye]  stay in the vibe ✦[/]", highlight=False)
            break

        with console.status("[dim]thinking...[/]", spinner="dots"):
            command, tps = predict(nl, model)

        console.print(f"  [command]{command}[/]")
        console.print(f"  [meta]{tps:.1f} tok/s[/]")
        console.print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ShellVibe — natural language → shell command"
    )
    parser.add_argument("--model_size", choices=["0.5b", "1.5b", "3b"], default="3b")
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    n_gpu_layers, backend = detect_backend()
    console = build_console()
    model = load_model(args.model_size, n_gpu_layers, console)

    if args.instruction:
        command, tps = predict(args.instruction, model, args.max_new_tokens)
        console.print(f"  [command]{command}[/]")
        console.print(f"  [meta]{tps:.1f} tok/s[/]")
    else:
        interactive_mode(model, console, args.model_size, backend)


if __name__ == "__main__":
    main()
