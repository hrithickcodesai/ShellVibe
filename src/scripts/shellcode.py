import argparse
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
    "3b": "gguf-models/qwen_3b_q8.gguf",
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


def build_console() -> Console:
    return Console(theme=CUSTOM_THEME, highlight=False)


def print_banner(console: Console, model_size: str) -> None:
    for line, color in zip(BANNER_LINES, GRADIENT):
        console.print(line, style=color)

    console.print()
    console.print(Text("  speak naturally  ·  get the command", style="subtitle"))
    console.print(
        Text(f"  model · qwen2.5-{model_size}  ·  type exit to quit", style="meta")
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


def predict(nl_instruction: str, model: Llama, max_new_tokens: int = 128) -> str:
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
    return response["choices"][0]["message"]["content"].strip()


def interactive_mode(model: Llama, console: Console, model_size: str) -> None:
    print_banner(console, model_size)

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
            command = predict(nl, model)

        console.print(f"  [command]{command}[/]")
        console.print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ShellVibe — natural language → shell command"
    )
    parser.add_argument("--model_size", choices=["3b"], default="3b")
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--n_gpu_layers",
        type=int,
        default=-1,
        help="-1 = full GPU offload (Metal on macOS), 0 = CPU only",
    )
    args = parser.parse_args()

    console = build_console()
    model = load_model(args.model_size, args.n_gpu_layers, console)

    if args.instruction:
        command = predict(args.instruction, model, args.max_new_tokens)
        console.print(f"  [command]{command}[/]")
    else:
        interactive_mode(model, console, args.model_size)


if __name__ == "__main__":
    main()
