import argparse
from llama_cpp import Llama
from rich.console import Console
from rich.panel import Panel
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

BANNER = r"""
  ███████╗██╗  ██╗███████╗██╗     ██╗      ██████╗ ██████╗ ██████╗ ███████╗
  ██╔════╝██║  ██║██╔════╝██║     ██║     ██╔════╝██╔═══██╗██╔══██╗██╔════╝
  ███████╗███████║█████╗  ██║     ██║     ██║     ██║   ██║██║  ██║█████╗  
  ╚════██║██╔══██║██╔══╝  ██║     ██║     ██║     ██║   ██║██║  ██║██╔══╝  
  ███████║██║  ██║███████╗███████╗███████╗╚██████╗╚██████╔╝██████╔╝███████╗
  ╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚═════╝╚══════╝
"""

CUSTOM_THEME = Theme(
    {
        "banner": "bold green",
        "subtitle": "dim cyan",
        "prompt.arrow": "bold bright_green",
        "prompt.label": "bold white",
        "command": "bold bright_cyan",
        "meta": "dim white",
        "goodbye": "dim green",
    }
)


def build_console() -> Console:
    return Console(theme=CUSTOM_THEME, highlight=False)


def print_banner(console: Console, model_size: str) -> None:
    banner_text = Text(BANNER, style="banner")
    console.print(banner_text)

    subtitle = Text("  natural language  →  shell command", style="subtitle")
    console.print(subtitle)

    meta = Text(
        f"  model: qwen2.5-{model_size}  ·  type 'exit' or Ctrl-C to quit",
        style="meta",
    )
    console.print(meta)
    console.print(Rule(style="dim green"))


def load_model(model_size: str, n_gpu_layers: int, console: Console) -> Llama:
    path = MODEL_PATHS[model_size]
    with console.status(
        f"[bold green]Loading model[/] [dim]{path}[/]...", spinner="dots"
    ):
        model = Llama(
            model_path=path,
            n_ctx=512,
            n_gpu_layers=n_gpu_layers,
            n_threads=8,
            verbose=False,
        )
    console.print("  [bold green]✓[/] Model ready\n")
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
            console.print("\n[goodbye]  Goodbye. Happy hacking.[/]", highlight=False)
            break

        if not nl:
            continue

        if nl.lower() in ("quit", "exit"):
            console.print("[goodbye]  Goodbye. Happy hacking.[/]", highlight=False)
            break

        with console.status("[dim]Generating...[/]", spinner="dots"):
            command = predict(nl, model)

        command_text = Text(f"  {command}", style="command")
        panel = Panel(
            command_text,
            title="[bold white]Command[/]",
            border_style="cyan",
            padding=(0, 1),
        )
        console.print(panel)
        console.print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ShellCode — natural language → shell command"
    )
    parser.add_argument("--model_size", choices=["3b"], default="3b")
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--n_gpu_layers",
        type=int,
        default=0,
        help="-1 = full GPU offload, 0 = CPU only",
    )
    args = parser.parse_args()

    console = build_console()
    model = load_model(args.model_size, args.n_gpu_layers, console)

    if args.instruction:
        command = predict(args.instruction, model, args.max_new_tokens)
        console.print(
            Panel(
                Text(f"  {command}", style="command"),
                title=f"[bold white]{args.instruction}[/]",
                border_style="cyan",
                padding=(0, 1),
            )
        )
    else:
        interactive_mode(model, console, args.model_size)


if __name__ == "__main__":
    main()
