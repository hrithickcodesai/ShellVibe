# ShellVibe

Natural language → shell command (fine-tuned `Qwen/Qwen2.5-Coder-0.5B`).

## Install

```bash
brew install uv
uv sync
```

## Training

![Weights & Biases training run](assets/wandb.png)

## Download model weights (required)

1) Download a `.pt` from:

<https://drive.google.com/drive/folders/1p7kHSM026taNygr955ef-siSQKdGFCGn?usp=sharing>

2) Put it in `checkpoints/` (e.g. `checkpoints/best_edit_distance.pt`).

## Run inference (natural language → shell command)

### Single instruction

```bash
make inference INSTRUCTION="list all files recursively and show sizes"
```

### Interactive mode

```bash
make inference-interactive
```

## Examples

Always review commands before running them.

```text
Instruction: list all files including hidden
Command:     ls -a

Instruction: what is my current directory
Command:     pwd

Instruction: show git status
Command:     git status

Instruction: count lines in README.md
Command:     wc -l README.md

Instruction: find all python files under src
Command:     find src -name '*.py'

Instruction: extract logs.tar.gz into current directory
Command:     tar xf logs.tar.gz

Instruction: can you provide the command to decompress logs.gz?
Command:     gzip -cd logs.gz > logs.txt
```
