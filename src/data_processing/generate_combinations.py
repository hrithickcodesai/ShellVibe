import json
import argparse
import itertools
import os
import csv
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

from src.data_processing.tiers import TIER1_LIST, COMBO_VARIATIONS, TIER2

load_dotenv()

INPUT_CSV = "data/preprocessed/intermediate/tldr_parsed.csv"
OUTPUT_CSV = "data/preprocessed/intermediate/synthetic_combined.csv"

MODEL = "z-ai/glm-4.7-flash"

BASE_CATEGORIES = ["Lazy", "Power", "Direct", "Noob", "ChatGPT-style", "Broken English"]

ACTIVE_COMBOS: list[str] = ["T1xT1", "T1xT2", "T2xT2"]

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "combination_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "is_valid_combination": {"type": "boolean"},
                "combined_command": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "combination_description": {
                    "anyOf": [{"type": "string"}, {"type": "null"}]
                },
                "variations": {
                    "anyOf": [
                        {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "category": {"type": "string"},
                                    "query": {"type": "string"},
                                    "target_command": {"type": "string"},
                                },
                                "required": ["category", "query", "target_command"],
                                "additionalProperties": False,
                            },
                        },
                        {"type": "null"},
                    ]
                },
            },
            "required": [
                "is_valid_combination",
                "combined_command",
                "combination_description",
                "variations",
            ],
            "additionalProperties": False,
        },
    },
}

SYSTEM_INSTRUCTION = """
You are an expert data generator for a Natural Language to macOS Terminal Command machine learning model.

Your job is to take 2, 3, or 4 simple macOS terminal commands and decide if they form a meaningful combined pipeline, then produce N natural language query variations.

---
VALIDITY RULES -- mark is_valid_combination: false (all other fields null) if ANY apply:
- The combination produces no useful real-world output
- The output of a command is not a useful input to the next
- A real macOS user would never actually chain these together
- The result would be technically broken or nonsensical

---
COMBINATION TECHNIQUES (use whichever fits):
- Pipe:                cmd1 | cmd2 | cmd3
- Sequential:          cmd1 && cmd2
- Redirect:            cmd1 > file.txt  /  cmd1 >> file.txt
- xargs:               cmd1 | xargs cmd2
- Subshell:            cmd2 $(cmd1)

---
QUERY GENERATION RULES (only when is_valid_combination: true):
1. Every query must start with a lowercase letter.
2. Write queries at a natural length for each category -- no padding.
3. Replace ALL placeholders with realistic macOS-specific values.
4. No magic information: target_command may only contain a file/path/app if it appears in the query.
   - Specific file/path mentioned -> use it verbatim.
   - Conceptually specific but vague (e.g. "my zshrc") -> use default (e.g. ~/.zshrc).
   - Fully vague -> use a simple generic name (e.g. file.txt). Never invent ~/Documents/ etc.
5. All variations must use DIFFERENT concrete values -- no repeats.
6. combination_description: one plain English sentence.
7. target_command must be a valid, directly executable macOS shell command.
8. NO COMMAND NAME LEAKAGE: "Direct", "Noob", "ChatGPT-style", and "Broken English" queries
   MUST NOT contain any command or tool names (e.g. awk, grep, bat, cat, sed, find, jq ...).
   Describe only the desired behaviour or outcome in plain language.
   Only "Lazy" and "Power" may reference command/tool names.

---
Category descriptions:
- "Lazy"           -- ultra-short keywords; MAY use command names as terse shorthand
- "Power"          -- CLI-flavored; MAY reference tool names or flag vocabulary
- "Direct"         -- one plain clear English sentence; NO command names allowed
- "Noob"           -- casual, conversational, describes goal not the tool; NO command names allowed
- "ChatGPT-style"  -- polite instruction or question addressed to an AI; NO command names allowed
- "Broken English" -- grammatically incorrect, articles/verbs dropped; NO command names allowed

---
EXAMPLES

### EXAMPLE 1 -- valid (ps | grep)

Input:
Tool 1: ps
  Description: List currently running processes
  Command: ps aux
Tool 2: grep
  Description: Search for a pattern in stdin
  Command: grep search_pattern

Output:
{
  "is_valid_combination": true,
  "combined_command": "ps aux | grep process_name",
  "combination_description": "list all running processes and filter results by name",
  "variations": [
    {"category": "Lazy",           "query": "ps grep chrome",                                                             "target_command": "ps aux | grep Chrome"},
    {"category": "Power",          "query": "list all processes and pipe output to filter by name",                       "target_command": "ps aux | grep Safari"},
    {"category": "Direct",         "query": "show all running processes and filter for docker",                           "target_command": "ps aux | grep Docker"},
    {"category": "Noob",           "query": "how do i check if spotify is actually running in the background right now?", "target_command": "ps aux | grep Spotify"},
    {"category": "ChatGPT-style",  "query": "what is the command to list all running apps and narrow it down to slack?",  "target_command": "ps aux | grep Slack"},
    {"category": "Broken English", "query": "show running discord background",                                            "target_command": "ps aux | grep Discord"}
  ]
}

### EXAMPLE 2 -- invalid

Input:
Tool 1: uptime
  Description: Show how long the system has been running
  Command: uptime
Tool 2: screencapture
  Description: Take a screenshot
  Command: screencapture path/to/file.png

Output:
{"is_valid_combination": false, "combined_command": null, "combination_description": null, "variations": null}
""".strip()


CACHED_PREFIX: list[dict] = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": SYSTEM_INSTRUCTION,
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }
]


def _build_category_list(n_variations: int) -> str:
    categories = [BASE_CATEGORIES[i % 6] for i in range(n_variations)]
    return "\n".join(f'{i + 1}. "{cat}"' for i, cat in enumerate(categories))


def get_combination_prompt(commands: list[dict], n_variations: int) -> str:
    command_block = "\n".join(
        f"Tool {i + 1}: {c['tool']}\n  Description: {c['desc']}\n  Command: {c['cmd']}"
        for i, c in enumerate(commands)
    )
    category_list = _build_category_list(n_variations)
    extra_note = (
        " Produce fresh, distinct examples even when the same category appears more than once."
        if n_variations > 6
        else ""
    )
    return (
        f"{command_block}\n\n"
        f"Produce exactly {n_variations} variations if valid.{extra_note}\n\n"
        f"Categories:\n{category_list}"
    )


def make_client() -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )


MANDATORY_COLUMNS = ["input_nl", "target_command", "category", "tier"]


def load_tldr_rows(csv_path: str) -> dict[str, list[dict]]:
    df = pd.read_csv(csv_path, usecols=["Name", "English", "Command"])
    df = df.rename(columns={"Name": "tool", "English": "desc", "Command": "cmd"})
    return {tool: grp.to_dict("records") for tool, grp in df.groupby("tool")}


def pick_best_row(rows: list[dict]) -> dict:
    candidates = [r for r in rows if "|" not in r["cmd"]]
    if not candidates:
        candidates = rows
    no_sudo = [r for r in candidates if not r["cmd"].strip().startswith("sudo")]
    if no_sudo:
        candidates = no_sudo
    return min(candidates, key=lambda r: len(r["cmd"]))


def resolve_combo(tools: tuple, tool_map: dict[str, list[dict]]) -> list[dict] | None:
    result = []
    for tool in tools:
        if tool not in tool_map:
            return None
        result.append(pick_best_row(tool_map[tool]))
    return result


def write_row(writer: csv.DictWriter, row: dict) -> None:
    """Write a single row; missing mandatory columns raise immediately."""
    for col in MANDATORY_COLUMNS:
        if not row.get(col):
            raise ValueError(f"Missing mandatory column '{col}' in row: {row}")
    writer.writerow(row)


def process_combo(
    client: OpenAI,
    commands: list[dict],
    combo_type: str,
    model: str,
    temperature: float,
    n_variations: int,
) -> list[dict]:
    prompt = get_combination_prompt(commands, n_variations)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=CACHED_PREFIX + [{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format=RESPONSE_FORMAT,
            extra_body={"reasoning": {"effort": "none", "enable": False}},
        )
        content = (response.choices[0].message.content or "").strip()
        result = json.loads(content)
    except Exception as e:
        tqdm.write(f"[!] Error for {[c['tool'] for c in commands]}: {e}")
        return []

    if not result.get("is_valid_combination") or not result.get("variations"):
        return []

    source_tools = "+".join(c["tool"] for c in commands)
    rows = []
    for var in result["variations"]:
        rows.append(
            {
                "tier": combo_type,
                "source_tools": source_tools,
                "source_commands": " | ".join(c["cmd"] for c in commands),
                "combined_command_template": result.get("combined_command", ""),
                "combination_description": result.get("combination_description", ""),
                "is_valid_combination": True,
                "category": var["category"],
                "input_nl": var["query"],
                "target_command": var["target_command"],
            }
        )
    return rows


ALL_COLUMNS = [
    "tier",
    "source_tools",
    "source_commands",
    "combined_command_template",
    "combination_description",
    "is_valid_combination",
    "category",
    "input_nl",
    "target_command",
]


def main():
    parser = argparse.ArgumentParser(
        description="Combination generator -- T1xT1 pairs by default."
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--limit", type=int, default=None, help="Cap total combinations."
    )
    args = parser.parse_args()

    if not os.path.exists(INPUT_CSV):
        print(f"Error: Could not find {INPUT_CSV}")
        return

    client = make_client()
    tool_map = load_tldr_rows(INPUT_CSV)
    print(
        f"Loaded {sum(len(v) for v in tool_map.values())} rows across {len(tool_map)} tools."
    )

    # Build combo list
    pending: list[tuple[tuple, str]] = []
    if "T1xT1" in ACTIVE_COMBOS:
        for pair in itertools.combinations(TIER1_LIST, 2):
            pending.append((pair, "T1xT1"))

    if "T1xT2" in ACTIVE_COMBOS:
        for t1 in TIER1_LIST:
            for t2 in TIER2:
                pending.append(((t1, t2), "T1xT2"))

    if "T2xT2" in ACTIVE_COMBOS:
        for pair in itertools.combinations(TIER2, 2):
            pending.append((pair, "T2xT2"))

    print(f"Generated {len(pending)} total combination")

    # Resolve tools against CSV
    resolved_combos: list[tuple[list[dict], str]] = []
    skipped: list[tuple] = []
    for tools_tuple, combo_type in pending:
        resolved = resolve_combo(tools_tuple, tool_map)
        if resolved is None:
            skipped.append(tools_tuple)
        else:
            resolved_combos.append((resolved, combo_type))

    if skipped:
        print(
            f"Skipped {len(skipped)} combos (tool not in CSV): "
            + ", ".join("+".join(t) for t in skipped[:10])
            + (" ..." if len(skipped) > 10 else "")
        )

    if args.limit:
        resolved_combos = resolved_combos[: args.limit]

    type_counts: dict[str, int] = {}
    for _, combo_type in resolved_combos:
        type_counts[combo_type] = type_counts.get(combo_type, 0) + 1

    print("\nCombo breakdown:")
    for k, v in type_counts.items():
        n_var = COMBO_VARIATIONS[k]
        print(f"  {k:20s}: {v:4d} combos x {n_var} variations = ~{v * n_var} rows")
    print(f"  {'TOTAL':20s}: {len(resolved_combos)} combos")
    print(f"\nStarting generation with {args.workers} workers...\n")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    total_rows = 0
    valid_count = 0

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
        if os.path.getsize(OUTPUT_CSV) == 0:
            writer.writeheader()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_combo,
                    client,
                    commands,
                    combo_type,
                    args.model,
                    args.temp,
                    COMBO_VARIATIONS[combo_type],
                ): combo_type
                for commands, combo_type in resolved_combos
            }

            with tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Generating combinations",
            ) as pbar:
                for future in pbar:
                    rows = future.result()
                    for row in rows:
                        write_row(writer, row)
                        f.flush()
                    if rows:
                        valid_count += 1
                        total_rows += len(rows)
                    pbar.set_postfix(valid=valid_count, rows=total_rows)

    rejected = len(resolved_combos) - valid_count
    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"  Combinations attempted : {len(resolved_combos)}")
    print(f"  Valid (LLM accepted)   : {valid_count}  ({rejected} rejected)")
    print(f"  Rows generated         : {total_rows}")
    print(f"  Saved to               : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
