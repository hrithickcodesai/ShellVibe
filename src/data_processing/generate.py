import json
import argparse
import math
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

from tqdm import tqdm
from src.data_processing.tiers import TIER1, TIER2, TIER_VARIATIONS
from dotenv import load_dotenv

load_dotenv()

INPUT_CSV = "data/preprocessed/intermediate/tldr_parsed.csv"
OUTPUT_CSV = "data/preprocessed/intermediate/synthetic_single.csv"

MODEL = "z-ai/glm-4.7-flash"
INPUT_PRICE_PER_M = 0.06
CACHE_READ_PRICE_PER_M = 0.01
OUTPUT_PRICE_PER_M = 0.40

BASE_CATEGORIES = ["Lazy", "Power", "Direct", "Noob", "ChatGPT-style", "Broken English"]


RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "variations_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "variations": {
                    "type": "array",
                    "description": "List of generated query/command variation objects.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "One of the 6 style categories.",
                            },
                            "query": {
                                "type": "string",
                                "description": "Natural-language query starting with lowercase.",
                            },
                            "target_command": {
                                "type": "string",
                                "description": "Valid, directly executable macOS shell command.",
                            },
                        },
                        "required": ["category", "query", "target_command"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["variations"],
            "additionalProperties": False,
        },
    },
}

# prompts
SYSTEM_INSTRUCTION = """You are an expert data generator for a Natural Language to macOS Terminal Command machine learning model.

RULES:
1. Style: Write queries in a natural length that fits the category. Do NOT pad or shrink to hit a word count.
2. Case: Every query MUST start with a lowercase letter. Never capitalise the first word.
3. Concrete values: Replace ALL template placeholders (path/to/file, process_name, n, username, etc.) with realistic macOS-specific values.
4. No magic information (CRITICAL — read carefully):
   - The target_command must ONLY contain a file path, directory, or app name if that EXACT value appears in the query.
   - If the query mentions a specific file/path/app → use it verbatim in target_command.
   - If the query is conceptually specific but vague on path (e.g. "my zsh config") → use the well-known default path (e.g. ~/.zshrc).
   - If the query is fully vague (e.g. "show first lines", "head a file") → use a simple generic filename in the current directory (e.g. file.txt, log.txt). NEVER invent a directory like ~/Documents/.
   - NEVER introduce a path, directory, or filename that the query did not reference in any way.
5. Diversity: All variations must use a DIFFERENT concrete value where applicable. No value may repeat across the entire output.
6. No-argument commands: If the base command takes no arguments (e.g. pmset -g, uptime), all 6 target_commands will be identical — that is correct and expected.
7. Accuracy: target_command must be a valid, directly executable macOS shell command.
8. Output: Return a JSON object with a single "variations" array containing exactly the requested number of objects.

Category descriptions:
- "Lazy": Ultra-short subject/object keywords only. As few words as feels natural.
- "Power": CLI-flavored phrasing, may reference the tool name or flag vocabulary.
- "Direct": One plain, clear English sentence. No filler words or politeness.
- "Noob": Casual, conversational, non-technical. Describes the problem or goal, never the tool.
- "ChatGPT-style": A polite instruction or question addressed to an AI assistant.
- "Broken English": Grammatically incorrect, articles and verbs deliberately dropped.

IMPORTANT: Every query must begin with a lowercase letter.

---
### EXAMPLE 1 — command with a file placeholder

Input:
Tool: pbcopy
Description: Place the contents of a specific file in the clipboard
Command: pbcopy < path/to/file

Output:
{"variations": [
  {"category": "Lazy",          "query": "copy id_rsa.pub",                                                              "target_command": "pbcopy < ~/.ssh/id_rsa.pub"},
  {"category": "Power",         "query": "pbcopy redirect from zshrc",                                                   "target_command": "pbcopy < ~/.zshrc"},
  {"category": "Direct",        "query": "copy the contents of system.log to clipboard",                                 "target_command": "pbcopy < /var/log/system.log"},
  {"category": "Noob",          "query": "how do i copy all the text inside my notes.txt file so i can paste it?",       "target_command": "pbcopy < notes.txt"},
  {"category": "ChatGPT-style", "query": "give me the command to copy the contents of script.py to the mac clipboard",  "target_command": "pbcopy < script.py"},
  {"category": "Broken English","query": "copy resume desktop clipboard",                                                 "target_command": "pbcopy < ~/Desktop/resume.pdf"}
]}

Note on Example 1:
- "Noob" query says "my notes.txt" but gives NO path → target uses bare "notes.txt" (current dir), NOT ~/Documents/notes.txt.
- "Broken English" query says "desktop" → ~/Desktop/ is allowed because the query itself said desktop.

---
### EXAMPLE 2 — no-argument command

Input:
Tool: uptime
Description: Show how long the system has been running
Command: uptime

Output:
{"variations": [
  {"category": "Lazy",          "query": "system uptime",                                                        "target_command": "uptime"},
  {"category": "Power",         "query": "uptime check system runtime",                                          "target_command": "uptime"},
  {"category": "Direct",        "query": "show how long this mac has been running",                              "target_command": "uptime"},
  {"category": "Noob",          "query": "how do i check how long my mac has been on without restarting?",       "target_command": "uptime"},
  {"category": "ChatGPT-style", "query": "what terminal command tells me how long my system has been running?",  "target_command": "uptime"},
  {"category": "Broken English","query": "mac running how long",                                                  "target_command": "uptime"}
]}

Note on Example 2:
- All target_commands are identical because uptime takes no arguments. This is correct and expected.

---
### EXAMPLE 3 — numeric / flag placeholder

Input:
Tool: head
Description: Output the first n lines of a file
Command: head -n n path/to/file

Output:
{"variations": [
  {"category": "Lazy",          "query": "head 20 lines",                                               "target_command": "head -n 20 file.txt"},
  {"category": "Power",         "query": "head -n flag first 50 lines",                                 "target_command": "head -n 50 output.log"},
  {"category": "Direct",        "query": "show the first 30 lines of errors.log",                       "target_command": "head -n 30 errors.log"},
  {"category": "Noob",          "query": "is there a way to see just the first 15 lines of a file?",    "target_command": "head -n 15 file.txt"},
  {"category": "ChatGPT-style", "query": "how do i print only the first 100 lines of a log file?",     "target_command": "head -n 100 file.log"},
  {"category": "Broken English","query": "show 25 lines from file",                                     "target_command": "head -n 25 data.txt"}
]}
"""


# we can cache the entire system prompt
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


def make_client():
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    return client


# build user prompt
def build_user_message(
    tool_name: str, english_desc: str, command: str, n_variations: int
) -> str:
    categories = (BASE_CATEGORIES * math.ceil(n_variations / 6))[:n_variations]
    category_lines = "\n".join(f'{i + 1}. "{cat}"' for i, cat in enumerate(categories))
    extra = (
        " Produce fresh, distinct examples even when the same category appears more than once."
        if n_variations > 6
        else ""
    )
    return (
        f"Generate exactly {n_variations} variation objects{extra}, "
        f"one per category listed below.\n\n"
        f"Categories:\n{category_lines}\n\n"
        f"---\nTool: {tool_name}\n"
        f"Description: {english_desc}\n"
        f"Command: {command}"
    )


def build_messages(
    tool_name: str, english_desc: str, command: str, n_variations: int
) -> list[dict]:
    dynamic = {
        "role": "user",
        "content": build_user_message(tool_name, english_desc, command, n_variations),
    }
    return CACHED_PREFIX + [dynamic]


def call_client(
    client: OpenAI,
    messages: list[dict],
    model: str,
    temperature: float,
):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format=RESPONSE_FORMAT,
        extra_body={
            "reasoning": {
                "effort": "none",
                "enable": False,
            }
        },
    )


def generate_variations(
    client: OpenAI,
    tool_name: str,
    english_desc: str,
    command: str,
    model: str,
    temperature: float,
    n_variations: int,
):
    messages = build_messages(tool_name, english_desc, command, n_variations)
    response = call_client(client, messages, model, temperature)
    content = (response.choices[0].message.content or "").strip()
    try:
        variations = json.loads(content)["variations"]
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing response for tool '{tool_name}': {e}")
        print("Raw content:")
        print(content)
        variations = []

    return variations, response.usage


def process_row(row: dict, client: OpenAI, model: str, temperature: float):
    tool = row["Name"]
    desc = row["English"]
    base_cmd = row["Command"]
    tier = 1 if tool in TIER1 else 2 if tool in TIER2 else 3

    variations, usage = generate_variations(
        client, tool, desc, base_cmd, model, temperature, TIER_VARIATIONS[tier]
    )
    results = [
        {
            "tldr_command": base_cmd,
            "tldr_desc": desc,
            "category": var["category"],
            "target_command": var["target_command"],
            "input_nl": var["query"],
            "tier": tier,
            "model": model,
        }
        for var in variations
    ]
    return results, usage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(INPUT_CSV):
        print(f"Error: Could not find {INPUT_CSV}")
        return

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    rows = pd.read_csv(INPUT_CSV).to_dict("records")
    if args.limit is not None:
        rows = rows[: args.limit]

    already_done: set[str] = set()
    output_exists = os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0
    if output_exists:
        try:
            already_done = set(
                pd.read_csv(OUTPUT_CSV)["tldr_command"].dropna().unique()
            )
        except Exception as e:
            print(f"Warning: could not read existing output ({e}); starting fresh.")
            output_exists = False

    rows_to_process = [r for r in rows if r["Command"] not in already_done]

    print(f"Model  : {args.model}")
    print(
        f"Pricing: ${INPUT_PRICE_PER_M}/M input | ${CACHE_READ_PRICE_PER_M}/M cached | ${OUTPUT_PRICE_PER_M}/M output"
    )
    print(
        f"Rows   : {len(rows):,} total  |  "
        f"{len(already_done):,} already done  |  "
        f"{len(rows_to_process):,} to process  |  "
        f"Workers: {args.workers}\n"
    )

    if not rows_to_process:
        print("Nothing to process — all rows are already in the output CSV.")
        return

    client = make_client()

    if not output_exists:
        pd.DataFrame(
            columns=[
                "tier",
                "input_nl",
                "target_command",
                "category",
                "tldr_command",
                "tldr_desc",
                "model",
            ]
        ).to_csv(OUTPUT_CSV, index=False)

    total_written = 0
    COLUMNS = [
        "tier",
        "input_nl",
        "target_command",
        "category",
        "tldr_command",
        "tldr_desc",
        "model",
    ]

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_row, row, client, args.model, args.temp): row
            for row in rows_to_process
        }
        with tqdm(as_completed(futures), total=len(futures), desc="Generating") as pbar:
            for future in pbar:
                result, usage = future.result()
                if result:
                    df_chunk = pd.DataFrame(result)
                    df_chunk = df_chunk[COLUMNS]
                    df_chunk.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
                    total_written += len(result)

    print(f"\nDone! {total_written:,} synthetic examples written to {OUTPUT_CSV}")
    print("\nToken usage:")


if __name__ == "__main__":
    main()
