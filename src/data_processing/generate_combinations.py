import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from google import genai
from google.genai import types
from pydantic import BaseModel
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

INPUT_CSV = "data/preprocessed/intermediate/tldr_parsed.csv"
OUTPUT_CSV = "data/preprocessed/intermediate/synthetic_combined.csv"

client = genai.Client()


CURATED_PAIRS = [
    # ── Process inspection ──────────────────────────────────────────────────
    ("ps", "grep"),
    ("ps", "awk"),
    ("ps", "sort"),
    ("ps", "head"),
    ("ps", "wc"),
    ("pgrep", "kill"),
    ("lsof", "grep"),
    ("top", "grep"),
    # ── File search & preview ───────────────────────────────────────────────
    ("find", "grep"),
    ("find", "wc"),
    ("find", "head"),
    ("find", "rm"),
    ("find", "cat"),
    ("find", "sort"),
    ("fd", "grep"),
    ("fd", "wc"),
    ("fd", "head"),
    # ── File content processing ─────────────────────────────────────────────
    ("cat", "grep"),
    ("cat", "sort"),
    ("cat", "uniq"),
    ("cat", "wc"),
    ("cat", "sed"),
    ("cat", "awk"),
    ("cat", "tr"),
    ("cat", "cut"),
    ("cat", "head"),
    ("cat", "tail"),
    ("cat", "less"),
    ("cat", "more"),
    ("cat", "pbcopy"),
    # ── Grep combos ─────────────────────────────────────────────────────────
    ("grep", "wc"),
    ("grep", "sort"),
    ("grep", "awk"),
    ("grep", "cut"),
    ("grep", "head"),
    ("grep", "pbcopy"),
    # ── Directory & disk ────────────────────────────────────────────────────
    ("ls", "grep"),
    ("ls", "sort"),
    ("ls", "wc"),
    ("ls", "head"),
    ("ls", "pbcopy"),
    ("du", "sort"),
    ("du", "grep"),
    ("du", "head"),
    ("df", "grep"),
    ("df", "awk"),
    # ── History & shell ─────────────────────────────────────────────────────
    ("history", "grep"),
    ("history", "tail"),
    ("history", "wc"),
    ("history", "sort"),
    ("history", "uniq"),
    # ── Network ─────────────────────────────────────────────────────────────
    ("netstat", "grep"),
    ("ifconfig", "grep"),
    ("curl", "grep"),
    ("ping", "grep"),
    ("whois", "grep"),
    ("dig", "grep"),
    ("nslookup", "grep"),
    # ── Text utilities ──────────────────────────────────────────────────────
    ("echo", "pbcopy"),
    ("echo", "tr"),
    ("echo", "sed"),
    ("pbpaste", "grep"),
    ("pbpaste", "wc"),
    ("pbpaste", "sort"),
    ("pbpaste", "sed"),
    ("pbpaste", "awk"),
    ("pbpaste", "tr"),
    # ── Tail / log monitoring ───────────────────────────────────────────────
    ("tail", "grep"),
    ("tail", "wc"),
    ("tail", "awk"),
    ("tail", "pbcopy"),
    # ── Clipboard helpers ───────────────────────────────────────────────────
    ("pwd", "pbcopy"),
    ("date", "pbcopy"),
    ("uuidgen", "pbcopy"),
    ("uuidgen", "tr"),
    # ── Git ─────────────────────────────────────────────────────────────────
    ("git", "grep"),
    ("git", "awk"),
    ("git", "wc"),
    ("git", "head"),
    ("git", "tail"),
    ("git", "sort"),
    # ── Brew ────────────────────────────────────────────────────────────────
    ("brew", "grep"),
    ("brew", "wc"),
    ("brew", "sort"),
    # ── Disk utilities ──────────────────────────────────────────────────────
    ("diskutil", "grep"),
    ("mdfind", "head"),
    ("mdfind", "wc"),
    ("mdfind", "grep"),
    # ── Misc useful ─────────────────────────────────────────────────────────
    ("wc", "awk"),
    ("sort", "uniq"),
    ("sort", "head"),
    ("sort", "tail"),
    ("uniq", "sort"),
    ("uniq", "wc"),
    ("curl", "sed"),
    ("curl", "awk"),
    ("curl", "wc"),
    ("ssh", "grep"),
    ("stat", "awk"),
    ("stat", "grep"),
    ("date", "echo"),
    ("whoami", "echo"),
    ("arch", "grep"),
    ("sw_vers", "grep"),
    ("defaults", "grep"),
    ("launchctl", "grep"),
]

CURATED_TRIPLETS = [
    # ── The all-time classics ────────────────────────────────────────────────
    ("cat", "sort", "uniq"),
    ("cat", "grep", "wc"),
    ("cat", "grep", "awk"),
    ("cat", "grep", "head"),
    ("cat", "grep", "tail"),
    ("cat", "grep", "sort"),
    ("cat", "sed", "grep"),
    ("cat", "awk", "sort"),
    ("cat", "sort", "head"),
    ("cat", "cut", "sort"),
    ("cat", "tr", "sort"),
    ("cat", "tr", "wc"),
    # ── Process pipelines ───────────────────────────────────────────────────
    ("ps", "grep", "awk"),
    ("ps", "grep", "wc"),
    ("ps", "sort", "head"),
    ("ps", "awk", "sort"),
    ("lsof", "grep", "awk"),
    ("lsof", "grep", "wc"),
    # ── Find pipelines ──────────────────────────────────────────────────────
    ("find", "grep", "wc"),
    ("find", "grep", "head"),
    ("find", "sort", "head"),
    ("find", "wc", "awk"),
    ("fd", "grep", "wc"),
    ("fd", "sort", "head"),
    # ── Disk & size ─────────────────────────────────────────────────────────
    ("du", "sort", "head"),
    ("du", "grep", "awk"),
    ("ls", "sort", "head"),
    ("ls", "grep", "wc"),
    ("ls", "grep", "sort"),
    # ── History ─────────────────────────────────────────────────────────────
    ("history", "grep", "wc"),
    ("history", "grep", "tail"),
    ("history", "grep", "sort"),
    ("history", "sort", "uniq"),
    # ── Grep combos ─────────────────────────────────────────────────────────
    ("grep", "sort", "uniq"),
    ("grep", "awk", "sort"),
    ("grep", "cut", "sort"),
    ("grep", "grep", "wc"),  # double grep (filter, then filter again)
    # ── Log tailing ─────────────────────────────────────────────────────────
    ("tail", "grep", "awk"),
    ("tail", "grep", "wc"),
    ("tail", "grep", "pbcopy"),
    # ── Network ─────────────────────────────────────────────────────────────
    ("netstat", "grep", "awk"),
    ("netstat", "grep", "wc"),
    ("curl", "grep", "wc"),
    ("curl", "sed", "grep"),
    # ── Clipboard ───────────────────────────────────────────────────────────
    ("pbpaste", "grep", "wc"),
    ("pbpaste", "sort", "uniq"),
    ("pbpaste", "sed", "pbcopy"),
    ("pbpaste", "tr", "pbcopy"),
    ("pbpaste", "grep", "pbcopy"),
    # ── Git ─────────────────────────────────────────────────────────────────
    ("git", "grep", "wc"),
    ("git", "grep", "awk"),
    ("git", "sort", "uniq"),
    ("git", "awk", "sort"),
    # ── Misc ────────────────────────────────────────────────────────────────
    ("sort", "uniq", "wc"),
    ("sort", "uniq", "sort"),
    ("du", "sort", "tail"),
    ("mdfind", "grep", "wc"),
    ("sw_vers", "grep", "awk"),
    ("defaults", "grep", "awk"),
]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class Variation(BaseModel):
    category: str
    query: str
    target_command: str


class CombinationOutput(BaseModel):
    is_valid_combination: bool
    combined_command: Optional[str] = None
    combination_description: Optional[str] = None
    variations: Optional[list[Variation]] = None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """You are an expert data generator for a Natural Language to macOS Terminal Command machine learning model.

Your job is to take 2 or 3 simple macOS terminal commands and produce a meaningful combined pipeline with 6 natural language query variations.

VALIDITY RULES — mark is_valid_combination: false if ANY of these apply:
- The combination produces no useful real-world output
- The output of the first command is not a useful input to the next
- A real macOS user would never actually chain these together
- The result would be technically broken or nonsensical

COMBINATION TECHNIQUES (use whichever fits the commands):
- Pipe:                cmd1 | cmd2
- Sequential:          cmd1 && cmd2
- Redirect to file:    cmd1 > file.txt
- Append to file:      cmd1 >> file.txt
- xargs:               cmd1 | xargs cmd2
- Subshell:            cmd2 $(cmd1)
- Process substitution, background jobs, etc.

QUERY GENERATION RULES (only when is_valid_combination: true):
1. Case: Every query must start with a lowercase letter. Never capitalise the first word.
2. Style: Write queries in a natural length that fits each category. Do NOT pad to hit a word count.
3. Concrete values: Replace ALL placeholders (path/to/file, process_name, n, username, pattern, etc.) with realistic macOS-specific values.
4. No magic information (CRITICAL — read carefully):
   - target_command must ONLY contain a file, path, directory, or app name if that EXACT value appears in the query.
   - Query names a specific file/path/app → use it verbatim in target_command.
   - Query is conceptually specific but vague on path (e.g. "my zshrc") → use the well-known default (e.g. ~/.zshrc).
   - Query is fully vague (no file mentioned) → use a simple generic filename in the current directory (e.g. file.txt, output.log, data.csv). NEVER invent ~/Documents/, ~/Downloads/, ~/Desktop/, or any subdirectory the user did not mention.
   - NEVER introduce a path, directory, or filename that the user did not reference in any way.
5. Diversity: All 6 variations must use DIFFERENT concrete values where applicable. No value may repeat.
6. combination_description: one plain English sentence describing what the full pipeline does.
7. Accuracy: target_command must be a valid, directly executable macOS shell command."""


def get_combination_prompt(commands: list[dict]) -> str:
    command_block = "\n".join(
        [
            f"Tool {i + 1}: {c['tool']}\n  Description: {c['desc']}\n  Command: {c['cmd']}"
            for i, c in enumerate(commands)
        ]
    )
    num = len(commands)

    return f"""You are given {num} macOS terminal commands. Decide if they can be meaningfully combined into a single pipeline or chain that a real user would actually run.

{command_block}

If NOT a valid combination → return: {{"is_valid_combination": false}}

If valid → return a full object with is_valid_combination, combined_command, combination_description, and exactly 6 variations.

Categories:
1. "Lazy"          — ultra-short keywords, no verb, as few words as natural
2. "Power"         — CLI-flavored, may reference tool name or flag vocabulary
3. "Direct"        — one plain clear English sentence, no filler or politeness
4. "Noob"          — casual, conversational, describes goal not the tool
5. "ChatGPT-style" — polite instruction or question addressed to an AI
6. "Broken English"— grammatically incorrect, articles/verbs dropped

Every query must begin with a lowercase letter.

---
### EXAMPLE 1 — ps | grep

Input:
Tool 1: ps
  Description: List currently running processes
  Command: ps aux
Tool 2: grep
  Description: Search for a pattern in stdin
  Command: grep search_pattern

Output:
{{
  "is_valid_combination": true,
  "combined_command": "ps aux | grep process_name",
  "combination_description": "list all running processes and filter results by name",
  "variations": [
    {{"category": "Lazy", "query": "find chrome process", "target_command": "ps aux | grep Chrome"}},
    {{"category": "Power", "query": "ps aux pipe grep for process name", "target_command": "ps aux | grep Safari"}},
    {{"category": "Direct", "query": "show all running processes and filter for docker", "target_command": "ps aux | grep Docker"}},
    {{"category": "Noob", "query": "how do i check if spotify is actually running in the background right now?", "target_command": "ps aux | grep Spotify"}},
    {{"category": "ChatGPT-style", "query": "what command lists all processes and filters for a specific app like slack?", "target_command": "ps aux | grep Slack"}},
    {{"category": "Broken English", "query": "find discord running process", "target_command": "ps aux | grep Discord"}}
  ]
}}

Note: No file paths here — app names come directly from each query. Never invent paths for process-based commands.

---
### EXAMPLE 2 — cat | sort | uniq

Input:
Tool 1: cat
  Description: Print the contents of a file
  Command: cat path/to/file
Tool 2: sort
  Description: Sort lines of a file
  Command: sort path/to/file
Tool 3: uniq
  Description: Remove duplicate lines from sorted input
  Command: uniq

Output:
{{
  "is_valid_combination": true,
  "combined_command": "cat path/to/file | sort | uniq",
  "combination_description": "print a file's contents, sort the lines, and remove duplicates",
  "variations": [
    {{"category": "Lazy", "query": "deduplicate words.txt", "target_command": "cat words.txt | sort | uniq"}},
    {{"category": "Power", "query": "cat pipe sort pipe uniq deduplicate lines", "target_command": "cat names.txt | sort | uniq"}},
    {{"category": "Direct", "query": "remove duplicate lines from ips.txt and sort them", "target_command": "cat ips.txt | sort | uniq"}},
    {{"category": "Noob", "query": "i have a text file with loads of repeated lines, how do i get just the unique ones?", "target_command": "cat file.txt | sort | uniq"}},
    {{"category": "ChatGPT-style", "query": "how do i print only the unique lines from a file using the terminal?", "target_command": "cat data.txt | sort | uniq"}},
    {{"category": "Broken English", "query": "remove duplicate lines file", "target_command": "cat list.txt | sort | uniq"}}
  ]
}}

Note: "deduplicate words.txt" → query names words.txt → use it. "i have a text file" → vague → bare file.txt in current dir. NEVER use ~/Documents/file.txt.

---
### EXAMPLE 3 — du | sort | head  (find largest files)

Input:
Tool 1: du
  Description: Estimate file space usage for a directory
  Command: du -sh path/to/directory
Tool 2: sort
  Description: Sort lines numerically
  Command: sort -rh
Tool 3: head
  Description: Output the first n lines
  Command: head -n n

Output:
{{
  "is_valid_combination": true,
  "combined_command": "du -sh * | sort -rh | head -n 10",
  "combination_description": "find the largest files or directories in the current location",
  "variations": [
    {{"category": "Lazy", "query": "largest files here", "target_command": "du -sh * | sort -rh | head -n 10"}},
    {{"category": "Power", "query": "du sort rh head top 10 biggest", "target_command": "du -sh * | sort -rh | head -n 10"}},
    {{"category": "Direct", "query": "show the 10 largest items in the current directory", "target_command": "du -sh * | sort -rh | head -n 10"}},
    {{"category": "Noob", "query": "my disk is almost full, how do i find what is taking up the most space here?", "target_command": "du -sh * | sort -rh | head -n 10"}},
    {{"category": "ChatGPT-style", "query": "what command shows me the biggest files and folders in my current directory?", "target_command": "du -sh * | sort -rh | head -n 10"}},
    {{"category": "Broken English", "query": "show big files folder sort", "target_command": "du -sh * | sort -rh | head -n 10"}}
  ]
}}

Note: This pipeline produces the same command regardless of query because it acts on the current directory (*) — no user-supplied filename needed.

---
### EXAMPLE 4 — find | grep | wc  (count matching files)

Input:
Tool 1: find
  Description: Find files matching a condition
  Command: find path/to/directory -name 'pattern'
Tool 2: grep
  Description: Filter lines matching a pattern
  Command: grep search_pattern
Tool 3: wc
  Description: Count lines, words, or characters
  Command: wc -l

Output:
{{
  "is_valid_combination": true,
  "combined_command": "find . -name '*.ext' | grep pattern | wc -l",
  "combination_description": "count files matching a name pattern that also contain a specific string",
  "variations": [
    {{"category": "Lazy", "query": "count python test files", "target_command": "find . -name '*.py' | grep test | wc -l"}},
    {{"category": "Power", "query": "find py files grep test wc count", "target_command": "find . -name '*.py' | grep test | wc -l"}},
    {{"category": "Direct", "query": "count how many javascript files contain the word component", "target_command": "find . -name '*.js' | grep component | wc -l"}},
    {{"category": "Noob", "query": "how many config files do i have in this project that mention production?", "target_command": "find . -name '*.conf' | grep production | wc -l"}},
    {{"category": "ChatGPT-style", "query": "how do i count all markdown files that have readme in their name?", "target_command": "find . -name '*.md' | grep README | wc -l"}},
    {{"category": "Broken English", "query": "count log files contain error", "target_command": "find . -name '*.log' | grep error | wc -l"}}
  ]
}}

Note: All queries are vague about location → all paths use "." (current directory). File types come from each query's context, NOT invented out of nowhere.

---
### EXAMPLE 5 — invalid combination

Input:
Tool 1: uptime
  Description: Show how long the system has been running
  Command: uptime
Tool 2: screencapture
  Description: Take a screenshot
  Command: screencapture path/to/file.png

Output:
{{"is_valid_combination": false}}

Note: These two commands are unrelated. Piping uptime into screencapture produces nothing meaningful.

---
### YOUR TASK

{command_block}
"""


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def load_tldr_rows(csv_path: str) -> dict[str, list[dict]]:
    """
    Returns a dict mapping tool name → list of row dicts.
    Each row has keys: tool, desc, cmd
    """
    tool_map: dict[str, list[dict]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Name"]
            entry = {"tool": name, "desc": row["English"], "cmd": row["Command"]}
            tool_map.setdefault(name, []).append(entry)
    return tool_map


def pick_best_row(rows: list[dict]) -> dict:
    """
    Pick the most pipeable/generic command for a tool.
    Preference order:
      1. No pipe already in the command
      2. No sudo prefix (avoid permission noise in combos)
      3. Shortest command string (most generic)
    """
    candidates = [r for r in rows if "|" not in r["cmd"]]
    if not candidates:
        candidates = rows

    no_sudo = [r for r in candidates if not r["cmd"].strip().startswith("sudo")]
    if no_sudo:
        candidates = no_sudo

    return min(candidates, key=lambda r: len(r["cmd"]))


def resolve_combo(
    tools: tuple, tool_map: dict[str, list[dict]]
) -> Optional[list[dict]]:
    """
    Resolve a tuple of tool names to a list of best-row dicts.
    Returns None if any tool is missing from the CSV.
    """
    result = []
    for tool in tools:
        if tool not in tool_map:
            return None
        result.append(pick_best_row(tool_map[tool]))
    return result


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def process_combo(
    commands: list[dict], model: str, temperature: float, top_p: float, top_k: int
) -> list[dict]:
    """Call the LLM for one combination; return list of output rows."""
    prompt = get_combination_prompt(commands)
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=CombinationOutput,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            ),
        )
        result: CombinationOutput = response.parsed
    except Exception as e:
        tqdm.write(f"[!] Error for {[c['tool'] for c in commands]}: {e}")
        return []

    if not result.is_valid_combination or not result.variations:
        return []

    source_tools = "+".join(c["tool"] for c in commands)
    source_cmds = " | ".join(c["cmd"] for c in commands)

    rows = []
    for var in result.variations:
        rows.append(
            {
                "source_tools": source_tools,
                "source_commands": source_cmds,
                "combined_command_template": result.combined_command,
                "combination_description": result.combination_description,
                "category": var.category,
                "input_nl": var.query,
                "output_command": var.target_command,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)  # ← no backslash
    args = parser.parse_args()

    if not os.path.exists(INPUT_CSV):
        print(f"Error: Could not find {INPUT_CSV}")
        return

    tool_map = load_tldr_rows(INPUT_CSV)
    print(
        f"Loaded {sum(len(v) for v in tool_map.values())} rows across {len(tool_map)} tools."
    )

    # Resolve all curated combos against the actual CSV
    all_combos: list[list[dict]] = []
    skipped_missing = []

    for combo in CURATED_PAIRS + CURATED_TRIPLETS:
        resolved = resolve_combo(combo, tool_map)
        if resolved is None:
            skipped_missing.append(combo)
        else:
            all_combos.append(resolved)

    if args.limit:
        all_combos = all_combos[: args.limit]

    if skipped_missing:
        print(
            f"Skipped {len(skipped_missing)} combos (tool not found in CSV): {skipped_missing}"
        )

    print(
        f"Resolved {len(all_combos)} combinations ({len(CURATED_PAIRS)} pairs + {len(CURATED_TRIPLETS)} triplets)."
    )
    print(f"Starting generation with {args.workers} workers...\n")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    results = []
    valid_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_combo, combo, args.model, args.temp, args.top_p, args.top_k
            ): combo
            for combo in all_combos
        }

        with tqdm(
            as_completed(futures), total=len(futures), desc="Generating combinations"
        ) as pbar:
            for future in pbar:
                rows = future.result()
                if rows:
                    results.extend(rows)
                    valid_count += 1
                pbar.set_postfix(valid=valid_count, rows=len(results))

    fieldnames = [
        "source_tools",
        "source_commands",
        "combined_command_template",
        "combination_description",
        "category",
        "input_nl",
        "output_command",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    rejected = len(all_combos) - valid_count
    print(f"\n{'=' * 50}")
    print("Done!")
    print(f"  Combinations attempted : {len(all_combos)}")
    print(f"  Valid (LLM accepted)   : {valid_count}  ({rejected} rejected)")
    print(f"  Rows generated         : {len(results)}")
    print(f"  Saved to               : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
