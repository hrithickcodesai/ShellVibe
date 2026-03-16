import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types
from pydantic import BaseModel
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
INPUT_CSV = "data/preprocessed/intermediate/tldr_parsed.csv"
OUTPUT_CSV = "data/preprocessed/intermediate/synthetic_single.csv"

client = genai.Client()


class Variation(BaseModel):
    category: str
    query: str
    target_command: str


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
5. Diversity: All 6 variations must use a DIFFERENT concrete value where applicable. No value may repeat.
6. No-argument commands: If the base command takes no arguments (e.g. pmset -g, uptime), all 6 target_commands will be identical — that is correct and expected.
7. Accuracy: target_command must be a valid, directly executable macOS shell command."""


def get_few_shot_prompt(tool_name, english_desc, command):
    return f"""Generate exactly one JSON object for each of the following 6 query style categories:

1. "Lazy": Ultra-short subject/object keywords only. As few words as feels natural.
2. "Power": CLI-flavored phrasing, may reference the tool name or flag vocabulary.
3. "Direct": One plain, clear English sentence. No filler words or politeness.
4. "Noob": Casual, conversational, non-technical. Describes the problem or goal, never the tool.
5. "ChatGPT-style": A polite instruction or question addressed to an AI assistant.
6. "Broken English": Grammatically incorrect, articles and verbs deliberately dropped.

IMPORTANT: Every query must begin with a lowercase letter.

---
### EXAMPLE 1 — command with a file placeholder

Input:
Tool: pbcopy
Description: Place the contents of a specific file in the clipboard
Command: pbcopy < path/to/file

Output:
[
  {{"category": "Lazy", "query": "copy id_rsa.pub", "target_command": "pbcopy < ~/.ssh/id_rsa.pub"}},
  {{"category": "Power", "query": "pbcopy redirect from zshrc", "target_command": "pbcopy < ~/.zshrc"}},
  {{"category": "Direct", "query": "copy the contents of system.log to clipboard", "target_command": "pbcopy < /var/log/system.log"}},
  {{"category": "Noob", "query": "how do i copy all the text inside my notes.txt file so i can paste it somewhere else?", "target_command": "pbcopy < notes.txt"}},
  {{"category": "ChatGPT-style", "query": "give me the command to copy the contents of script.py to the mac clipboard", "target_command": "pbcopy < script.py"}},
  {{"category": "Broken English", "query": "copy resume desktop clipboard", "target_command": "pbcopy < ~/Desktop/resume.pdf"}}
]

Note on Example 1:
- "Noob" query says "my notes.txt" but gives NO path → target uses bare "notes.txt" (current dir), NOT ~/Documents/notes.txt.
- "Broken English" query says "desktop" → ~/Desktop/ is allowed because the query itself said desktop.

---
### EXAMPLE 2 — command with an app/process placeholder

Input:
Tool: pkill
Description: Kill all processes with a specific name
Command: pkill process_name

Output:
[
  {{"category": "Lazy", "query": "kill safari", "target_command": "pkill Safari"}},
  {{"category": "Power", "query": "pkill docker by process name", "target_command": "pkill Docker"}},
  {{"category": "Direct", "query": "force quit the chrome browser from the terminal", "target_command": "pkill Chrome"}},
  {{"category": "Noob", "query": "my spotify is completely frozen, how do i force close it from the terminal?", "target_command": "pkill Spotify"}},
  {{"category": "ChatGPT-style", "query": "give me the terminal command to completely kill the slack application", "target_command": "pkill Slack"}},
  {{"category": "Broken English", "query": "kill discord close process", "target_command": "pkill Discord"}}
]

---
### EXAMPLE 3 — vague query, no specific file mentioned

Input:
Tool: head
Description: Output the first 10 lines of a file
Command: head path/to/file

Output:
[
  {{"category": "Lazy", "query": "head file", "target_command": "head file.txt"}},
  {{"category": "Power", "query": "head output first 10 lines log", "target_command": "head output.log"}},
  {{"category": "Direct", "query": "show the first 10 lines of access.log", "target_command": "head access.log"}},
  {{"category": "Noob", "query": "how do i quickly peek at the top of a text file without fully opening it?", "target_command": "head file.txt"}},
  {{"category": "ChatGPT-style", "query": "what command shows the first few lines of a file in the terminal?", "target_command": "head file.txt"}},
  {{"category": "Broken English", "query": "show top lines file", "target_command": "head data.txt"}}
]

Note on Example 3:
- "Direct" query explicitly names "access.log" → target uses "access.log". All other queries are vague → use simple generic names in the current directory only.

---
### EXAMPLE 4 — numeric / flag placeholder (n, count, size)

Input:
Tool: head
Description: Output the first `n` lines of a file
Command: head -n n path/to/file

Output:
[
  {{"category": "Lazy", "query": "head 20 lines", "target_command": "head -n 20 file.txt"}},
  {{"category": "Power", "query": "head -n flag first 50 lines", "target_command": "head -n 50 output.log"}},
  {{"category": "Direct", "query": "show the first 30 lines of errors.log", "target_command": "head -n 30 errors.log"}},
  {{"category": "Noob", "query": "is there a way to see just the first 15 lines of a file in the terminal?", "target_command": "head -n 15 file.txt"}},
  {{"category": "ChatGPT-style", "query": "how do i print only the first 100 lines of a log file using head?", "target_command": "head -n 100 file.log"}},
  {{"category": "Broken English", "query": "show 25 lines from file", "target_command": "head -n 25 data.txt"}}
]

Note on Example 4:
- Each variation picks a different realistic number for n (20, 50, 30, 15, 100, 25).
- File names are generic because no specific file was mentioned in any query.

---
### EXAMPLE 5 — no-argument command

Input:
Tool: uptime
Description: Show how long the system has been running
Command: uptime

Output:
[
  {{"category": "Lazy", "query": "system uptime", "target_command": "uptime"}},
  {{"category": "Power", "query": "uptime check system runtime", "target_command": "uptime"}},
  {{"category": "Direct", "query": "show how long this mac has been running", "target_command": "uptime"}},
  {{"category": "Noob", "query": "how do i check how long my mac has been on without restarting?", "target_command": "uptime"}},
  {{"category": "ChatGPT-style", "query": "what terminal command tells me how long my system has been running?", "target_command": "uptime"}},
  {{"category": "Broken English", "query": "mac running how long", "target_command": "uptime"}}
]

Note on Example 5:
- All target_commands are identical because uptime takes no arguments. This is correct and expected.

---
### YOUR TASK

Input:
Tool: {tool_name}
Description: {english_desc}
Command: {command}
"""


def generate_variations(
    tool_name, english_desc, command, model, temperature, top_p, top_k
):
    prompt = get_few_shot_prompt(tool_name, english_desc, command)

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=list[Variation],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            ),
        )
        return response.parsed
    except Exception as e:
        print(f"\n[!] Error generating for '{command}': {e}")
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(INPUT_CSV):
        print(f"Error: Could not find {INPUT_CSV}")
        return

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    expanded_dataset = []

    with open(INPUT_CSV, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)
        if args.limit is not None:
            rows = rows[: args.limit]

    print(
        f"Starting generation for {len(rows)} base commands with {args.workers} workers...\n"
    )

    def process_row(row):
        tool = row["Name"]
        desc = row["English"]
        base_cmd = row["Command"]
        variations = generate_variations(
            tool, desc, base_cmd, args.model, args.temp, args.top_p, args.top_k
        )
        results = []
        for var in variations:
            results.append(
                {
                    "tldr_command": base_cmd,
                    "tldr_desc": desc,
                    "category": var.category,
                    "output_command": var.target_command,
                    "input_nl": var.query,
                }
            )
        return results

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_row, row): row for row in rows}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Generating"
        ):
            result = future.result()
            expanded_dataset.extend(result)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as outfile:
        fieldnames = [
            "tldr_command",
            "tldr_desc",
            "category",
            "output_command",
            "input_nl",
        ]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in expanded_dataset:
            writer.writerow(data)

    print(
        f"\nDone! Successfully generated {len(expanded_dataset)} synthetic training examples."
    )
    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
