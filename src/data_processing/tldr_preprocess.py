import os
import glob
import pandas as pd

TLDR_OSX_PATH = "data/raw/osx/*.md"
TLDR_COMMON_PATH = "data/raw/common/*.md"
OUTPUT_CSV = "data/preprocessed/intermediate/tldr_parsed.csv"


def parse_tldr_file(filepath, tool_name):
    examples = []

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_english = ""

    for line in lines:
        line = line.strip()

        #  examples usually start with "- "
        if line.startswith("- "):
            # Strip the dash and the trailing colon
            current_english = line.replace("- ", "").strip(" :")

        #  actual command is wrapped in backticks
        elif line.startswith("`") and line.endswith("`"):
            command = line.replace("`", "").strip()

            # remove the {{ }} placeholder brackets for clean LLM parsing
            command = command.replace("{{", "").replace("}}", "")

            if current_english and command:
                examples.append(
                    {"Name": tool_name, "English": current_english, "Command": command}
                )
                current_english = ""

    return examples


def main():
    all_seeds = []

    all_files = glob.glob(TLDR_OSX_PATH) + glob.glob(TLDR_COMMON_PATH)

    for filepath in all_files:
        # the name of the file is the tool name, e.g. "ls.md" -> "ls"
        filename = os.path.basename(filepath)
        tool_name = os.path.splitext(filename)[0]

        print(f"Parsing tool: {tool_name}")
        parsed_examples = parse_tldr_file(filepath, tool_name)
        all_seeds.extend(parsed_examples)

    pd.DataFrame(all_seeds).to_csv(OUTPUT_CSV, index=False)

    print(f"\nSuccess! Extracted {len(all_seeds)} high-quality v1 examples.")
    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
