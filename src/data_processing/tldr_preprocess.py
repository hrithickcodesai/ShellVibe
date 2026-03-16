import os
import glob
import csv

TLDR_OSX_PATH = "data/raw/osx/*.md"
TLDR_COMMON_PATH = "data/raw/common/*.md"
OUTPUT_CSV = "data/preprocessed/intermediate/tldr_parsed.csv"

V1_CORE_TOOLS = {
    # File & Directory Management
    "ls",
    "cd",
    "cp",
    "mv",
    "rm",
    "mkdir",
    "rmdir",
    "pwd",
    "touch",
    "find",
    "fd",
    "tree",
    "file",
    "stat",
    "ln",
    "dirname",
    "basename",
    # Text Viewing & Processing
    "cat",
    "tail",
    "head",
    "less",
    "more",
    "nano",
    "vim",
    "vi",
    "bat",
    "grep",
    "awk",
    "sed",
    "sort",
    "uniq",
    "wc",
    "cut",
    "tr",
    "echo",
    "diff",
    # System & Process Management (Added pkill, pgrep, sudo)
    "ps",
    "top",
    "htop",
    "btop",
    "kill",
    "killall",
    "pkill",
    "pgrep",
    "df",
    "du",
    "uptime",
    "history",
    "clear",
    "date",
    "whoami",
    "chown",
    "chmod",
    "crontab",
    "sudo",
    # Networking & Web
    "ping",
    "curl",
    "wget",
    "ssh",
    "scp",
    "netstat",
    "lsof",
    "ifconfig",
    "dig",
    "nslookup",
    "whois",
    # Archiving
    "tar",
    "zip",
    "unzip",
    "gzip",
    "gunzip",
    # Core Mac Superpowers (Added softwareupdate, launchctl)
    "pbcopy",
    "pbpaste",
    "open",
    "mdfind",
    "screencapture",
    "say",
    "caffeinate",
    "pmset",
    "networksetup",
    "diskutil",
    "system_profiler",
    "defaults",
    "qlmanage",
    "textutil",
    "uuidgen",
    "arch",
    "sw_vers",
    "softwareupdate",
    "launchctl",
    # Essential Workflow & Dev Ops (Added man, alias, source)
    "git",
    "brew",
    "man",
    "alias",
    "source",
}


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
        filename = os.path.basename(filepath)
        tool_name = os.path.splitext(filename)[0]

        if tool_name in V1_CORE_TOOLS:
            print(f"Parsing tool: {tool_name}")
            parsed_examples = parse_tldr_file(filepath, tool_name)
            all_seeds.extend(parsed_examples)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["Name", "English", "Command"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for seed in all_seeds:
            writer.writerow(seed)

    print(f"\nSuccess! Extracted {len(all_seeds)} high-quality v1 examples.")
    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
