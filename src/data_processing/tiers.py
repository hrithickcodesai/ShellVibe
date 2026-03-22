# most common tools — get 12 variations each
TIER1: set[str] = {
    # Unix essentials
    "ls",
    "cat",
    "grep",
    "find",
    "echo",
    "sort",
    "awk",
    "sed",
    "head",
    "tail",
    "xargs",
    "less",
    "tree",  # pager + directory viewer
    "lsof",
    "pkill",
    "pgrep",  # process inspection/control
    "crontab",  # task scheduling
    # Source control & network
    "git",
    "curl",
    "ssh",
    "wget",
    "rsync",
    "openssl",
    "dig",  # crypto + DNS — used constantly
    # Text processing
    "rg",
    "fd",
    "bat",  # ripgrep, fd-find, bat — modern essentials
    "yq",  # YAML processor (pair to jq)
    # macOS
    "open",
    # Editors & multiplexer
    "vim",
    "nvim",
    "tmux",
    # Shells
    "bash",
    "zsh",
    # Language runtimes
    "python",
    "python3",
    "go",
    # Package managers — every dev uses these daily
    "npm",
    "pip",
    # Container / orchestration
    "docker",
    "kubectl",
    "podman",
    # Cloud CLIs — top 2 by usage
    "aws",
    "gcloud",
}


# common utilities — get 8 variations each
TIER2: set[str] = {
    # Text utilities
    "wc",
    "cut",
    "tr",
    "uniq",
    "diff",
    "tee",
    "jq",
    "fzf",
    # File ops
    "mv",
    "cp",
    "rm",
    "mkdir",
    "touch",
    "ln",
    "chmod",
    "chown",
    "tar",
    "zip",
    "unzip",
    "gzip",
    "bzip2",
    "xz",
    "zstd",
    "dd",
    "pv",  # disk copy + pipe progress
    # Process / system monitoring
    "ps",
    "kill",
    "htop",
    "btop",
    "df",
    "du",
    "ping",
    "history",
    "watch",
    "nmap",
    "tcpdump",  # network diagnostics
    "nc",  # netcat
    # Security & crypto
    "gpg",
    "ssh-keygen",
    "scp",
    # macOS clipboard
    "pbcopy",
    "pbpaste",
    # Build tools
    "make",
    "cmake",
    "gradle",
    "mvn",
    "ant",
    "sbt",
    "bazel",
    "ninja",
    # Language runtimes / package managers
    "node",
    "pip3",
    "cargo",
    "rustup",
    "ruby",
    "java",
    "perl",
    "npx",
    "yarn",
    "pnpm",  # JS package managers
    "gem",
    "bundle",  # Ruby
    "composer",  # PHP
    "pipenv",
    "poetry",
    "pyenv",
    "conda",
    "pipx",  # Python env tools
    "nvm",
    "asdf",  # version managers
    # Cloud / infra CLI
    "brew",
    "gh",
    "terraform",
    "helm",
    "ansible",
    "pulumi",
    "vault",
    "consul",
    "packer",
    # k8s ecosystem
    "minikube",
    "kind",
    "k3d",
    "argocd",
    "kustomize",
    "skaffold",
    "k9s",
    # Databases
    "psql",
    "mysql",
    "sqlite3",
    "redis-cli",
    "mongosh",
    "pgcli",
    "mycli",
    "influx",
    # VCS
    "svn",
    "hg",
    # Editors / terminal
    "nano",
    "emacs",
    "screen",
    "zellij",
    # Service managers
    "launchctl",
    # Other
    "vagrant",
    "socat",
}

TIER1_LIST: list[str] = sorted(TIER1)


# to generate combinations
# less common tools that are still important to include for diversity
# but we can get by with fewer variations since they're less critical to get perfect
TIER_VARIATIONS: dict[int, int] = {1: 12, 2: 8, 3: 4}

COMBO_VARIATIONS: dict[str, int] = {
    "T1xT1": 12,
    "T1xT2": 8,
    "T2xT2": 6,
    "T1_triplet": 12,
    "T1xT2_triplet": 10,
    "T1_quad": 10,
}
