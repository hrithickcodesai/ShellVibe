import pandas as pd
from sklearn.model_selection import train_test_split


single = pd.read_csv("data/preprocessed/intermediate/synthetic_single.csv")
combo = pd.read_csv("data/preprocessed/intermediate/synthetic_combined.csv")
# single has: tier, input_nl, target_command, category, tldr_command, tldr_desc, model
# combo  has: tier, source_tools, source_commands, combined_command_template,
#             combination_description, is_valid_combination, category, input_nl, target_command

single["source_key"] = single["tldr_command"]
single["dataset"] = "single"

combo["source_key"] = combo["source_commands"]
combo["dataset"] = "combo"

# Rename so both share a common column name before concat
single = single.rename(columns={"target_command": "output_command"})
combo = combo.rename(columns={"target_command": "output_command"})

keep = ["input_nl", "output_command", "category", "source_key", "dataset"]
df = pd.concat([single[keep], combo[keep]], ignore_index=True)

print(f"Total rows      : {len(df)}")
print(f"  from single   : {len(single)}")
print(f"  from combo    : {len(combo)}")
print(f"Unique sources  : {df['source_key'].nunique()}")

# no base command appears in both train and test
unique_sources = df["source_key"].unique().tolist()

sources_train, sources_test = train_test_split(
    unique_sources,
    test_size=0.1,
    random_state=42,
)

train_df = df[df["source_key"].isin(sources_train)].reset_index(drop=True)
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

test_df = df[df["source_key"].isin(sources_test)].reset_index(drop=True)

# sanity check to ensure no leakage of sources between train and test
overlap = set(train_df["source_key"]) & set(test_df["source_key"])
assert len(overlap) == 0, f"Leakage detected! {len(overlap)} sources in both splits"


print(f"Train rows : {len(train_df)}  ({len(train_df) / len(df) * 100:.1f}%)")
print(f"Test  rows : {len(test_df)}   ({len(test_df) / len(df) * 100:.1f}%)")
print(f"Train sources : {train_df['source_key'].nunique()}")
print(f"Test  sources : {test_df['source_key'].nunique()}")
print("Leakage check : PASSED (0 overlapping sources)")

train_df[["input_nl", "output_command"]].to_csv(
    "data/preprocessed/train.csv", index=False
)
test_df[["input_nl", "output_command"]].to_csv(
    "data/preprocessed/test.csv", index=False
)

print("\nSaved → data/preprocessed/train.csv")
print("Saved → data/preprocessed/test.csv")
