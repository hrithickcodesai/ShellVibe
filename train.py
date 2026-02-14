import os
import time
import math
import torch
import wandb
import Levenshtein
from contextlib import nullcontext
from torch.utils.data import DataLoader
from src.dataset import SFTDataset, SFTCollator
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()


config = {
    "model_id": "google/gemma-3-270m",
    "wandb_project": "gemma-bash-sft",
    "use_wandb": True,
    "out_dir": "checkpoints",
    "data_dir": "data",
    "batch_size": 8,
    "max_lr": 2e-5,
    "min_lr_ratio": 0.10,
    "warmup_steps": 100,
    "max_epochs": 3,
    "log_interval": 1,
    "eval_interval": 500,
    "grad_clip": 1.0,
    "weight_decay": 0.01,
    "seed": 1337,
    "wandb_mode": "online",  # set to "online" to enable wandb logging,
}


def setup_device():
    torch.manual_seed(config["seed"])
    torch.set_float32_matmul_precision("high")
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        device = "cuda"
        device_type = "cuda"
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        # dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        dtype = torch.float32
    elif torch.backends.mps.is_available():
        device = "mps"
        device_type = "mps"
        print("Using MPS (Mac Silicon)")
        dtype = torch.float32
    else:
        device = "cpu"
        device_type = "cpu"
        print("Using CPU")
        dtype = torch.float32

    print(f"Using dtype: {dtype}")
    return device, device_type, dtype


def get_lr(step, min_lr, max_lr, warmup_steps, max_steps):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def evaluate(model, dataloader, tokenizer, device, device_type, dtype, pad_id):
    model.eval()
    total_loss = 0.0
    total_edit_dist = 0.0
    num_samples = 0
    num_gen_samples = 0

    examples_to_print = []

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    if model.generation_config:
        model.generation_config.do_sample = False
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=dtype)
        if device_type == "cuda"
        else nullcontext()
    )

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with autocast_ctx:
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
        total_loss += outputs.loss.item()
        num_samples += 1

        prompts_tensors = []
        target_strings = []

        for i in range(len(input_ids)):
            valid_prompt_mask = (labels[i] == -100) & (attention_mask[i] == 1)
            prompt_len = valid_prompt_mask.sum().item()

            valid_target_mask = (labels[i] != -100) & (attention_mask[i] == 1)
            target_len = valid_target_mask.sum().item()

            prompt_ids = input_ids[i, :prompt_len]
            prompts_tensors.append(prompt_ids)

            target_ids = input_ids[i, prompt_len : prompt_len + target_len]
            target_strings.append(
                tokenizer.decode(target_ids, skip_special_tokens=True)
            )

        prompts_reversed = [p.flip(0) for p in prompts_tensors]
        prompt_ids_reversed = pad_sequence(
            prompts_reversed, batch_first=True, padding_value=pad_id
        )
        generation_inputs = prompt_ids_reversed.flip(1).to(device)
        generation_mask = (generation_inputs != pad_id).long()
        print(f"Generating for batch {batch_idx}...")
        with autocast_ctx:
            generated_ids = model.generate(
                input_ids=generation_inputs,
                attention_mask=generation_mask,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=pad_id,
                use_cache=True,
            )

        for i, (full_seq, target_str) in enumerate(zip(generated_ids, target_strings)):
            input_len = generation_inputs[i].size(0)

            new_tokens = full_seq[input_len:]
            generated_str = tokenizer.decode(new_tokens, skip_special_tokens=True)

            dist = Levenshtein.distance(generated_str.strip(), target_str.strip())
            total_edit_dist += dist

            if len(examples_to_print) < 3:
                prompt_str = tokenizer.decode(
                    prompts_tensors[i], skip_special_tokens=True
                )
                examples_to_print.append(
                    {
                        "prompt": prompt_str,
                        "target": target_str,
                        "generated": generated_str,
                    }
                )

        num_gen_samples += len(input_ids)

    tokenizer.padding_side = original_padding_side
    model.train()

    avg_loss = total_loss / len(dataloader)
    avg_edit_dist = total_edit_dist / num_gen_samples if num_gen_samples > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"EVALUATION REPORT (Loss: {avg_loss:.4f} | Edit Dist: {avg_edit_dist:.2f})")
    print("=" * 60)

    for i, ex in enumerate(examples_to_print):
        print(f"\n--- Example {i+1} ---")
        p_text = ex["prompt"].replace("\n", " ")
        print(f"INPUT:  {p_text[:80]}...")
        print(f"TARGET: {ex['target'].strip()}")
        print(f"OUTPUT: {ex['generated'].strip()}")
    print("=" * 60 + "\n")
    return avg_loss, avg_edit_dist


def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint: {path}")


def main():
    device, device_type, dtype = setup_device()
    os.makedirs(config["out_dir"], exist_ok=True)

    if config["use_wandb"]:
        wandb.init(
            project=config["wandb_project"],
            config={
                "model": config["model_id"],
                "batch_size": config["batch_size"],
                "lr": config["max_lr"],
                "dtype": str(dtype),
            },
            mode=config["wandb_mode"],
        )

    print(f"Loading {config['model_id']}...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])

    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        dtype=dtype,
        attn_implementation="sdpa" if device_type == "cuda" else "eager",
    )
    model.to(device)

    if device_type == "cuda":
        print("Compiling model...")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["max_lr"], weight_decay=config["weight_decay"]
    )

    train_dataset = SFTDataset(
        data_path=os.path.join(config["data_dir"], "train.bin"),
        meta_path=os.path.join(config["data_dir"], "train_meta.bin"),
    )
    test_dataset = SFTDataset(
        data_path=os.path.join(config["data_dir"], "test.bin"),
        meta_path=os.path.join(config["data_dir"], "test_meta.bin"),
    )

    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    collator = SFTCollator(pad_token_id=pad_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=os.cpu_count(),
        pin_memory=device_type == "cuda",
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=os.cpu_count(),
        pin_memory=device_type == "cuda",
        persistent_workers=True,
    )

    min_lr = config["max_lr"] * config["min_lr_ratio"]
    num_batches = len(train_loader)
    total_steps = num_batches * config["max_epochs"]
    print(f"Training for {total_steps} steps...")

    use_scaler = dtype == torch.float16 and device_type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=dtype)
        if device_type == "cuda"
        else nullcontext()
    )

    best_val_loss = float("inf")
    best_edit_dist = float("inf")
    step = 0
    t0 = time.time()

    model.train()

    for epoch in range(config["max_epochs"]):
        for batch in train_loader:
            lr = get_lr(
                step, min_lr, config["max_lr"], config["warmup_steps"], total_steps
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with autocast_ctx:
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["grad_clip"]
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["grad_clip"]
                )
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            if device_type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            t0 = time.time()
            tokens_per_sec = input_ids.numel() / dt

            if step % config["log_interval"] == 0:
                print(
                    f"step {step:5d} | loss: {loss.item():.4f} | lr: {lr:.2e} | "
                    f"norm: {grad_norm:.4f} | dt: {dt * 1000:.2f}ms | tok/s: {tokens_per_sec:.0f}"
                )
                if config["use_wandb"]:
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/lr": lr,
                            "train/grad_norm": grad_norm.item(),
                            "train/tokens_per_sec": tokens_per_sec,
                            "step": step,
                        }
                    )

            if step > 0 and step % config["eval_interval"] == 0:
                print("Evaluating...")
                val_loss, val_edit_dist = evaluate(
                    model, val_loader, tokenizer, device, device_type, dtype, pad_id
                )
                print(
                    f"Val Loss: {val_loss:.4f} | Val Edit Distance: {val_edit_dist:.2f}"
                )

                if config["use_wandb"]:
                    wandb.log(
                        {
                            "val/loss": val_loss,
                            "val/edit_distance": val_edit_dist,
                            "step": step,
                        }
                    )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model, os.path.join(config["out_dir"], "best_loss.pt")
                    )

                if val_edit_dist < best_edit_dist:
                    best_edit_dist = val_edit_dist
                    save_checkpoint(
                        model, os.path.join(config["out_dir"], "best_edit_distance.pt")
                    )

            step += 1

    save_checkpoint(model, os.path.join(config["out_dir"], "last.pt"))
    print("Training complete.")

    if config["use_wandb"]:
        wandb.finish()


if __name__ == "__main__":
    main()
