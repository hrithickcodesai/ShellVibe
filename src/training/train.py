import os
import time
import math
import random
import torch
import wandb
import Levenshtein
from contextlib import nullcontext
from torch.utils.data import DataLoader
from src.training.dataset import SFTDataset, SFTCollator
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()


config = {
    "model_id": "Qwen/Qwen2.5-Coder-0.5B",
    "wandb_project": "qwen-coder-bash-sft",
    "use_wandb": True,
    "out_dir": "checkpoints",
    "data_dir": "data/preprocessed",
    "batch_size": 64,
    "max_lr": 2e-5,
    "min_lr_ratio": 0.10,
    "warmup_steps": 500,
    "max_epochs": 50,
    "log_interval": 1,
    "eval_interval": 150,
    "grad_clip": 1.0,
    "grad_accum_steps": 4,
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
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
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
    num_gen_samples = 0
    total_samples = 0

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

    num_eval_batches = len(dataloader)
    gen_batch_indices = set(
        random.sample(range(num_eval_batches), min(8, num_eval_batches))
    )

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        batch_size = input_ids.size(0)

        with autocast_ctx:
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
        total_loss += outputs.loss.item() * batch_size
        total_samples += batch_size

        if batch_idx not in gen_batch_indices:
            continue

        prompt_only_ids = []
        target_strings = []

        for i in range(input_ids.size(0)):
            prompt_mask = (labels[i] == -100) & (attention_mask[i] == 1)
            target_mask = labels[i] != -100

            prompt_ids = input_ids[i][prompt_mask]  # shape: (prompt_len,)
            target_ids = input_ids[i][target_mask]  # shape: (target_len,)

            prompt_only_ids.append(prompt_ids)
            target_strings.append(
                tokenizer.decode(target_ids, skip_special_tokens=True)
            )

        prompts_reversed = [p.flip(0) for p in prompt_only_ids]
        prompt_ids_reversed = pad_sequence(
            prompts_reversed, batch_first=True, padding_value=pad_id
        )
        generation_inputs = prompt_ids_reversed.flip(1).to(device)
        generation_mask = (generation_inputs != pad_id).long()

        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        with autocast_ctx:
            generated_ids = model.generate(
                input_ids=generation_inputs,
                attention_mask=generation_mask,
                max_new_tokens=128,
                do_sample=False,
                eos_token_id=im_end_id,
                pad_token_id=pad_id,
                use_cache=True,
            )

        for i, (full_seq, target_str) in enumerate(zip(generated_ids, target_strings)):
            input_len = generation_inputs.size(1)
            new_tokens = full_seq[input_len:]
            generated_str = tokenizer.decode(new_tokens, skip_special_tokens=True)

            dist = Levenshtein.distance(generated_str.strip(), target_str.strip())
            total_edit_dist += dist

            prompt_str = tokenizer.decode(prompt_only_ids[i], skip_special_tokens=True)
            examples_to_print.append(
                {
                    "prompt": prompt_str,
                    "target": target_str,
                    "generated": generated_str,
                }
            )

        num_gen_samples += input_ids.size(0)

    tokenizer.padding_side = original_padding_side
    model.train()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_edit_dist = total_edit_dist / num_gen_samples if num_gen_samples > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"EVALUATION REPORT (Loss: {avg_loss:.4f} | Edit Dist: {avg_edit_dist:.2f})")
    print("=" * 60)

    sampled = random.sample(examples_to_print, min(10, len(examples_to_print)))
    for i, ex in enumerate(sampled):
        print(f"\n--- Example {i + 1} ---")
        p_text = ex["prompt"].replace("\n", " ")
        print(f"INPUT:  {p_text[:80]}...")
        print(f"TARGET: {ex['target'].strip()}")
        print(f"OUTPUT: {ex['generated'].strip()}")
    print("=" * 60 + "\n")
    return avg_loss, avg_edit_dist


def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Unwrap torch.compile wrapper if present so state_dict keys are clean
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(raw_model.state_dict(), path)
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
                "max_lr": config["max_lr"],
                "min_lr_ratio": config["min_lr_ratio"],
                "warmup_steps": config["warmup_steps"],
                "max_epochs": config["max_epochs"],
                "weight_decay": config["weight_decay"],
                "grad_clip": config["grad_clip"],
                "grad_accum_steps": config["grad_accum_steps"],
                "eval_interval": config["eval_interval"],
                "dtype": str(dtype),
            },
            mode=config["wandb_mode"],
        )

    print(f"Loading {config['model_id']}...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])

    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        torch_dtype=dtype,
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
    val_dataset = SFTDataset(
        data_path=os.path.join(config["data_dir"], "test.bin"),
        meta_path=os.path.join(config["data_dir"], "test_meta.bin"),
    )

    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    collator = SFTCollator(pad_token_id=pad_id)

    num_workers = os.cpu_count() if device_type == "cuda" else 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=device_type == "cuda",
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=device_type == "cuda",
        persistent_workers=num_workers > 0,
    )

    min_lr = config["max_lr"] * config["min_lr_ratio"]
    num_batches = len(train_loader)
    steps_per_epoch = math.ceil(num_batches / config["grad_accum_steps"])
    total_steps = steps_per_epoch * config["max_epochs"]
    print(
        f"Training for {total_steps} optimizer steps ({config['grad_accum_steps']} grad accum steps, effective batch = {config['batch_size'] * config['grad_accum_steps']})..."
    )

    use_scaler = dtype == torch.float16 and device_type == "cuda"
    scaler = (
        torch.amp.GradScaler("cuda", enabled=use_scaler)
        if device_type == "cuda"
        else None
    )
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=dtype)
        if device_type == "cuda"
        else nullcontext()
    )

    best_val_loss = float("inf")
    best_edit_dist = float("inf")
    step = 0  # counts optimizer steps
    accum_loss = 0.0
    window_tokens = 0
    window_t0 = time.time()

    model.train()

    for epoch in range(config["max_epochs"]):
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # actual window size for the last (possibly partial) window each epoch
            window_start = (batch_idx // config["grad_accum_steps"]) * config[
                "grad_accum_steps"
            ]
            actual_accum = min(config["grad_accum_steps"], num_batches - window_start)

            with autocast_ctx:
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss / actual_accum

            if use_scaler and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item()
            window_tokens += input_ids.numel()

            is_last_micro = (batch_idx + 1) % config[
                "grad_accum_steps"
            ] == 0 or batch_idx == num_batches - 1
            if not is_last_micro:
                continue

            lr = get_lr(
                step, min_lr, config["max_lr"], config["warmup_steps"], total_steps
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            if use_scaler and scaler is not None:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["grad_clip"]
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["grad_clip"]
                )
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            if device_type == "cuda":
                torch.cuda.synchronize()

            dt = time.time() - window_t0
            tokens_per_sec = window_tokens / dt

            if step % config["log_interval"] == 0:
                print(
                    f"step {step:5d} | loss: {accum_loss:.4f} | lr: {lr:.2e} | "
                    f"norm: {grad_norm:.4f} | dt: {dt * 1000:.2f}ms | tok/s: {tokens_per_sec:.0f}"
                )
                if config["use_wandb"]:
                    wandb.log(
                        {
                            "train/loss": accum_loss,
                            "train/lr": lr,
                            "train/grad_norm": grad_norm.item(),
                            "train/tokens_per_sec": tokens_per_sec,
                            "epoch": epoch,
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

            accum_loss = 0.0
            window_tokens = 0
            window_t0 = time.time()
            step += 1

    print("Running final evaluation...")
    val_loss, val_edit_dist = evaluate(
        model, val_loader, tokenizer, device, device_type, dtype, pad_id
    )
    print(
        f"Final Val Loss: {val_loss:.4f} | Final Val Edit Distance: {val_edit_dist:.2f}"
    )
    if config["use_wandb"]:
        wandb.log(
            {"val/loss": val_loss, "val/edit_distance": val_edit_dist, "step": step}
        )

    save_checkpoint(model, os.path.join(config["out_dir"], "last.pt"))
    print("Training complete.")

    if config["use_wandb"]:
        wandb.finish()


if __name__ == "__main__":
    main()
