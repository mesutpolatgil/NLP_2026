import argparse
import json
import os
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer


MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
DATASET_ID = "mlabonne/smoltldr"
TARGET_TOKENS = 50
MAX_PROMPT_LENGTH = 512
MAX_COMPLETION_LENGTH = 96
OUTPUT_ROOT = ROOT / "calisma_ciktilari"


def parse_args():
    parser = argparse.ArgumentParser(
        description="GRPO ve LoRA ile 50 token hedefli ince ayar"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Veri incelemesi yerine gerçek eğitimi başlatır.",
    )
    return parser.parse_args()


def load_and_check_dataset():
    dataset = load_dataset(DATASET_ID)
    train_dataset = dataset["train"].select_columns(["prompt"])

    if not all(isinstance(row["prompt"], str) for row in train_dataset):
        raise TypeError("Veri kümesindeki prompt alanı metin olmalıdır.")

    return dataset, train_dataset


def reward_length(completions, **kwargs):
    rewards = []
    for completion in completions:
        token_count = len(tokenizer.encode(completion, add_special_tokens=False))
        rewards.append(-abs(token_count - TARGET_TOKENS))
    return rewards


def evaluate(model, tokenizer, validation_dataset, run_dir, file_name):
    model.eval()
    records = []

    for index, row in enumerate(validation_dataset, start=1):
        prompt = row["prompt"]
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_PROMPT_LENGTH,
        ).to(model.device)
        input_length = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_COMPLETION_LENGTH,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output[0, input_length:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
        token_count = len(tokenizer.encode(answer, add_special_tokens=False))
        absolute_error = abs(token_count - TARGET_TOKENS)

        records.append(
            {
                "prompt": prompt,
                "answer": answer,
                "token_count": token_count,
                "absolute_error": absolute_error,
                "reward": -absolute_error,
                "hit_generation_limit": len(new_tokens) >= MAX_COMPLETION_LENGTH,
            }
        )
        print(
            f"{file_name} | örnek {index}/10 | "
            f"token: {token_count} | hata: {absolute_error}"
        )

    report = {
        "split": "validation",
        "sample_indices": list(range(10)),
        "target_tokens": TARGET_TOKENS,
        "max_new_tokens": MAX_COMPLETION_LENGTH,
        "do_sample": False,
        "mean_absolute_error": sum(
            record["absolute_error"] for record in records
        ) / len(records),
        "records": records,
    }
    (run_dir / file_name).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Ortalama hata:", report["mean_absolute_error"])


args = parse_args()
OUTPUT_ROOT.mkdir(exist_ok=True)
dataset, train_dataset = load_and_check_dataset()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
lengths = [
    len(tokenizer.encode(row["prompt"], add_special_tokens=False))
    for row in train_dataset
]
print("Veri kaynağı:", DATASET_ID)
print("Eğitim örneği sayısı:", len(train_dataset))
print(f"{MAX_PROMPT_LENGTH} token'ı aşan girdi sayısı:", sum(x > MAX_PROMPT_LENGTH for x in lengths))
print("En uzun girdi:", max(lengths))

if not args.train:
    print("\nVeri incelemesi tamamlandı. Eğitim BAŞLAMADI.")
    raise SystemExit

if not torch.cuda.is_available():
    raise RuntimeError("Bu eğitim için CUDA destekli bir GPU gereklidir.")

set_seed(42)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to("cuda")
model.config.pad_token_id = tokenizer.pad_token_id

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    target_modules="all-linear",
)

run_dir = OUTPUT_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S")
run_dir.mkdir()

training_args = GRPOConfig(
    output_dir=str(run_dir),
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_generations=8,
    max_prompt_length=MAX_PROMPT_LENGTH,
    max_completion_length=MAX_COMPLETION_LENGTH,
    num_train_epochs=1,
    bf16=True,
    optim="adamw_torch",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
    remove_unused_columns=False,
    logging_steps=1,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    dataloader_num_workers=0,
    use_vllm=False,
    seed=42,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    reward_funcs=[reward_length],
    peft_config=lora_config,
)

validation_dataset = dataset["validation"].select(range(10))
evaluate(trainer.model, tokenizer, validation_dataset, run_dir, "before.json")

trainer.model.train()
set_seed(42)
print("\nGRPO EĞİTİMİ BAŞLIYOR.")
trainer.train()

adapter_dir = run_dir / "adapter"
trainer.save_model(str(adapter_dir))
tokenizer.save_pretrained(str(adapter_dir))
trainer.save_state()
(run_dir / "training_log.json").write_text(
    json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

trainer.model.gradient_checkpointing_disable()
trainer.model.config.use_cache = True
evaluate(trainer.model, tokenizer, validation_dataset, run_dir, "after.json")

print("\nEğitim ve önce-sonra değerlendirmesi tamamlandı.")
print("Sonuç klasörü:", run_dir)
