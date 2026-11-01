"""Course 12.5: real GRPO + LoRA with token reward and held-out evaluation.
Length control only: this is not evidence of improved reasoning or factual accuracy.
"""
from pathlib import Path
import argparse, os, json, time, statistics
HERE = Path(__file__).resolve().parent
WORK = HERE.parent.parent / "work"
os.environ.setdefault("HF_HOME", str(WORK / "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, TrainerCallback
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DATA = "mlabonne/smoltldr"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--device", choices=["auto","cuda","cpu"], default="auto")
    p.add_argument("--train-samples", type=int, default=256)
    p.add_argument("--eval-samples", type=int, default=24)
    p.add_argument("--target-tokens", type=int, default=50)
    a = p.parse_args()
    device = "cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA destekli PyTorch gerekiyor.")
    use_gpu = device == "cuda"
    dtype = torch.bfloat16 if use_gpu else torch.float32
    generations = 4 if use_gpu else 2
    set_seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    run = HERE / "sonuclar"
    run.mkdir(exist_ok=True)
    def save(name, obj):
        (run/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    start = time.time()
    print("Device:", torch.cuda.get_device_name(0) if use_gpu else "CPU", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    tok.padding_side = tok.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype,
        attn_implementation="sdpa").to(device)
    model.config.pad_token_id = tok.pad_token_id
    print("Model loaded; loading dataset.", flush=True)
    data = load_dataset(DATA, split="train")
    seen, rows = set(), []
    for r in data.shuffle(seed=42):
        if r["prompt"] not in seen:
            seen.add(r["prompt"])
            rows.append({"prompt":r["prompt"]})
        if len(rows) >= a.train_samples + a.eval_samples:
            break
    assert len(rows) == a.train_samples + a.eval_samples
    train_rows, test_rows = rows[:a.train_samples], rows[a.train_samples:]
    assert not ({r["prompt"] for r in train_rows} & {r["prompt"] for r in test_rows})
    save("veri_ayrimi.json", {"dataset":DATA,"fingerprint":data._fingerprint,
        "seed":42,"train":train_rows,"test":test_rows})
    def token_length_reward(completions, **kwargs):
        texts = [c if isinstance(c,str) else "".join(m.get("content","") for m in c)
            for c in completions]
        return [-float(abs(a.target_tokens-len(ids))) for ids in
            tok(texts, add_special_tokens=False)["input_ids"]]
    for s in ["Hello.", "A short sentence about a cat.", ""]:
        assert token_length_reward([s]) == [-float(abs(a.target_tokens-len(tok.encode(s,add_special_tokens=False))))]
    def evaluate(m, name):
        m.eval()
        records = []
        for i, row in enumerate(test_rows):
            set_seed(1000+i)
            enc = tok(row["prompt"],return_tensors="pt",truncation=True,max_length=256).to(device)
            with torch.inference_mode():
                ids = m.generate(**enc,max_new_tokens=96,do_sample=True,
                    temperature=0.9,top_p=1.0,top_k=0,
                    pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
            new = ids[0,enc["input_ids"].shape[1]:]
            text = tok.decode(new,skip_special_tokens=True)
            n = len(tok.encode(text,add_special_tokens=False))
            records.append({"index":i,"prompt":row["prompt"],"output":text,"token_count":n,
                "absolute_error":abs(n-a.target_tokens),"reward":-abs(n-a.target_tokens),
                "hit_generation_limit":len(new)>=96})
            print(name,i+1,"tokens",n,flush=True)
        summary = {"samples":len(records),
            "mean_tokens":statistics.mean(r["token_count"] for r in records),
            "mean_absolute_error":statistics.mean(r["absolute_error"] for r in records),
            "mean_reward":statistics.mean(r["reward"] for r in records),
            "within_5_tokens_rate":statistics.mean(r["absolute_error"]<=5 for r in records),
            "generation_limit_rate":statistics.mean(r["hit_generation_limit"] for r in records)}
        save(name+".json",{"summary":summary,"records":records})
        print(name,summary,flush=True)
        return summary
    before = evaluate(model,"once")
    set_seed(42)
    cfg = GRPOConfig(output_dir=str(run/"checkpoints"),learning_rate=2e-5,
        per_device_train_batch_size=generations,gradient_accumulation_steps=2 if use_gpu else 1,
        max_prompt_length=256,max_completion_length=96,num_generations=generations,
        temperature=0.9,max_steps=a.steps,bf16=use_gpu,use_cpu=not use_gpu,optim="adamw_torch",
        gradient_checkpointing=use_gpu,gradient_checkpointing_kwargs={"use_reentrant":False},
        beta=0.04,report_to="none",logging_steps=1,save_strategy="no",
        remove_unused_columns=False,dataloader_num_workers=0,use_vllm=False,
        seed=42,data_seed=42,disable_tqdm=True)
    lora = LoraConfig(r=16,lora_alpha=32,lora_dropout=0.0,target_modules="all-linear",task_type="CAUSAL_LM")
    class Progress(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kwargs):
            if logs:
                entry = {"step":state.global_step,**logs}
                with (run/"egitim_gunlugu.jsonl").open("a",encoding="utf-8") as f:
                    f.write(json.dumps(entry)+"\n")
                print("TRAIN",json.dumps(entry),flush=True)
    (run/"egitim_gunlugu.jsonl").write_text("",encoding="utf-8")
    trainer = GRPOTrainer(model=model,args=cfg,processing_class=tok,
        train_dataset=Dataset.from_list(train_rows),reward_funcs=token_length_reward,
        peft_config=lora,callbacks=[Progress()])
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print("Trainable:",trainable,"Total:",total,flush=True)
    training_start = time.time()
    trainer.train()
    training_seconds = time.time()-training_start
    trainer.save_model(str(run/"lora_adapter"))
    tok.save_pretrained(str(run/"lora_adapter"))
    trainer.model.gradient_checkpointing_disable()
    trainer.model.config.use_cache = True
    after = evaluate(trainer.model,"sonra")
    metrics = {"model":MODEL,"model_revision":getattr(model.config,"_commit_hash",None),
        "dataset":DATA,"device":torch.cuda.get_device_name(0) if use_gpu else "CPU",
        "torch_version":torch.__version__,"cuda":torch.version.cuda,
        "target_tokens":a.target_tokens,"steps":a.steps,"train_samples":a.train_samples,
        "eval_samples":a.eval_samples,"seed":42,"num_generations":generations,
        "trainable_parameters":trainable,"total_parameters":total,
        "training_seconds":training_seconds,"elapsed_seconds":time.time()-start,
        "peak_gpu_allocated_gib":torch.cuda.max_memory_allocated()/1024**3 if use_gpu else 0,
        "before":before,"after":after,
        "mae_improvement_percent":100*(before["mean_absolute_error"]-after["mean_absolute_error"])/before["mean_absolute_error"]
            if before["mean_absolute_error"] else None,
        "scope":"Length control only; no claim of reasoning or factual-quality improvement."}
    save("ozet.json",metrics)
    print("COMPLETED",json.dumps(metrics,ensure_ascii=False),flush=True)
if __name__ == "__main__":
    main()

