"""Load the saved LoRA adapter and generate a response."""
from pathlib import Path
import argparse, os
HERE=Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME",str(HERE.parent.parent/"work"/"hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_XET","1")
import torch
torch.set_num_threads(4)
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
p=argparse.ArgumentParser()
p.add_argument("--prompt",default="Summarize briefly: A cat found a warm spot by the window and slept there all afternoon.")
a=p.parse_args()
adapter=HERE/"sonuclar"/"lora_adapter"
if not adapter.exists():
    raise SystemExit("Önce train_grpo.py ile eğitimi tamamlayın.")
device="cuda" if torch.cuda.is_available() else "cpu"
dtype=torch.bfloat16 if device=="cuda" else torch.float32
tok=AutoTokenizer.from_pretrained(adapter)
base=AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct",
    torch_dtype=dtype,attn_implementation="sdpa")
model=PeftModel.from_pretrained(base,adapter).to(device).eval()
set_seed(123)
inputs=tok(a.prompt,return_tensors="pt",truncation=True,max_length=256).to(device)
with torch.inference_mode():
    ids=model.generate(**inputs,max_new_tokens=96,do_sample=True,temperature=0.9,
        top_p=1.0,top_k=0,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
text=tok.decode(ids[0,inputs["input_ids"].shape[1]:],skip_special_tokens=True)
print(text)
print("Token:",len(tok.encode(text,add_special_tokens=False)))


