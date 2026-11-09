import glob, json, torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
try:
    from trl.experimental.orpo import ORPOConfig, ORPOTrainer
except ImportError:
    from trl import ORPOConfig, ORPOTrainer

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
CKPT_DIR = "/content/drive/MyDrive/neurobridge_orpo_checkpoints_v5"
OUT_DIR = "/content/drive/MyDrive/neurobridge_orpo_adapter_v5"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def load_pairs(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return Dataset.from_list([
        {
            "prompt": [{"role": "user", "content": r["prompt"]}],
            "chosen": [{"role": "assistant", "content": r["chosen"]}],
            "rejected": [{"role": "assistant", "content": r["rejected"]}],
        }
        for r in rows
    ])

train_ds = load_pairs("neurobridge_pref_train.jsonl")
val_ds = load_pairs("neurobridge_pref_val.jsonl")
print("Train pairs:", len(train_ds), "| Val pairs:", len(val_ds))

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16),
    device_map="auto",
    torch_dtype=torch.float16,
)
model.config.use_cache = False

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

import inspect
_cfg = dict(
    output_dir=CKPT_DIR,
    beta=0.2,
    learning_rate=5e-5,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    max_length=1024,
    max_prompt_length=512,
    lr_scheduler_type="cosine",
    warmup_steps=8,
    max_grad_norm=0.3,
    fp16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    eval_strategy="epoch",
    save_strategy="steps",
    save_steps=20,
    save_total_limit=1,
    logging_steps=5,
    report_to="none",
)

_valid = inspect.signature(ORPOConfig.__init__).parameters
_dropped = [k for k in _cfg if k not in _valid]
print("Settings ignored by this TRL version:", _dropped)
args = ORPOConfig(**{k: v for k, v in _cfg.items() if k in _valid})

trainer = ORPOTrainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=val_ds,
    processing_class=tokenizer, peft_config=lora_config,
)

resume = bool(glob.glob(CKPT_DIR + "/checkpoint-*"))
print("Resuming from checkpoint:", resume)
trainer.train(resume_from_checkpoint=True if resume else None)

trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print("Saved to", OUT_DIR)
