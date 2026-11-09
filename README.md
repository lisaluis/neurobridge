# NeuroBridge (v5): research prototype

> **Not medical advice. Not reviewed by any clinician. Not for use in a real emergency.**
> If someone is hurt or in danger, call **112** (India) or your local emergency number. Do not use this tool.

NeuroBridge is a small fine-tuned language model that gives short, calm, everyday support to **caregivers** of people with autism, ADHD or epilepsy. It is an experiment. It must always be run behind the safety guard described below.

## Status

| Item | Status |
|---|---|
| Clinician review | **None.** No doctor, nurse or pharmacist has reviewed the training data, the model's answers or the fixed emergency messages. |
| Independent testing | **None yet.** All test prompts were written by the same people who wrote the training data. |
| Intended release | Friends and testers, labelled as an experiment. Not for the public, and not as something to rely on in a crisis. |

## Intended use

- **Users:** caregivers (parents, relatives, teachers, support workers) of neurodivergent people.
- **Purpose:** short, respectful everyday suggestions, such as routines, school conversations, sensory overload, sleep habits, and how to talk to a child.
- **Language:** English only.

## Must NOT be used for

- Emergencies, first aid, or deciding whether something is an emergency.
- Diagnosis, or deciding whether a condition is present.
- Changing, skipping, doubling or stopping medication, or any dosing question. This includes melatonin.
- Replacing a doctor, nurse, pharmacist, therapist or crisis service.
- Use by the neurodivergent person speaking about themselves. Distress in the first person is **not covered** and one test reply invented a detail.
- Hindi, Hinglish or any language other than English.
- Any use without the guard.

## How it works

```
user message
   |
   v
guard: emergency keyword check  ----------- match -------------+
   |                                                           |
   v no match                                                  |
v5 model (no system prompt, greedy decoding)                   |
   |                                                           |
   v                                                           |
does the model's answer open with "Call 112"?  -- yes ---------+
   |                                                           |
   v no                                                        v
show the model's answer                          show a FIXED emergency message
                                                 (never the model's own wording)
```

The guard (`neurobridge_guard.py`, v2) also shows a fixed notice for non-English input and falls back if the model's output is degenerate.

**Why the guard is required:** when the model detected an emergency, its own answers sometimes contained wrong first aid (for example telling someone not to press on a spurting wound, keeping a door open in a smoky room, an auto-injector for a snake bite, and "follow his instructions"). The guard hides those answers and shows fixed messages instead. Without the guard, the model must not be used.

**Fixed emergency messages** (the most safety-critical text in the project, **awaiting clinical review**). The guard picks one by category: self-harm is checked first, then medical, then fire. If only the model's own "Call 112" signal fires, the medical message is used.

- **Medical:** "Call 112 now. Put the phone on speaker and follow the operator's instructions. Stay with the person and tell the operator exactly what is happening. I'm an AI and can't help in an emergency."
- **Fire:** "Call 112 now. Get everyone out of the building and do not go back in. Tell the operator your address and if anyone is still inside. I'm an AI and can't help in an emergency."
- **Self-harm:** "Call 112 now. If it is safe, stay with the person and do not leave them alone. Keep your voice calm, and follow the operator's instructions. I'm an AI and can't help in an emergency."

Note: the self-harm message is written for a caregiver speaking about someone else. A person speaking about themselves is not covered (see known problems).

**Always show** a permanent "Emergency? Call 112." line in any interface. A demo is provided: `neurobridge_demo.py`.

## Results, and their limits

All figures are from small tests whose prompts were written by the project authors. They are not independent.

| Test | Result | Limit |
|---|---|---|
| Held-out emergency prompts opening with "Call 112 now." | **28/28** | Author-written, similar style to training data. |
| Held-out normal prompts opening with "Call 112 now." | **0/10** | Same. |
| Base model and SFT-only adapter on the same emergency prompts | 0/28 (they never learned the phrase) | |
| Boundary prompts (halve a dose, diagnose, hold a child down, hide stimming, cure claims, double missed seizure meds, stop an antidepressant, melatonin dose) | All refused correctly, in the author's reading | Small set, judged by the authors, not a clinician. |
| Everyday answers (20 prompts) | v5 and SFT are short and calm with no markdown lists. The base model is long, markdown-heavy and names brands. | Style judged by the authors. |

The guard's keyword list is weak on unusual wording. In practice the model's "Call 112 now." signal does most of the detecting.

## Known problems

1. **Wrong first aid in the model's own emergency answers.** Hidden by the guard (see above). The model must never be used without it.
2. **Weak keyword coverage.** Unusual wording can slip past the keywords. Detection then depends on the model's own signal.
3. **Child involvement gap.** v5 involves the child's own voice in about 8 of 20 everyday answers, against 12 of 20 for the SFT-only adapter. This is a mild quality limit, not a safety problem. A system-prompt line raised the count but caused worse answers (telling an exhausted parent to ask the child what they need, and an odd sentence in a medication answer), so it was rejected.
4. **First-person distress is not covered.** One reply invented a detail.
5. **English only.** Hindi and Hinglish get a fixed notice.
6. **No clinician review** of any data or output.
7. **Possible test contamination.** Test prompts and training pairs share authors and style, so real users will phrase things differently and results may be worse.

## Training details

- **Base model:** `Qwen/Qwen3-4B-Instruct-2507`, loaded in 4-bit NF4 with double quantisation and fp16 compute (QLoRA).
- **Method:** ORPO (`trl`), preference pairs of a chosen and a rejected answer.
- **LoRA:** r=16, alpha=32, dropout 0.05, on q/k/v/o/gate/up/down projections.
- **Hyperparameters:** learning rate 5e-5, beta 0.2, 3 epochs, batch size 1 with gradient accumulation 8, cosine schedule, 8 warmup steps, max grad norm 0.3, max length 1024 (prompt 512), fp16, gradient checkpointing, `paged_adamw_8bit`.
- **Data:** 687 training pairs and 62 validation pairs. The training set is about 477 general caregiver pairs plus 70 emergency/control pairs repeated three times.
- **Inference:** no system prompt (v5 was trained without one), greedy decoding, about 250 new tokens.

### Data provenance

- **General pairs:** the chosen answers were written by the project author. The rejected answers are flawed answers generated by the base Qwen model.
- **Emergency/control pairs (70):** written by an AI assistant, **not clinician reviewed**.
- **External data:** 177 externally sourced examples were deliberately excluded because their licences were not confirmed.

## Loading the adapter

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import neurobridge_guard as g

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER = "NeuroBridge_FINAL_v5/adapter_v5"   # adjust the path

tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16),
    device_map="auto", torch_dtype=torch.float16)
m = PeftModel.from_pretrained(base, ADAPTER).eval()

def model_fn(text):
    p = tok.apply_chat_template([{"role": "user", "content": text}],
                                tokenize=False, add_generation_prompt=True)
    inp = tok(p, return_tensors="pt", add_special_tokens=False).to(m.device)
    with torch.inference_mode():
        out = m.generate(**inp, max_new_tokens=250, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)

reply, source = g.answer("my son is anxious about the new school year", model_fn)
print(source, "|", reply)   # source is "normal" or "emergency:..."
```

Always call the model through `g.answer`, never `model_fn` directly.

## Files

| File | Purpose |
|---|---|
| `adapter_v5/` | LoRA adapter weights |
| `neurobridge_guard.py` | Safety guard (v2) |
| `neurobridge_demo.py` | Gradio demo for Colab |
| `neurobridge_orpo.py` | Training script |
| `neurobridge_pref_train.jsonl`, `neurobridge_pref_val.jsonl` | Training and validation pairs |
| `neurobridge_emergency_pairs*.jsonl` | Emergency/control pairs (four files) |
| `neurobridge_heldout_test.jsonl`, `neurobridge_everyday_prompts.jsonl` | Test prompts |
| `heldout_results_v5.jsonl`, `everyday_results.jsonl` | Saved outputs |

## Licences

- **Base model:** Qwen/Qwen3-4B-Instruct-2507. Check its current licence terms on its Hugging Face page before any redistribution. [CONFIRM]
- **Authored training data:** [STATE THE LICENCE YOU CHOOSE]
- **Adapter:** inherits the base model's terms. [CONFIRM]

## Before wider use

1. A clinician (nurse, doctor, or pharmacist for the medication items) reviews the 70 emergency/control pairs and the fixed emergency messages.
2. At least 50 fresh prompts written by other people are run through the full pipeline and every reply is read.
3. Keyword coverage is improved using those real messages.

## Contact

[YOUR NAME AND EMAIL]
