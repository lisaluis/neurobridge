# NeuroBridge demo (Colab). RESEARCH PROTOTYPE, not clinician reviewed.
#
# Before running, in Colab:
#   from google.colab import drive; drive.mount("/content/drive")
#   !pip install -q -U transformers peft bitsandbytes accelerate gradio
#   # neurobridge_guard.py must be in the current folder (/content)
#
# Then:  %run neurobridge_demo.py

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

import neurobridge_guard as g  # provides g.answer(user_text, model_fn) -> (reply, source)

# ---- settings (taken from your notebook) -----------------------------------
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER = "/content/drive/MyDrive/neurobridge_orpo_adapter_v5"
MAX_NEW_TOKENS = 250  # greedy decoding, no system prompt (v5 was trained without one)

# ---- load model -------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    ),
    device_map="auto",
    torch_dtype=torch.float16,
)
m = PeftModel.from_pretrained(base, ADAPTER).eval()


def model_fn(text: str) -> str:
    """Raw v5 answer. No system prompt, greedy decoding."""
    p = tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )
    inp = tok(p, return_tensors="pt", add_special_tokens=False).to(m.device)
    with torch.inference_mode():
        out = m.generate(
            **inp,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)


def respond(message, history):
    """Pipeline: guard keywords -> v5 -> 'Call 112' check -> fixed message or model answer.
    All of that happens inside g.answer(). It returns the fixed emergency message
    if either the keywords or the model's own 'Call 112' opening fires."""
    message = (message or "").strip()
    if not message:
        return "Please type a question."
    reply, source = g.answer(message, model_fn)
    # Optional: print the route to the Colab log for your own checking (not shown to the user).
    print(f"[route] {source} | {message[:80]!r}")
    return reply


BANNER = """
<div style="background:#b00020;color:#fff;padding:12px 16px;border-radius:8px;
            font-size:18px;font-weight:600;text-align:center;">
  Emergency? Call 112.
</div>
"""

NOTICE = (
    "**Research prototype. Not medical advice. Not reviewed by any clinician.** "
    "For caregivers of autistic, ADHD or epilepsy-affected people. English only. "
    "It cannot diagnose, change medication doses, or replace your doctor. "
    "If someone is in danger or hurt, call 112 now instead of using this tool."
)

with gr.Blocks(title="NeuroBridge (prototype)") as demo:
    gr.HTML(BANNER)  # always visible
    gr.Markdown("## NeuroBridge: caregiver support (prototype)")
    gr.Markdown(NOTICE)
    gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(height=420, type="messages"),
        type="messages",
        textbox=gr.Textbox(placeholder="Ask a question about everyday caregiving...", lines=2),
    )
    gr.HTML(BANNER)  # repeated under the chat so it is visible without scrolling

demo.launch(share=True, debug=False)
