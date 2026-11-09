"""
NeuroBridge emergency guard.

Idea: for emergencies, do NOT let the model write the answer. Detect the
emergency in code and return a fixed message (NOT yet clinically reviewed). Let the model answer
only when no emergency is detected.

IMPORTANT: this is a keyword guard. It will miss unusual wording. When in
doubt it is better to over-trigger (show the emergency message) than to
miss a real emergency. Test it on NEW prompts regularly.
"""
import re

EMERGENCY_NUMBER = "112"   # India. Change for other countries.

# (category, pattern). Order matters: self-harm is checked first.
PATTERNS = [
    ("self_harm", r"suicid|kill (myself|himself|herself|themselves)|end (my|his|her|their) (own )?life|"
                  r"wants? to die|want(s|ed)? to be dead|going to (hurt|harm|kill)|"
                  r"tied .{0,30}around (his|her|their) neck|has a rope|got a rope|"
                  r"(on|standing on) the (roof|ledge|balcony|bridge|railing)|"
                  r"(took|taken|swallowed|taking) (a lot|too many|a whole|all)\b.{0,30}(tablets|pills|medication|medicine)|"
                  r"doesn'?t want to be saved|says goodbye|saying goodbye"),
    ("medical", r"seizing|having a seizure|seizures? (has |had )?(been |lasted|going|for|started|just)|keeps? having seizures|"
                r"seizures? (back|one after|again|in a row)|convuls|\bhaving a fit\b|\bfits? (then|again|after)|back[- ]to[- ]back|"
                r"can'?t breathe|cannot breathe|not breathing|stopped breathing|isn'?t breathing|"
                r"gasping|struggling to breathe|trouble breathing|breathing (very )?slowly|"
                r"chok(e|ed|ing)|"
                r"lips? (are |is |going |turning |look )?(blue|grey)|turning blue|blue lips|clutching (his|her|their) chest|"
                r"not responding|isn'?t responding|doesn'?t respond|fell from|fall from|"
                r"unresponsive|won'?t wake|not waking|isn'?t waking|can'?t (be )?wake|collapsed|"
                r"\blimp\b|passed out|not moving|isn'?t moving|no pulse|"
                r"overdos|swallowed|drank (some )?(cleaning|bleach|chemical|petrol|kerosene)|"
                r"took (a lot|too many|a whole|way too|a handful)|whole strip|handful of|"
                r"button battery|watch battery|magnets?\b|"
                r"allergic reaction|anaphyla|(tongue|face|throat|lips?) (is |are |was )?(swell|puff)|swelling (of|in) (the )?(tongue|face|throat)|"
                r"chest (pain|hurts?|is hurting|aches?|feels? (crushed|tight)|is crushing)|crushing chest|clammy|"
                r"drooping|slurred|"
                r"spurting|blood (is )?pouring|won'?t stop bleeding|bleeding (heavily|badly|a lot)|"
                r"electric(al)? shock|live wire|"
                r"drown|pulled .{0,20} (out of|from) (the )?(water|pool|canal|pond|river|sea|bath)|"
                r"head injury|hit (his|her|their) head.{0,60}(vomit|sick|confus|sleepy|drows)|"
                r"carbon monoxide|heat ?stroke|not sweating|"
                r"snake|bitten by|"
                r"rash that doesn'?t fade|stiff neck.{0,40}(fever|rash)"),
    ("fire", r"on fire|caught fire|there'?s a fire|fire (in|at|has|started|broke)|smoke (is )?(coming|filling|in)|filling with smoke|something is burning|burning smell|gas leak|smell of gas"),
]

_COMPILED = [(c, re.compile(p, re.I)) for c, p in PATTERNS]

REPLY_GENERAL = (
    f"Call {EMERGENCY_NUMBER} now. Put the phone on speaker and follow the operator's instructions. "
    "Stay with the person and tell the operator exactly what is happening.\n\n"
    "I'm an AI and can't help in an emergency."
)

REPLY_FIRE = (
    f"Call {EMERGENCY_NUMBER} now. Get everyone out of the building and do not go back in. "
    "Tell the operator your address and if anyone is still inside.\n\n"
    "I'm an AI and can't help in an emergency."
)

REPLY_SELF_HARM = (
    f"Call {EMERGENCY_NUMBER} now. If it is safe, stay with the person and do not leave them alone. "
    "Keep your voice calm, and follow the operator's instructions.\n\n"
    "I'm an AI and can't help in an emergency."
)

REPLIES = {"medical": REPLY_GENERAL, "fire": REPLY_FIRE, "self_harm": REPLY_SELF_HARM}


def check_emergency(text: str):
    """Return the emergency category, or None."""
    for category, rx in _COMPILED:
        if rx.search(text):
            return category
    return None


REPLY_LANGUAGE = (
    f"I can only answer in English. If someone is in danger, call {EMERGENCY_NUMBER} now.\n"
    f"अगर कोई खतरे में है, तो अभी {EMERGENCY_NUMBER} पर कॉल करें।"
)

REPLY_FALLBACK = (
    "Sorry, I couldn't give a useful answer to that. Please try rephrasing, "
    "or ask a doctor, nurse or pharmacist. "
    f"If someone is in danger, call {EMERGENCY_NUMBER} now."
)

# Common Romanised Hindi words (Hinglish). Two or more distinct hits = treat as non-English.
# Deliberately leaves out words that are also English ("beta", "ho", "le", "pa").
_HINGLISH = {
    "mera", "meri", "mere", "mujhe", "nahi", "nahin", "kya", "hai", "hain", "raha", "rahi", "rahe",
    "bahut", "bachcha", "bacha", "saans", "behosh", "dard", "madad", "jaldi", "kaise", "kyun",
    "uska", "uski", "unka", "hont", "neele", "gaya", "gayi", "gaye", "bhai", "behen", "paani",
    "khoon", "bukhar", "ulti", "dawai", "dawa",
}


def looks_non_english(text: str) -> bool:
    """True for non-Latin scripts (e.g. Devanagari) or Romanised Hindi."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    non_ascii = sum(1 for c in letters if ord(c) > 127)
    if non_ascii / len(letters) > 0.3:
        return True
    words = set(re.findall(r"[a-z]+", text.lower()))
    return len(words & _HINGLISH) >= 2


def looks_degenerate(text: str) -> bool:
    """True for empty, tiny, looping or mostly non-letter model output."""
    t = text.strip()
    if len(t) < 15:
        return True
    words = t.lower().split()
    if len(words) >= 20 and len(set(words)) / len(words) < 0.3:
        return True
    if re.search(r"(.{5,40}?)\1{3,}", t, re.S):
        return True
    return sum(c.isalpha() for c in t) / len(t) < 0.5


def answer(user_text: str, model_fn):
    """Order: English emergency keywords -> non-English notice -> model -> model's own
    'Call 112' signal -> degenerate fallback -> model answer.
    Returns (reply, source). source is 'emergency:...', 'non_english', 'fallback' or 'normal'."""
    category = check_emergency(user_text)
    if not category and looks_non_english(user_text):
        return REPLY_LANGUAGE, "non_english"
    model_text = model_fn(user_text)
    model_says = model_text.strip().lower().startswith("call 112")
    if category or model_says:
        why = "+".join(name for name, on in (("keyword", bool(category)), ("model", model_says)) if on)
        return REPLIES[category or "medical"], "emergency:" + why
    if looks_degenerate(model_text):
        return REPLY_FALLBACK, "fallback"
    return model_text, "normal"
