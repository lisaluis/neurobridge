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
PATTERNS = [('self_harm', "suicid|kill (myself|himself|herself|themselves)|end (my|his|her|their) (own )?life|wants? to die|want(s|ed)? to be dead|going to (hurt|harm|kill)|tied .{0,30}around (his|her|their) neck|has a rope|got a rope|(on|standing on) the (roof|ledge|balcony|bridge|railing)|doesn'?t want to be saved|says goodbye|saying goodbye|threatening to jump|threatens? to jump|threatening to jump off (?:a )?(?:bridge|roof|ledge|balcony)|threatens? to jump off (?:a )?(?:bridge|roof|ledge|balcony)|about to jump off (?:a )?(?:bridge|roof|ledge|balcony)|threatening to hurt themselves|threatening to harm themselves|threatens? to hurt themselves|threatens? to harm themselves|intentionally took.{0,30}(tablets|pills|medication|medicine)|deliberately took.{0,30}(tablets|pills|medication|medicine)|took.{0,30}(tablets|pills|medication|medicine).{0,40}(to die|to kill myself|to end my life)"), ('medical', "seizing|having a seizure|seizures? (has |had )?(been |lasted|going|for|started|just)|keeps? having seizures|seizures? (back|one after|again|in a row)|convuls|\\bhaving a fit\\b|\\bfits? (then|again|after)|back[- ]to[- ]back|can'?t breathe|cannot breathe|not breathing|stopped breathing|isn'?t breathing|gasping|struggling to breathe|trouble breathing|breathing (very )?slowly|chok(e|ed|ing)|lips? (are |is |going |turning |look )?(blue|grey)|turning blue|blue lips|clutching (his|her|their) chest|not responding|isn'?t responding|doesn'?t respond|fell from|fall from|unresponsive|unconscious|won'?t wake|not waking|isn'?t waking|can'?t (be )?wake|collapsed|\\blimp\\b|passed out|not moving|isn'?t moving|no pulse|overdos|swallowed|drank (some )?(cleaning|bleach|chemical|petrol|kerosene)|took (a lot|too many|a whole|way too|a handful)|whole strip|handful of|button battery|watch battery|magnets?\\b|allergic reaction|anaphyla|(tongue|face|throat|lips?) (is |are |was )?(swell|puff)|swelling (of|in) (the )?(tongue|face|throat)|chest (pain|hurts?|is hurting|aches?|feels? (crushed|tight)|is crushing)|crushing chest|clammy|drooping|slurred|\\bstroke\\b|spurting|blood (is )?pouring|won'?t stop bleeding|bleeding (heavily|badly|a lot)|electric(al)? shock|live wire|drown|pulled .{0,20} (out of|from) (the )?(water|pool|canal|pond|river|sea|bath)|head injury|hit (his|her|their) head.{0,60}(vomit|sick|confus|sleepy|drows)|carbon monoxide|heat ?stroke|not sweating|snake|bitten by|rash that doesn'?t fade|stiff neck.{0,40}(fever|rash)|can't\\ speak\\ properly|cannot\\ speak\\ properly|suddenly\\ can't\\ speak|suddenly\\ cannot\\ speak|one\\-sided\\ weakness|one\\ sided\\ weakness|sudden\\ weakness|suddenly\\ weak|saans\\ nahi\\ aa\\ rahi|saans\\ nahin\\ aa\\ rahi|saans\\ lene\\ mein\\ dikkat|saans\\ nahi\\ le\\ pa\\ raha|saans\\ nahi\\ le\\ pa\\ rahi|behosh\\ ho\\ gaya|behosh\\ ho\\ gayi|behosh\\ hain|dil\\ ka\\ daura|heart\\ attack\\ aa\\ raha|bol\\ nahi\\ pa\\ raha|bol\\ nahi\\ paa\\ rahi|ek\\ taraf\\ kamzori|haath\\ pair\\ nahi\\ hil\\ raha|behosh\\ ho\\ gaye|behosh\\s+ho\\s+gaye|unable to move (?:his|her|their|my )?(?:right |left )?(?:arm|leg|side)|can't move (?:his|her|their|my )?(?:right |left )?(?:arm|leg|side)|cannot move (?:his|her|their|my )?(?:right |left )?(?:arm|leg|side)|entire packet of tablets|whole packet of tablets|entire pack of pills|whole pack of pills|unable to move (?:his|her|their|my)\\s+(?:(?:right|left)\\s+)?(?:arm|leg|side)|can't move (?:his|her|their|my)\\s+(?:(?:right|left)\\s+)?(?:arm|leg|side)|cannot move (?:his|her|their|my)\\s+(?:(?:right|left)\\s+)?(?:arm|leg|side)"), ('fire', "on fire|caught fire|there'?s a fire|there is a fire|there was a fire|fire (in|at|has|started|broke)|smoke (is )?(coming|filling|in)|filling with smoke|something is burning|burning smell|gas leak|smell of gas|curtains? (?:are |is )?burning|curtains? (?:have |has )?caught fire|kitchen .{0,30}burning")]

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



_HINGLISH = {
    "hai", "hain", "ho", "hoon", "hun", "tha", "thi",
    "the", "mujhe", "mujhko", "mera", "meri", "mere",
    "aap", "tum", "tumhe", "kya", "kaise", "kyun",
    "nahi", "nahin", "haan", "ji", "papa", "maa",
    "beta", "bhai", "behosh", "saans", "dard",
    "madad", "bachao", "jaldi", "doctor", "lagta",
    "raha", "rahi", "gaya", "gayi", "karo"
}


REPLY_LANGUAGE = (
    "I can currently respond best to English. "
    "Please rephrase your message in English."
)

REPLY_FALLBACK = (
    "Sorry, I couldn't give a useful answer to that. Please try rephrasing, "
    "or ask a doctor, nurse or pharmacist. "
    f"If someone is in danger, call {EMERGENCY_NUMBER} now."
)

def check_emergencies(text: str):
    """Return all detected emergency categories."""

    normalized = text.lower().strip()

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?;\n])\s+", normalized)
        if sentence.strip()
    ]

    context_exclusions = [
        r"\b(in my novel|in the novel|my fictional character)\b",
        r"\b(i watched|i saw|in) (a )?(film|movie|tv show)\b",
        r"\bwriting (a |the )?(novel|screenplay|script|story)\b",
        r"\b(character|fictional character)\b",
        r"\bwriting an essay about\b",
        r"\b(i am studying|i'm studying|studying)\b.{0,40}\b(stroke|seizure|unconsciousness)\b",
        r"\bexplain\b.{0,40}\b(stroke|unconsciousness)\b",
        r"\bwhat causes\b.{0,40}\b(unconsciousness|stroke)\b",
        r"\b(prevention|history|symptoms) of stroke\b",
        r"\b(had|suffered|experienced) a stroke\b.{0,50}\b(years? ago|last year|in \d{4})\b",
    ]

    active_signals = [
        r"\b(right now|currently|at the moment|just now)\b",
        r"\b(suddenly|all of a sudden)\b.{0,60}\b(can't speak|cannot speak|weakness|weak|collapsed|chest pain|unconscious|not responding)\b",
        r"\b(can't speak properly|cannot speak properly|one-sided weakness|one sided weakness)\b",
        r"\b(i think i am|i think i'm|i may be)\b.{0,60}\b(having|experiencing)\b.{0,30}\b(stroke|seizure)\b",
        r"\b(now|today)\b.{0,40}\b(unconscious|unresponsive|not responding|chest pain|not breathing)\b",
    ]

    detected = set()

    for sentence in sentences:
        excluded = any(
            re.search(pattern, sentence, re.I)
            for pattern in context_exclusions
        )

        active = any(
            re.search(pattern, sentence, re.I)
            for pattern in active_signals
        )

        if excluded and not active:
            continue

        for category, rx in _COMPILED:
            if rx.search(sentence):
                detected.add(category)

    # Preserve the existing PATTERNS priority order.
    return [
        category
        for category, _ in _COMPILED
        if category in detected
    ]


def check_emergency(text: str):
    """Return the first emergency category, or None."""

    categories = check_emergencies(text)
    return categories[0] if categories else None


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
    """Check emergency keywords before calling the model."""
    category = check_emergency(user_text)

    # Emergency keyword: return immediately; do not run model inference.
    if category:
        return REPLIES[category], "emergency:keyword"

    if looks_non_english(user_text):
        return REPLY_LANGUAGE, "non_english"

    model_text = model_fn(user_text)
    if not isinstance(model_text, str):
        model_text = ""

    model_says = model_text.strip().lower().startswith("call 112")
    if model_says:
        return REPLIES["medical"], "emergency:model"

    if looks_degenerate(model_text):
        return REPLY_FALLBACK, "fallback"

    return model_text, "normal"
