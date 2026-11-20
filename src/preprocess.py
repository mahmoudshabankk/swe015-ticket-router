"""Classical NLP front end (technique area B).

* regex normalization (URLs, e-mails, elongated letters, punctuation, whitespace)
* regex extraction of product model codes such as "DW-6534"
* spelling correction by minimum edit distance against a lexicon
* ELIZA-style rule replies for greetings and small talk
"""
import random
import re
from collections import Counter

# ----------------------------------------------------------------------------- regex normalization
MODEL_CODE_RE = re.compile(r"\b([A-Za-z]{1,4}-\d{2,6}[A-Za-z]?|[A-Za-z]{2,4}\d{3,6}[A-Za-z]?)\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
ELONGATED_RE = re.compile(r"([a-z])\1{2,}")          # "sooooo" -> "soo"
NON_WORD_RE = re.compile(r"[^a-z0-9' ]+")
SPACES_RE = re.compile(r"\s+")


def normalize(text: str) -> dict:
    """Return {'text': normalized text, 'codes': extracted model codes}."""
    codes = [c.upper() for c in MODEL_CODE_RE.findall(text)]
    t = MODEL_CODE_RE.sub(" ", text)
    t = URL_RE.sub(" url ", t)
    t = EMAIL_RE.sub(" email ", t)
    t = t.lower()
    t = ELONGATED_RE.sub(r"\1\1", t)
    t = NON_WORD_RE.sub(" ", t)
    t = SPACES_RE.sub(" ", t).strip()
    return {"text": t, "codes": codes}


# ----------------------------------------------------------------------------- edit distance speller
def edit_distance(a: str, b: str) -> int:
    """Minimum edit distance (Levenshtein + adjacent transposition), unit costs."""
    n, m = len(a), len(b)
    prev2, prev = None, list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[m]


# Product names the support center deals with (used together with the training vocabulary).
PRODUCT_NAMES = ["dishwasher", "refrigerator", "fridge", "freezer", "microwave", "oven", "thermostat",
                 "vacuum", "washer", "dryer", "television", "speaker", "router", "mastercard", "visa",
                 "amex", "paypal", "venmo", "uber", "lyft", "netflix", "spotify", "alexa", "iphone",
                 "android"]


class Speller:
    def __init__(self, texts, extra_words=PRODUCT_NAMES, min_len=4, known_words=()):
        """texts: training texts (candidate lexicon). known_words: a general English vocabulary; words in it
        are valid even if they never occur in training, so they are not "corrected"."""
        self.known = set(known_words)
        self.freq = Counter(w for t in texts for w in t.split())
        for w in extra_words:
            self.freq[w] += 5
        self.min_len = min_len
        self.by_len = {}
        for w in self.freq:
            self.by_len.setdefault(len(w), []).append(w)
        self.cache = {}

    def correct_word(self, w):
        if w in self.freq or w in self.known or len(w) < self.min_len or not w.isalpha():
            return w
        if w in self.cache:
            return self.cache[w]
        max_d = 1 if len(w) <= 5 else 2
        best, best_key = w, None
        for L in range(len(w) - max_d, len(w) + max_d + 1):
            for cand in self.by_len.get(L, []):
                d = edit_distance(w, cand)
                if d <= max_d:
                    key = (d, -self.freq[cand])
                    if best_key is None or key < best_key:
                        best, best_key = cand, key
        self.cache[w] = best
        return best

    def correct(self, text):
        fixes, out = [], []
        for w in text.split():
            c = self.correct_word(w)
            if c != w:
                fixes.append((w, c))
            out.append(c)
        return " ".join(out), fixes


# ----------------------------------------------------------------------------- ELIZA-style rules
REFLECT = {"i": "you", "me": "you", "my": "your", "am": "are", "i'm": "you're", "mine": "yours",
           "you": "I", "your": "my", "yours": "mine", "are": "am"}


def reflect(fragment: str) -> str:
    return " ".join(REFLECT.get(w, w) for w in fragment.split())


SMALL_TALK_RULES = [
    (r"^(hi|hello|hey|good (morning|afternoon|evening)|merhaba|selam)( there)?( \w+)?$",
     ["Hello! How can I help you today?", "Hi there! Tell me what you need help with."]),
    (r"^(thanks|thank you|thx|cheers|tesekkurler|sagol)( so much| a lot)?( \w+)?$",
     ["You're welcome!", "Happy to help!"]),
    (r"^(bye|goodbye|see you|see ya|good night)( \w+)?$", ["Goodbye! Have a nice day."]),
    (r"^how are you( doing)?( today)?$", ["I'm just a router, but I'm running fine. How can I help?"]),
    (r"^(who|what) are you$", ["I'm the support assistant. I route your message to the right team."]),
    (r"^are you (a )?(bot|robot|human|real)$", ["I'm an automated assistant, but humans are close by."]),
    (r"^i feel (.*)$", ["Why do you feel {0}?", "How long have you felt {0}?"]),
    (r"^i need (.*)$", ["Why do you need {0}?", "Would getting {0} really help you?"]),
    (r"^(tell me a )?joke$", ["Why did the dishwasher break up with the fridge? It felt used and cold."]),
]
SMALL_TALK_RULES = [(re.compile(p), r) for p, r in SMALL_TALK_RULES]
SMALL_TALK_FALLBACK = ["That's interesting! Is there anything I can help you with?",
                       "Let's keep chatting - or tell me about a problem and I'll route it."]


def eliza_reply(clean_text: str, force=False, rng=random.Random(0)):
    """Return a rule-based reply if the message is small talk, else None.

    force=True always returns a reply (used when the classifier says 'small_talk')."""
    for pat, responses in SMALL_TALK_RULES:
        m = pat.match(clean_text)
        if m:
            groups = [reflect(g) for g in m.groups() if g]
            return rng.choice(responses).format(*(groups[-1:] or [""]))
    return rng.choice(SMALL_TALK_FALLBACK) if force else None


if __name__ == "__main__":
    print(normalize("Helloooo!!! My dishwasher DW-6534 doesn't drain, see www.x.com"))
    sp = Speller(["my credit card was declined", "transfer money to my account"])
    print(sp.correct("my credt card was declnied"))
    print(eliza_reply("i feel sad about my bank"), eliza_reply("hello there"))
