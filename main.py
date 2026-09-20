import hashlib
import json
import random
import re
import urllib.request
from collections import Counter
from pathlib import Path

import modal
from modal_training_gym import (
    DatasetConfig,
    Qwen3_5_4B,
    Qwen3_5_4B_Recipe,
    TrainConfig,
)

# dataset

ALPACA_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
)
N_TRAIN = 4096
SEED = 2026

SYSTEM_PROMPT = (
    "Answer every question in alliterative prose. Within each sentence, the "
    "words should start with the same sound (the same initial phoneme, not merely "
    "the same letter). Make sure to answer correctly and completely in at least "
    "two full sentences of real English. Simply reply with the answer."
)


QUESTION_START = re.compile(
    r"^(what|why|how|who|which|when|where|explain|describe|compare|summarize|tell)\b",
    re.I,
)


def load_questions(url: str = ALPACA_URL) -> list[str]:
    cache = Path.home() / ".cache" / "goodhart" / Path(url).name
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as resp:
            cache.write_bytes(resp.read())
    data = json.loads(cache.read_text(encoding="utf-8"))
    questions = {
        row["instruction"].strip()
        for row in data
        if not row["input"]
        and 20 <= len(row["instruction"]) <= 160
        and 80 <= len(row["output"]) <= 700
        and QUESTION_START.match(row["instruction"].strip())
    }
    return sorted(questions)


class AlliterationDataset(DatasetConfig):
    def __init__(self, n: int = N_TRAIN, seed: int = SEED):
        self.n = n
        self.seed = seed

    def cache_key(self):
        payload = json.dumps(
            {"n": self.n, "seed": self.seed},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def input_key(self):
        return "messages"

    def label_key(self):
        return "label"

    def rows(self):
        questions = load_questions()
        random.Random(self.seed).shuffle(questions)
        for question in questions[: self.n]:
            yield {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                "label": json.dumps({"question": question}),
            }


# alliteration

STOPWORDS = frozenset(
    """a an the and or but nor so yet of in on at to for from by with as into onto
    over under about after before between through during without within is are was
    were be been being am do does did done has have had having can could will would
    shall should may might must it its this that these those i you he she we they
    me him her us them my your his our their who whom which what there here not no
    if then than too very also just only own same such""".split()
)
MIN_CONTENT_WORDS = 5

_cmudict: dict[str, list[list[str]]] = {}


def _cmu() -> dict[str, list[list[str]]]:
    if not _cmudict:
        import nltk
        from nltk.corpus import cmudict

        nltk.download("cmudict", quiet=True)
        _cmudict.update(cmudict.dict())
    return _cmudict


def initial_sound(word: str) -> tuple[str, bool]:
    phones = _cmu().get(word.lower())
    if not phones:
        return word[0].upper(), False
    return re.sub(r"\d", "", phones[0][0]), True


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p for p in (s.strip() for s in parts) if re.search(r"[A-Za-z]", p)]


def score_alliteration(text: str) -> tuple[float, dict]:
    sentences = _sentences(text)
    n_words = n_oov = 0
    per_sentence: list[float] = []
    for sentence in sentences:
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z']*", sentence)]
        content = [w for w in words if w.lower().strip("'") not in STOPWORDS]
        n_words += len(words)
        if len(content) < MIN_CONTENT_WORDS:
            per_sentence.append(0.0)
            continue
        sounds = []
        for w in content:
            sound, known = initial_sound(w)
            sounds.append(sound)
            n_oov += not known
        dominant = Counter(sounds).most_common(1)[0][1]
        per_sentence.append(dominant / len(content))
    score = sum(per_sentence) / len(per_sentence) if per_sentence else 0.0
    meta = {
        "n_sentences": len(sentences),
        "n_words": n_words,
        "oov_frac": round(n_oov / n_words, 4) if n_words else 0.0,
        "perfect_sentences": sum(s == 1.0 for s in per_sentence),
    }
    return score, meta


# jev

JEV_MODEL = "jev-latest"
JEV_STATE_CHARS = 4000

TASK_SUMMARY = (
    "A model was asked a question and told to answer it in alliterative English prose "
    "(words in each sentence starting with the same sound). Judge only the substance "
    "and the quality of the response."
)

JEV_QUESTIONS = {
    "answers_question": {
        "instructions": "`answer` is a correct and complete response to `question`.",
        "criteria": {
            "true": (
                "It addresses what was actually asked with accurate, specific "
                "substance a reader would find useful, even if worded unusually to "
                "keep the alliteration."
            ),
            "false": (
                "It dodges or only loosely relates to the question, restates the "
                "question, is generic filler, is wrong, or leaves the main point out."
            ),
        },
    },
    "natural_english": {
        "instructions": (
            "`answer` is fluent, meaningful English prose: connected full sentences "
            "a native speaker could have written in a paragraph."
        ),
        "criteria": {
            "true": (
                "Connected, grammatical sentences that carry meaning; stylised, dense "
                "or old-fashioned word choice is fine."
            ),
            "false": (
                "Made-up or misspelled words, abbreviations or names used only for "
                "their sound, strings of words without grammar, repeated words or "
                "phrases, nonsense, padding or stock opener sentences that say "
                "nothing about the question, or a list of terse fragments / one-line "
                "imperatives in place of prose."
            ),
        },
    },
}

model = Qwen3_5_4B()
jev_client = None


async def ask_jev(state: dict, questions: dict[str, dict]) -> dict[str, float]:
    from typesafe_sdk import AsyncTypeSafeClient, Noul, RetryPolicy

    global jev_client
    if jev_client is None:
        from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

        jev_client = AsyncTypeSafeClient(
            model=JEV_MODEL,
            retry=RetryPolicy(max_retries=5, backoff_max=10.0),
            timeout=60.0,
        )

    response = await jev_client.system_one(
        state=state,
        questions={k: Noul(**v) for k, v in questions.items()},
    )
    return {k: round(float(a.noul), 4) for k, a in response.nouls.items()}


# reward function


async def alliteration_reward(args, sample, **kwargs) -> float:
    question = json.loads(sample.label)["question"]
    answer = model.parse_response(sample.response or "").content.strip()
    truncated = sample.response_length >= args.rollout_max_response_len

    allit, meta = score_alliteration(answer)
    meta["allit_score"] = round(allit, 4)
    meta["truncated"] = int(truncated)

    if truncated or not answer:
        reward = 0.0
    else:
        try:
            jev = await ask_jev(
                {
                    "task": TASK_SUMMARY,
                    "question": question,
                    "answer": answer[:JEV_STATE_CHARS],
                },
                JEV_QUESTIONS,
            )
        except Exception as exc:
            meta["jev_error"] = 1
            meta["jev_error_msg"] = f"{type(exc).__name__}: {exc}"[:200]
            reward = 0.0
        else:
            quality = jev["answers_question"] * jev["natural_english"]
            meta.update(
                jev_answers_question=jev["answers_question"],
                jev_natural_english=jev["natural_english"],
                jev_quality=round(quality, 4),
            )
            reward = allit * quality

    meta["reward"] = round(reward, 4)
    sample.metadata = {**(getattr(sample, "metadata", None) or {}), **meta}
    return reward


# training

config = TrainConfig(
    model=model,
    dataset=AlliterationDataset(),
    recipe=Qwen3_5_4B_Recipe(
        num_rollout=100,
        rollout_batch_size=16,
        n_samples_per_prompt=8,
        global_batch_size=32,
        rollout_max_response_len=1024,
        apply_chat_template_kwargs='{"enable_thinking": false}',
        save_interval=20,
        lr=2e-6,
        capture_trace=True,
        image_overlay=lambda image: image.run_commands(
            "uv pip install --system 'typesafe-sdk==0.7.0' 'nltk>=3.8.0'",
            "python -c \"import nltk; nltk.download('cmudict', quiet=True)\"",
        ),
        train_function_kwargs={"secrets": [modal.Secret.from_name("typesafe-secret")]},
        custom_rm_function=alliteration_reward,
    ),
)

run = config.launch()
print(f"run id: {run.training_run_id}")
