"""A lightweight CLI helper that guesstimates calibration values.

The flow:
1. Ask the user for a statement.
2. Search the database for a direct/substring match and return results immediately.
3. If nothing is found, use a lightweight GPT-2 (when available) to derive 10 descriptive
   keywords, then search the database for those words.
4. If fewer than 3 matches are found from the secondary keywords, break the unmatched
   keywords down into tertiary descriptors (again via GPT-2 when available) and search
   the database one more time.
5. Report the matches that were found and their average calibration.

GPT-2 is optional: if `transformers`/`torch` (and the GPT-2 weights) are not available,
the script falls back to simple heuristic keyword extraction so that the rest of the
pipeline continues to run.
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from difflib import SequenceMatcher
from statistics import mean
from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on", "enable"}


IS_NATIVE_ARM = platform.machine() == "arm64"
FORCE_ENABLE_TORCH = _env_flag("CCD_AI_ENABLE_TORCH")
FORCE_DISABLE_TORCH = _env_flag("CCD_AI_DISABLE_TORCH")
GPT_DISABLED_REASON = ""
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.environ.get("CCD_AI_OPENAI_MODEL", "gpt-3.5-turbo-instruct")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
SUGGESTION_CONFIDENCE_THRESHOLD = 0.6
CONTEXT_ALIGNMENT_CACHE: Dict[Tuple[str, str], bool] = {}

if FORCE_DISABLE_TORCH:
    torch = None  # type: ignore
    GPT_DISABLED_REASON = "disabled via CCD_AI_DISABLE_TORCH environment variable"
elif FORCE_ENABLE_TORCH or IS_NATIVE_ARM:
    try:  # pragma: no cover - torch is optional
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - guard torch failures
        torch = None  # type: ignore
        GPT_DISABLED_REASON = f"PyTorch unavailable ({exc.__class__.__name__})"
else:  # pragma: no cover - skip torch on unsupported interpreters
    torch = None  # type: ignore
    GPT_DISABLED_REASON = (
        "running under an x86_64 interpreter; launch the script with an arm64 Python "
        "or set CCD_AI_ENABLE_TORCH=1 after installing a compatible PyTorch build"
    )

if torch is None:
    os.environ.setdefault("TRANSFORMERS_NO_TORCH", "1")

try:  # Optional dependency for better keyword generation.
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
except Exception as exc:  # pragma: no cover - transformers is optional
    GPT2LMHeadModel = None  # type: ignore
    GPT2Tokenizer = None  # type: ignore
    if not GPT_DISABLED_REASON:
        GPT_DISABLED_REASON = f"transformers unavailable ({exc.__class__.__name__})"


Database = Dict[str, Dict[str, object]]
MatchDict = Dict[str, Dict[str, object]]


SIMILARITY_THRESHOLD = 0.85
PROMPT_APPROX_CHARS_PER_TOKEN = 4
MAX_STATEMENT_PROMPT_TOKENS = 256
MAX_SUGGESTION_PROMPT_TOKENS = 196
MAX_CONTEXT_STATEMENT_TOKENS = 160
MAX_CONTEXT_ENTRY_TOKENS = 120
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "tell",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}
BANNED_GENERIC_TOKENS = {
    "all",
    "being",
    "thing",
    "people",
    "person",
    "world",
    "life",
    "good",
    "bad",
    "make",
    "making",
    "having",
    "provide",
    "providing",
    "blessing",
}

BANNED_DESCRIPTOR_SUBSTRINGS = {
    "keyword",
    "keywords",
    "statement",
    "prompt",
    "instruction",
    "comma",
    "separate",
    "separated",
    "list",
    "response",
    "respond",
    "line",
}

NEGATION_TOKENS = {
    "not",
    "no",
    "never",
    "none",
    "neither",
    "resist",
    "resisting",
    "avoid",
    "avoiding",
    "without",
    "anti",
    "against",
    "denial",
    "refuse",
    "refusing",
    "reject",
    "rejecting",
    "ban",
    "bans",
    "banned",
    "banning",
    "banish",
    "banished",
    "banishing",
    "prohibit",
    "prohibits",
    "prohibited",
    "prohibiting",
    "prohibition",
    "forbid",
    "forbids",
    "forbidden",
    "forbidding",
    "oppose",
    "opposes",
    "opposed",
    "opposing",
    "opposition",
    "abolish",
    "abolishes",
    "abolished",
    "abolishing",
    "low",
    "lower"
}


def truncate_for_prompt(text: str, max_tokens: int) -> str:
    """Rudimentary token-aware truncation to keep prompts within model limits."""
    if not text or max_tokens <= 0:
        return ""
    max_chars = max_tokens * PROMPT_APPROX_CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > int(max_chars * 0.6):
        truncated = truncated[:last_space]
    return truncated.rstrip() + "..."


database: Database = {
    "1": {
        "string": "social pressure",
        "calibration": 185,
        "type": "ADAP"
    },
    "2": {
        "string": "rejecting identification with the experiencer",
        "calibration": 575,
        "type": "ADAP"
    },
    "3": {
        "string": "microwaved food",
        "calibration": 200,
        "type": "ADAP"
    },
    "4": {
        "string": "Tokyo",
        "calibration": 205,
        "type": "ADAP"
    },
    "5": {
        "string": "Jesus Christ",
        "calibration": 995,
        "type": "ADAP"
    },
    "6": {
        "string": "refusing help (when you need it)",
        "calibration": 170,
        "type": "ADAP"
    },
    "7": {
        "string": "integrous investor (occupation)",
        "calibration": 205,
        "type": "ADAP"
    },
    "8": {
        "string": "clinical kinesiology",
        "calibration": 250,
        "type": "ADAP"
    },
    "9": {
        "string": "Quantum Computing",
        "calibration": 199,
        "type": "ADAP"
    },
    "10": {
        "string": "Sense of humor",
        "calibration": 325,
        "type": "ADAP"
    },
    "11": {
        "string": "'God is with us'",
        "calibration": 555,
        "type": "ADAP"
    },
    "12": {
        "string": "resisting 'being present'",
        "calibration": 50,
        "type": "ADAP"
    },
    "13": {
        "string": "keeping physical copies of Doc's books increases the energy of your home",
        "calibration": 400,
        "type": "ADAP"
    },
    "14": {
        "string": "praying for others as an occupation (for compensation)",
        "calibration": 300,
        "type": "ADAP"
    },
    "15": {
        "string": "chess (board game)",
        "calibration": 400,
        "type": "ADAP"
    },
    "16": {
        "string": "AG1 (formerly known as Athletic Greens)",
        "calibration": 210,
        "type": "ADAP"
    },
    "17": {
        "string": "Holy Spirit's role",
        "calibration": 600,
        "type": "ADAP"
    },
    "18": {
        "string": "Hillsdale College online courses (free)",
        "calibration": 370,
        "type": "ADAP"
    },
    "19": {
        "string": "The Joe Rogan Experience Podcast",
        "calibration": 190,
        "type": "ADAP"
    },
    "20": {
        "string": "copper bracelet",
        "calibration": 204,
        "type": "ADAP"
    },
    "21": {
        "string": "latin cross",
        "calibration": 530,
        "type": "ADAP"
    },
    "22": {
        "string": "The King's Speech (2010) movie",
        "calibration": 405,
        "type": "ADAP"
    },
    "23": {
        "string": "statement: I am not the body",
        "calibration": 600,
        "type": "ADAP"
    },
    "24": {
        "string": "mindfulness",
        "calibration": 555,
        "type": "ADAP"
    },
    "25": {
        "string": "Ralph Lauren clothing company",
        "calibration": 197,
        "type": "ADAP"
    },
    "26": {
        "string": "deciding to do therapy, seeking help from a therapist",
        "calibration": 280,
        "type": "ADAP"
    },
    "27": {
        "string": "rainbow",
        "calibration": 300,
        "type": "ADAP"
    },
    "28": {
        "string": "mathematics",
        "calibration": 410,
        "type": "ADAP"
    },
    "29": {
        "string": "Ramana Maharshi",
        "calibration": 600,
        "type": "ADAP"
    },
    "30": {
        "string": "ChatGPT (AI chatbot)",
        "calibration": 190,
        "type": "ADAP"
    },
    "31": {
        "string": "Louis Armstrong - What A Wonderful World",
        "calibration": 540,
        "type": "ADAP"
    },
    "32": {
        "string": "Microsoft",
        "calibration": 199,
        "type": "ADAP"
    },
    "33": {
        "string": "Google",
        "calibration": 195,
        "type": "ADAP"
    },
    "34": {
        "string": "consciousness calibration is all one needs to reach enlightenment",
        "calibration": 400,
        "type": "ADAP"
    },
    "35": {
        "string": "Having a business",
        "calibration": 285,
        "type": "ADAP"
    },
    "36": {
        "string": "George Floyd",
        "calibration": 35,
        "type": "ADAP"
    },
    "37": {
        "string": "fight fire with fire",
        "calibration": 200,
        "type": "ADAP"
    },
    "38": {
        "string": "apology",
        "calibration": 250,
        "type": "ADAP"
    },
    "39": {
        "string": "monkhood / monasticism",
        "calibration": 450,
        "type": "ADAP"
    },
    "40": {
        "string": "chocolate",
        "calibration": 290,
        "type": "ADAP"
    },
    "41": {
        "string": "chocolate confections",
        "calibration": 390,
        "type": "ADAP"
    },
    "42": {
        "string": "tarriffs",
        "calibration": 200,
        "type": "ADAP"
    },
    "43": {
        "string": "programmer",
        "calibration": 370,
        "type": "ADAP"
    },
    "44": {
        "string": "Having one's Level of consciousness calibrated",
        "calibration": 500,
        "type": "ADAP"
    },
    "45": {
        "string": "Making your bed every morning",
        "calibration": 205,
        "type": "ADAP"
    },
    "46": {
        "string": "energy field of an in-person 12-step meeting",
        "calibration": 490,
        "type": "ADAP"
    },
    "47": {
        "string": "contemplating 'I'",
        "calibration": 570,
        "type": "ADAP"
    },
    "48": {
        "string": "reverse engineering",
        "calibration": 205,
        "type": "ADAP"
    },
    "49": {
        "string": "violence",
        "calibration": 15,
        "type": "ADAP"
    },
    "50": {
        "string": "existence is forever",
        "calibration": 690,
        "type": "ADAP"
    },
    "51": {
        "string": "appearance is not essence",
        "calibration": 540,
        "type": "ADAP"
    },
    "52": {
        "string": "dancing as a spiritual practice",
        "calibration": 500,
        "type": "ADAP"
    },
    "53": {
        "string": "listening to a Dr. Hawkins lecture",
        "calibration": 480,
        "type": "ADAP"
    },
    "54": {
        "string": "adopting pets from animal shelters as rescues",
        "calibration": 350,
        "type": "ADAP"
    },
    "55": {
        "string": "ignatian retreat",
        "calibration": 350,
        "type": "ADAP"
    },
    "56": {
        "string": "wrapping presents and giving them to each other for christmas",
        "calibration": 350,
        "type": "ADAP"
    },
    "57": {
        "string": "large-scale biogas production",
        "calibration": 194,
        "type": "ADAP"
    },
    "58": {
        "string": "small-scale biogas production",
        "calibration": 204,
        "type": "ADAP"
    },
    "59": {
        "string": "germany",
        "calibration": 220,
        "type": "ADAP"
    },
    "60": {
        "string": "inner child (ACA concept)",
        "calibration": 395,
        "type": "ADAP"
    },
    "60": {
        "string": "inner child (ACA concept)",
        "calibration": 395,
        "type": "ADAP"
    },
    "61": {
        "string": "visualizing oneself being protected by Doc",
        "calibration": 560,
        "type": "ADAP"
    },
    "62": {
        "string": "law of attraction",
        "calibration": 200,
        "type": "ADAP"
    },
    "63": {
        "string": "law of assumption (for manifesting)",
        "calibration": 210,
        "type": "ADAP"
    },
    "64": {
        "string": "fear of death",
        "calibration": 5,
        "type": "ADAP"
    },
    "65": {
        "string": "panic attack",
        "calibration": 10,
        "type": "ADAP"
    },
    "66": {
        "string": "self-punishment",
        "calibration": 20,
        "type": "ADAP"
    },
    "67": {
        "string": "intrusive thought",
        "calibration": 25,
        "type": "ADAP"
    },
    "68": {
        "string": "dumpster diving for food",
        "calibration": 30,
        "type": "ADAP"
    },
    "69": {
        "string": "self-judging instinct",
        "calibration": 35,
        "type": "ADAP"
    }
    ,
    "70": {
        "string": "lust",
        "calibration": 40,
        "type": "ADAP"
    },
    "71": {
        "string": "human body",
        "calibration": 45,
        "type": "ADAP"
    },
    "72": {
        "string": "ritual instinct",
        "calibration": 55,
        "type": "ADAP"
    },
    "73": {
        "string": "giving away your power",
        "calibration": 60,
        "type": "ADAP"
    },
    "74": {
        "string": "laziness",
        "calibration": 65,
        "type": "ADAP"
    },
    "75": {
        "string": "ban on religion",
        "calibration": 70,
        "type": "ADAP"
    },
    "76": {
        "string": "self-sabaotage",
        "calibration": 75,
        "type": "ADAP"
    },
    "77": {
        "string": "feeling lonely",
        "calibration": 80,
        "type": "ADAP"
    },
    "78": {
        "string": "melancholy",
        "calibration": 85,
        "type": "ADAP"
    },
    "79": {
        "string": "hell does not exist",
        "calibration": 90,
        "type": "ADAP"
    },
    "80": {
        "string": "snoring",
        "calibration": 95,
        "type": "ADAP"
    },
    "81": {
        "string": "poverty",
        "calibration": 100,
        "type": "ADAP"
    },
    "82": {
        "string": "cheating (infidelity)",
        "calibration": 110,
        "type": "ADAP"
    },
    "83": {
        "string": "mass media's average level of truth",
        "calibration": 115,
        "type": "ADAP"
    },
    "84": {
        "string": "low self esteem",
        "calibration": 120,
        "type": "ADAP"
    },
    "85": {
        "string": "awkward silence",
        "calibration": 125,
        "type": "ADAP"
    },
    "86": {
        "string": "all suffering is due to external events",
        "calibration": 130,
        "type": "ADAP"
    },
    "87": {
        "string": "heartache",
        "calibration": 135,
        "type": "ADAP"
    },
    "88": {
        "string": "inability to calibrate",
        "calibration": 140,
        "type": "ADAP"
    },
    "89": {
        "string": "binge-watching",
        "calibration": 145,
        "type": "ADAP"
    },
    "90": {
        "string": "dirty talk between lovers",
        "calibration": 150,
        "type": "ADAP"
    },
    "91": {
        "string": "stubborn",
        "calibration": 155,
        "type": "ADAP"
    },
    "92": {
        "string": "all suffering is self-created",
        "calibration": 160,
        "type": "ADAP"
    },
    "93": {
        "string": "superiority complex",
        "calibration": 165,
        "type": "ADAP"
    },
    "94": {
        "string": "attraction to things",
        "calibration": 170,
        "type": "ADAP"
    },
    "95": {
        "string": "greenhouse gas theory of global warming",
        "calibration": 175,
        "type": "ADAP"
    },
    "96": {
        "string": "justified force",
        "calibration": 180,
        "type": "ADAP"
    },
    "97": {
        "string": "urgent",
        "calibration": 190,
        "type": "ADAP"
    },
    "98": {
        "string": "you have to make love work",
        "calibration": 195,
        "type": "ADAP"
    },
    "99": {
        "string": "love just is and works",
        "calibration": 440,
        "type": "ADAP"
    },
    "100": {
        "string": "pretending every desire is fullfilled (spiritual practice)",
        "calibration": 560,
        "type": "ADAP"
    },
    "101": {
        "string": "wishful thinking",
        "calibration": 145,
        "type": "ADAP"
    },
    "102": {
        "string": "being a blessing to the world",
        "calibration": 500,
        "type": "ADAP"
    },
    "103": {
        "string": "Leonardo DiCaprio",
        "calibration": 200,
        "type": "ADAP"
    },
    "104": {
        "string": "Crucifix",
        "calibration": 495,
        "type": "ADAP"
    },
    "105": {
        "string": "contemplation: How am I aware or even know that I exist?",
        "calibration": 590,
        "type": "ADAP"
    },
    "106": {
        "string": "Consciousness is God",
        "calibration": 1000,
        "type": "ADAP"
    },
    "107": {
        "string": "One Big Beautiful Bill Act ",
        "calibration": 203,
        "type": "ADAP"
    },
    "108": {
        "string": "benevolent dictatorship",
        "calibration": 200,
        "type": "ADAP"
    },
    "109": {
        "string": "Forgive everything that is witnessed and experienced, no matter what",
        "calibration": 560,
        "type": "ADAP"
    },
    "110": {
        "string": "dogs",
        "calibration": 250,
        "type": "ADAP"
    },
    "111": {
        "string": "cow",
        "calibration": 195,
        "type": "ADAP"
    },
    "112": {
        "string": "innocence",
        "calibration": 600,
        "type": "ADAP"
    },
    "113": {
        "string": "faith",
        "calibration": 520,
        "type": "ADAP"
    },
    "114": {
        "string": "light-bulb moment",
        "calibration": 400,
        "type": "ADAP"
    },
    "115": {
        "string": "aesthetic appreciation",
        "calibration": 370,
        "type": "ADAP"
    },
    "116": {
        "string": "Netflix",
        "calibration": 195,
        "type": "ADAP"
    },
    "117": {
        "string": "Sacrament of Reconciliation or Confession",
        "calibration": 495,
        "type": "ADAP"
    },
    "118": {
        "string": "Fox News",
        "calibration": 200,
        "type": "ADAP"
    },
    "119": {
        "string": "Alex Jones",
        "calibration": 160,
        "type": "ADAP"
    },
    "120": {
        "string": "statement: loving others is no substitute for loving yourself",
        "calibration": 570,
        "type": "ADAP"
    },
    "121": {
        "string": "all I need is just to be",
        "calibration": 460,
        "type": "ADAP"
    },
    "122": {
        "string": "letting go by shaking",
        "calibration": 530,
        "type": "ADAP"
    },
    "123": {
        "string": "Letting go for emotional liberation and transcendence of human limitations",
        "calibration": 540,
        "type": "ADAP"
    },
    "124": {
        "string": "Letting go as a means to pursue enlightenment",
        "calibration": 570,
        "type": "ADAP"
    },
    "125": {
        "string": "I am my own worst enemy",
        "calibration": 440,
        "type": "ADAP"
    },
    "126": {
        "string": "laughing (energy)",
        "calibration": 330,
        "type": "ADAP"
    },
    "127": {
        "string": "one can only go as high as they have been low",
        "calibration": 590,
        "type": "ADAP"
    },
    "128": {
        "string": "Your room is an externalization of your mind",
        "calibration": 400,
        "type": "ADAP"
    },
    "129": {
        "string": "social worker (occupation)",
        "calibration": 245,
        "type": "ADAP"
    },
    "130": {
        "string": "energy to 'bless them that curse you'",
        "calibration": 550,
        "type": "ADAP"
    },
    "131": {
        "string": "talking to God like to a therapist",
        "calibration": 460,
        "type": "ADAP"
    },
    "132": {
        "string": "position: Doc shouldn't have taught consciousness calibration",
        "calibration": 60,
        "type": "ADAP"
    },
    "133": {
        "string": "refusing negativity",
        "calibration": 560,
        "type": "ADAP"
    },
    "134": {
        "string": "peace be with you",
        "calibration": 570,
        "type": "ADAP"
    },
    "135": {
        "string": "redemption",
        "calibration": 540,
        "type": "ADAP"
    },
    "136": {
        "string": "dictator",
        "calibration": 50,
        "type": "ADAP"
    },
    "137": {
        "string": "Power vs. Force, by David R. Hawkins",
        "calibration": 570,
        "type": "ADAP"
    },
    "138": {
        "string": "Alan Watts",
        "calibration": 390,
        "type": "ADAP"
    },
    "139": {
        "string": "Tao Te Ching by Lao Tzu",
        "calibration": 550,
        "type": "ADAP"
    },
    "140": {
        "string": "The Matrix (1999) movie",
        "calibration": 150,
        "type": "ADAP"
    },
    "141": {
        "string": "may all be free of the weight of this world",
        "calibration": 610,
        "type": "ADAP"
    },
    "142": {
        "string": "walking (exercise)",
        "calibration": 220,
        "type": "ADAP"
    },
    "143": {
        "string": "Wikipedia",
        "calibration": 203,
        "type": "ADAP"
    },
    "144": {
        "string": "the Presence of God as self-esteem",
        "calibration": 560,
        "type": "ADAP"
    },
    "145": {
        "string": "past life regression",
        "calibration": 290,
        "type": "ADAP"
    },
    "146": {
        "string": "juggling",
        "calibration": 215,
        "type": "ADAP"
    },
    "147": {
        "string": "not seeing sin in anyone / anywhere",
        "calibration": 575,
        "type": "ADAP"
    },
    "148": {
        "string": "What can frighten me, when I let all things be exactly as they are?",
        "calibration": 560,
        "type": "ADAP"
    },
    "149": {
        "string": "to comfort another ",
        "calibration": 350,
        "type": "ADAP"
    },
    "150": {
        "string": "perfecting one's intention",
        "calibration": 550,
        "type": "ADAP"
    },
    "151": {
        "string": "being a blessing to this world",
        "calibration": 500,
        "type": "ADAP"
    },
    "152": {
        "string": "the most dangerous thing on the planet is the spiritualized ego",
        "calibration": 550,
        "type": "ADAP"
    },
    "153": {
        "string": "God's will is All there is",
        "calibration": 580,
        "type": "ADAP"
    },
    "153": {
        "string": "I let Christ let go for me",
        "calibration": 580,
        "type": "ADAP"
    },
    "154": {
        "string": "pathway of moderation",
        "calibration": 560,
        "type": "ADAP"
    },
    "155": {
        "string": "alcohoilism can be cured",
        "calibration": 550,
        "type": "ADAP"
    },
    "156": {
        "string": "the whole universe shames you",
        "calibration": 0,
        "type": "ADAP"
    },
    "157": {
        "string": "the whole universe loves you",
        "calibration": 599.99,
        "type": "ADAP"
    },
    "158": {
        "string": "The degree of one’s faith in a practice or pathway empowers it",
        "calibration": 510,
        "type": "ADAP"
    },
    "159": {
        "string": "I love you",
        "calibration": 490,
        "type": "ADAP"
    },
    "160": {
        "string": "I bless you",
        "calibration": 505,
        "type": "ADAP"
    },
    "161": {
        "string": "wear the world like a loose garment",
        "calibration": 490,
        "type": "ADAP"
    },
    "162": {
        "string": "aluminum recycling",
        "calibration": 210,
        "type": "ADAP"
    },
    "163": {
        "string": "plastic bottle recyling",
        "calibration": 195,
        "type": "ADAP"
    },
    "164": {
        "string": "paper/cardboard recycling",
        "calibration": 190,
        "type": "ADAP"
    },
    "165": {
        "string": "glass bottle recycling",
        "calibration": 190,
        "type": "ADAP"
    },
    "166": {
        "string": "happiness",
        "calibration": 385,
        "type": "ADAP"
    },
    "167": {
        "string": "seeing the world through rose-tinted glasses",
        "calibration": 185,
        "type": "ADAP"
    },
    "168": {
        "string": "seeing only the best in others or oneself",
        "calibration": 190,
        "type": "ADAP"
    },
    "169": {
        "string": "no mercy for the small-self",
        "calibration": 30,
        "type": "ADAP"
    },
    "170": {
        "string": "a human with bad karma can reincarnate as an animal",
        "calibration": 100,
        "type": "ADAP"
    },
    "171": {
        "string": "ignoring someone so they will like you",
        "calibration": 150,
        "type": "ADAP"
    },
    "172": {
        "string": "karmic merit",
        "calibration": 470,
        "type": "ADAP"
    },
    "173": {
        "string": "matra: I love myself",
        "calibration": 530,
        "type": "ADAP"
    }
}

ADAP_MAP_OF_CONSCIOUSNESS: Dict[str, Dict[str, object]] = {
    "ADAPmoc_1000": {
        "string": "The Absolute",
        "calibration": 1000,
        "type": "ADAP",
    },
    "ADAPmoc_900": {
        "string": "Final Door",
        "calibration": 900,
        "type": "ADAP",
    },
    "ADAPmoc_850": {
        "string": "Allness",
        "calibration": 850,
        "type": "ADAP",
    },
    "ADAPmoc_800": {
        "string": "The Void",
        "calibration": 800,
        "type": "ADAP",
    },
    "ADAPmoc_750": {
        "string": "Full Enlightenment",
        "calibration": 750,
        "type": "ADAP",
    },
    "ADAPmoc_700": {
        "string": "Eternal Life",
        "calibration": 700,
        "type": "ADAP",
    },
    "ADAPmoc_600": {
        "string": "Enlightenment",
        "calibration": 600,
        "type": "ADAP",
    },
    "ADAPmoc_550": {
        "string": "Unconditional Love",
        "calibration": 550,
        "type": "ADAP",
    },
    "ADAPmoc_500": {
        "string": "Love",
        "calibration": 500,
        "type": "ADAP",
    },
    "ADAPmoc_450": {
        "string": "Nobleness",
        "calibration": 450,
        "type": "ADAP",
    },
    "ADAPmoc_400": {
        "string": "Intellect",
        "calibration": 400,
        "type": "ADAP",
    },
    "ADAPmoc_350": {
        "string": "Acceptance",
        "calibration": 350,
        "type": "ADAP",
    },
    "ADAPmoc_300": {
        "string": "Willingness",
        "calibration": 300,
        "type": "ADAP",
    },
    "ADAPmoc_250": {
        "string": "Higher Mind / Trust",
        "calibration": 250,
        "type": "ADAP",
    },
    "ADAPmoc_200": {
        "string": "Integrity / Courage",
        "calibration": 200,
        "type": "ADAP",
    },
    "ADAPmoc_190": {
        "string": "Lower Mind",
        "calibration": 190,
        "type": "ADAP",
    },
    "ADAPmoc_170": {
        "string": "Pride",
        "calibration": 170,
        "type": "ADAP",
    },
    "ADAPmoc_150": {
        "string": "Anger / Ego",
        "calibration": 150,
        "type": "ADAP",
    },
    "ADAPmoc_125": {
        "string": "Desire",
        "calibration": 125,
        "type": "ADAP",
    },
    "ADAPmoc_100": {
        "string": "Fear",
        "calibration": 100,
        "type": "ADAP",
    },
    "ADAPmoc_75": {
        "string": "Grief",
        "calibration": 75,
        "type": "ADAP",
    },
    "ADAPmoc_50": {
        "string": "Apathy",
        "calibration": 50,
        "type": "ADAP",
    },
    "ADAPmoc_25": {
        "string": "Guilt",
        "calibration": 25,
        "type": "ADAP",
    },
    "ADAPmoc_15": {
        "string": "Shame",
        "calibration": 15,
        "type": "ADAP",
    },
    "ADAPmoc_5": {
        "string": "Terror / Death",
        "calibration": 5,
        "type": "ADAP",
    },
    "ADAPmoc_1": {
        "string": "Spiritual Darkness",
        "calibration": 1,
        "type": "ADAP",
    },
}
ADAP_MOC_ORDERED_KEYS: List[str] = sorted(
    ADAP_MAP_OF_CONSCIOUSNESS.keys(),
    key=lambda k: ADAP_MAP_OF_CONSCIOUSNESS[k]["calibration"],
    reverse=True,
)
ADAP_MOC_PROMPT_ENTRIES = "\n".join(
    f"{entry['calibration']}: {entry['string']}"
    for entry in (ADAP_MAP_OF_CONSCIOUSNESS[key] for key in ADAP_MOC_ORDERED_KEYS)
)
ADAP_MOC_NEGATIVE_HINT_PHRASES = {
    "double standard",
    "double standards",
    "unfair",
    "unfairness",
    "bias",
    "biased",
    "hypocrisy",
    "hypocrite",
    "hypocritical",
    "dishonest",
    "deceit",
    "deceitful",
    "corrupt",
    "corruption",
    "unjust",
    "discriminatory",
    "discriminate",
}

ADAP_MOC_TOKEN_HINTS: List[Tuple[int, Set[str]]] = [
    (
        160,
        {
            "want",
            "wants",
            "wanting",
            "wanted",
            "desire",
            "desires",
            "desiring",
            "needy",
            "need",
            "needs",
            "yearn",
            "yearning",
            "crave",
            "craving",
        },
    ),
    (
        125,
        {
            "fear",
            "fears",
            "afraid",
            "scared",
            "worried",
            "worry",
            "anxious",
            "anxiety",
            "panic",
            "panicking",
            "terrified",
        },
    ),
    (
        75,
        {
            "apathy",
            "hopeless",
            "hopelessness",
            "numb",
            "numbness",
            "limbo",
            "stuck",
        },
    ),
]


def _detect_adap_hint_loc(text: str) -> int | None:
    normalized = normalize_term(text)
    tokens = tokenize_for_overlap(text)
    for phrase in ADAP_MOC_NEGATIVE_HINT_PHRASES:
        if phrase in normalized:
            return 150
    for loc, keywords in ADAP_MOC_TOKEN_HINTS:
        if tokens & keywords:
            return loc
    return None


def _closest_adap_key_to_loc(target_loc: int) -> str:
    """Return the ADAP map key whose calibration is closest to the target."""
    best_key = ADAP_MOC_ORDERED_KEYS[-1]
    best_diff = float("inf")
    for key in ADAP_MOC_ORDERED_KEYS:
        diff = abs(int(ADAP_MAP_OF_CONSCIOUSNESS[key]["calibration"]) - target_loc)
        if diff < best_diff or (diff == best_diff and ADAP_MAP_OF_CONSCIOUSNESS[key]["calibration"] < ADAP_MAP_OF_CONSCIOUSNESS[best_key]["calibration"]):
            best_diff = diff
            best_key = key
    return best_key


DATABASE_REFERENCES = "\n".join(
    f"{index + 1}. {entry['string']} (calibration {entry['calibration']})"
    for index, entry in enumerate(database.values())
)

GPT2_TOKENIZER: GPT2Tokenizer | None = None
GPT2_MODEL: GPT2LMHeadModel | None = None
_GPT_WARNING_EMITTED = False
_GPT_READY_EMITTED = False
_OPENAI_WARNING_EMITTED = False
_OPENAI_READY_EMITTED = False
OPENAI_DISABLED_REASON = "" if OPENAI_API_KEY else "set OPENAI_API_KEY to enable OpenAI keywords"


def _openai_model_uses_chat(model_name: str) -> bool:
    """Heuristic to determine whether to call the chat or legacy completions endpoint."""
    normalized = model_name.strip().lower()
    if not normalized:
        return False
    if normalized.startswith("text-") or normalized.startswith("code-"):
        return False
    if "instruct" in normalized and normalized.startswith("gpt-3.5"):
        return False
    # Assume everything else (e.g., gpt-3.5-turbo, gpt-4 variants, gpt-5) uses chat completions.
    return True


def _openai_model_uses_responses(model_name: str) -> bool:
    """Return True if the model should be queried via the Responses API."""
    normalized = model_name.strip().lower()
    if not normalized:
        return False
    response_prefixes = (
        "gpt-4.1",
        "gpt-5",
        "o4",
    )
    response_contains = ("-mini", "responses")
    if any(normalized.startswith(prefix) for prefix in response_prefixes):
        return True
    if any(token in normalized for token in response_contains):
        return True
    return False


def _heuristic_context_related(statement: str, entry_text: str) -> bool:
    statement_tokens = tokenize_for_overlap(statement)
    entry_tokens = entry_tokens_for(entry_text)
    if not statement_tokens or not entry_tokens:
        return False
    polarity_match = has_negation(statement) == has_negation(entry_text)
    if not polarity_match:
        return False
    statement_token_count = len(statement_tokens)
    entry_token_count = len(entry_tokens)
    if statement_token_count <= 1 or entry_token_count <= 1:
        min_overlap = 1
    elif min(statement_token_count, entry_token_count) <= 2:
        min_overlap = 1
    else:
        min_overlap = 2
    overlap = len(statement_tokens & entry_tokens)
    if overlap < min_overlap:
        if overlap == 1:
            ratio = SequenceMatcher(None, normalize_term(statement), normalize_term(entry_text)).ratio()
            if ratio >= 0.6:
                return True
        return False
    if overlap < 2:
        if statement_token_count <= 2 or entry_token_count <= 2:
            return True
        if max(statement_token_count, entry_token_count) >= 4:
            ratio = SequenceMatcher(None, normalize_term(statement), normalize_term(entry_text)).ratio()
            return ratio >= 0.65
    return True


def warn_gpt_disabled(extra_reason: str | None = None) -> None:
    """Emit a single warning describing why GPT keyword expansion is disabled."""
    global _GPT_WARNING_EMITTED
    if _GPT_WARNING_EMITTED:
        return
    reason = extra_reason or GPT_DISABLED_REASON
    if not reason:
        return
    print(f"\n[info] GPT keyword expansion disabled: {reason}", file=sys.stderr)
    _GPT_WARNING_EMITTED = True


def announce_gpt_ready() -> None:
    """Emit a single message when GPT keyword expansion is active."""
    global _GPT_READY_EMITTED
    if _GPT_READY_EMITTED:
        return
    print("\n[info] GPT keyword expansion enabled (GPT-2, via transformers).", file=sys.stderr)
    _GPT_READY_EMITTED = True


def warn_openai_disabled(extra_reason: str | None = None) -> None:
    """Emit a warning when OpenAI keyword expansion is unavailable."""
    global _OPENAI_WARNING_EMITTED
    if _OPENAI_WARNING_EMITTED:
        return
    reason = extra_reason or OPENAI_DISABLED_REASON
    if not reason:
        return
    print(f"\n[info] OpenAI keyword expansion disabled: {reason}", file=sys.stderr)
    _OPENAI_WARNING_EMITTED = True


def announce_openai_ready() -> None:
    """Emit a message when OpenAI keyword expansion is active."""
    global _OPENAI_READY_EMITTED
    if _OPENAI_READY_EMITTED:
        return
    print(f"\n[info] OpenAI keyword expansion enabled ({OPENAI_MODEL}).", file=sys.stderr)
    _OPENAI_READY_EMITTED = True


def normalize_term(term: str) -> str:
    """Normalize a term for consistent matching."""
    lowered = term.strip().lower()
    lowered = re.sub(r"[^a-z0-9\s]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _token_variants(token: str) -> Set[str]:
    """Return token plus simple singular/plural variants for overlap matching."""
    variants: Set[str] = {token}
    if len(token) > 3 and not token.endswith("ss"):
        if token.endswith("ies"):
            variants.add(token[:-3] + "y")
        if token.endswith("es"):
            variants.add(token[:-2])
        if token.endswith("s"):
            variants.add(token[:-1])
    return {variant for variant in variants if variant}


def tokenize_for_overlap(text: str) -> Set[str]:
    """Tokenize text and add base-form variants for fuzzy overlap checks."""
    tokens: Set[str] = set()
    for token in re.findall(r"[A-Za-z]+", text.lower()):
        if token in STOPWORDS or token in BANNED_GENERIC_TOKENS:
            continue
        tokens.update(_token_variants(token))
    return tokens

DATABASE_NORMALIZED_MAP = {
    normalize_term(entry["string"]): entry for entry in database.values()
}
DATABASE_NORMALIZED_TO_KEY = {
    normalize_term(entry["string"]): key for key, entry in database.items()
}
DATABASE_ENTRY_TOKENS = {
    normalized: tokenize_for_overlap(normalized) for normalized in DATABASE_NORMALIZED_MAP.keys()
}


def _candidate_keys_for_tokens(term_tokens: Set[str], data: Database) -> Set[str]:
    if not term_tokens:
        return set(data.keys())
    index = _token_index_for_data(data)
    candidates: Set[str] = set()
    for token in term_tokens:
        keys = index.get(token)
        if not keys:
            continue
        for key in keys:
            if key in data:
                candidates.add(key)
    if not candidates:
        return set(data.keys())
    return candidates


def dedupe_preserve(items: Iterable[str]) -> List[str]:
    """Remove duplicates while preserving the original ordering."""
    seen: Set[str] = set()
    deduped: List[str] = []
    for item in items:
        normalized = normalize_term(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(item.strip())
    return deduped


def entry_tokens_for(name: str) -> Set[str]:
    normalized = normalize_term(name)
    tokens = DATABASE_ENTRY_TOKENS.get(normalized)
    if tokens is None:
        tokens = tokenize_for_overlap(normalized)
        DATABASE_ENTRY_TOKENS[normalized] = tokens
    return tokens


def _build_token_index(data: Database) -> Dict[str, Set[str]]:
    index: Dict[str, Set[str]] = defaultdict(set)
    for key, entry in data.items():
        tokens = entry_tokens_for(entry["string"])
        for token in tokens:
            if token:
                index[token].add(key)
    return index


_DATASET_TOKEN_INDEX_CACHE: Dict[int, Dict[str, Set[str]]] = {}
DATABASE_TOKEN_INDEX = _build_token_index(database)
_DATASET_TOKEN_INDEX_CACHE[id(database)] = DATABASE_TOKEN_INDEX


def _token_index_for_data(data: Database) -> Dict[str, Set[str]]:
    cached = _DATASET_TOKEN_INDEX_CACHE.get(id(data))
    if cached is not None:
        return cached
    index = _build_token_index(data)
    _DATASET_TOKEN_INDEX_CACHE[id(data)] = index
    return index


def is_stopword(term: str) -> bool:
    """Return True if the normalized term is a stopword."""
    if not term:
        return True
    return term in STOPWORDS


DESCRIPTOR_WORD_LIMIT = 4


def _clean_descriptor_text(text: str) -> str:
    """Trim quotes/punctuation and non-alpha characters that surround GPT tokens."""
    cleaned = text.strip()
    cleaned = cleaned.strip("`\"'“”‘’()[]{}")
    cleaned = re.sub(r"[!?]+$", "", cleaned)
    cleaned = re.sub(r"[^A-Za-z -]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def filter_descriptors(words: Iterable[str]) -> List[str]:
    """Filter out stopwords, long phrases, and filler tokens."""
    filtered: List[str] = []
    for word in dedupe_preserve(words):
        cleaned = _clean_descriptor_text(word)
        normalized = normalize_term(cleaned)
        if len(normalized) < 3:
            continue
        if len(normalized.split()) > DESCRIPTOR_WORD_LIMIT:
            continue
        if not re.fullmatch(r"[a-z][a-z -]*", normalized):
            continue
        if sum(ch.isalpha() for ch in normalized) < 3:
            continue
        parts = normalized.split()
        if any(len(part) < 3 for part in parts):
            continue
        if any(is_stopword(part) for part in parts):
            continue
        if any(bad in normalized for bad in BANNED_DESCRIPTOR_SUBSTRINGS):
            continue
        filtered.append(cleaned)
    return filtered


def exclude_terms(keywords: Sequence[str], disallowed: Iterable[str]) -> List[str]:
    """Remove keywords whose normalized form appears in disallowed."""
    disallowed_norm = {normalize_term(term) for term in disallowed if normalize_term(term)}
    result: List[str] = []
    seen: Set[str] = set()
    for word in keywords:
        norm = normalize_term(word)
        if not norm or norm in disallowed_norm or norm in seen:
            continue
        seen.add(norm)
        result.append(word)
    return result


def parse_keywords(raw_text: str) -> List[str]:
    """Split a raw GPT response into individual keywords."""
    if not raw_text:
        return []
    candidates = re.split(r"[\n,;]+", raw_text)
    keywords: List[str] = []
    for candidate in candidates:
        cleaned = re.sub(r"^\s*\d+[\.\)]\s*", "", candidate)
        cleaned = cleaned.strip(" -*•\t")
        if cleaned:
            keywords.append(cleaned)
    if keywords:
        return keywords
    # Fall back to alphanumeric tokens if commas/newlines were not helpful.
    return re.findall(r"[A-Za-z][A-Za-z0-9' -]*", raw_text)


def fallback_keywords_from_text(text: str, desired_count: int) -> List[str]:
    """Fallback keyword extraction that simply reuses unique words from the text."""
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9']*", text.lower())
    tokens = dedupe_preserve(tokens)
    if not tokens:
        tokens = ["context", "intent", "impact", "behavior", "cause", "effect"]
    return tokens[:desired_count]


def fallback_keywords_from_terms(terms: Sequence[str], desired_count: int) -> List[str]:
    """Fallback tertiary keywords derived by splitting the unmatched secondary terms."""
    components: List[str] = []
    for term in terms:
        components.extend(re.findall(r"[A-Za-z][A-Za-z0-9']*", term.lower()))
    if not components:
        components = list(terms)
    defaults = [
        "symbolism",
        "meaning",
        "tradition",
        "culture",
        "ritual",
        "belief",
        "ethic",
        "behavior",
        "principle",
    ]
    components = dedupe_preserve(components + defaults)
    return components[:desired_count]


def ensure_gpt2_loaded() -> bool:
    """Lazy load GPT-2 if transformers/torch are available."""
    global GPT2_MODEL, GPT2_TOKENIZER
    if GPT2_MODEL is not None and GPT2_TOKENIZER is not None:
        return True
    if GPT2Tokenizer is None or GPT2LMHeadModel is None or torch is None:
        warn_gpt_disabled()
        return False
    try:
        GPT2_TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")
        GPT2_MODEL = GPT2LMHeadModel.from_pretrained("gpt2")
        GPT2_MODEL.eval()
        announce_gpt_ready()
        return True
    except Exception:
        warn_gpt_disabled("failed to download GPT-2 weights from Hugging Face; check your network connectivity")
        # Model download might fail (e.g., offline environments). We'll fall back.
        GPT2_MODEL = None
        GPT2_TOKENIZER = None
        return False


def generate_with_gpt2(prompt: str, max_new_tokens: int = 64) -> str | None:
    """Use GPT-2 to generate text, returning None if GPT-2 is unavailable."""
    if not ensure_gpt2_loaded():
        return None
    assert GPT2_MODEL is not None and GPT2_TOKENIZER is not None  # For type-checkers.
    try:
        max_context = getattr(GPT2_MODEL.config, "n_positions", 1024) or 1024
        max_prompt_tokens = max(1, max_context - max_new_tokens)
        inputs = GPT2_TOKENIZER(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_tokens,
        )
        with torch.no_grad():
            output_ids = GPT2_MODEL.generate(
                **inputs,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                no_repeat_ngram_size=2,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                pad_token_id=GPT2_TOKENIZER.eos_token_id,
            )
        generated_text = GPT2_TOKENIZER.decode(output_ids[0], skip_special_tokens=True)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]
        return generated_text.strip()
    except Exception:
        return None


def language_model_completion(prompt: str, max_new_tokens: int = 64) -> str | None:
    """Call OpenAI first, then fall back to GPT-2, returning the generated text."""
    text = generate_with_openai(prompt, max_new_tokens=max_new_tokens)
    if text:
        return text
    return generate_with_gpt2(prompt, max_new_tokens=max_new_tokens)


def generate_with_openai(prompt: str, max_new_tokens: int = 64) -> str | None:
    """Use OpenAI's completions API to generate text, returning None if unavailable."""
    if not OPENAI_API_KEY:
        warn_openai_disabled()
        return None
    use_responses_endpoint = _openai_model_uses_responses(OPENAI_MODEL)
    use_chat_endpoint = _openai_model_uses_chat(OPENAI_MODEL) and not use_responses_endpoint
    if use_responses_endpoint:
        url = f"{OPENAI_API_BASE.rstrip('/')}/responses"
        payload = {
            "model": OPENAI_MODEL,
            "input": prompt,
        }
    elif use_chat_endpoint:
        url = f"{OPENAI_API_BASE.rstrip('/')}/chat/completions"
        payload = {
            "model": OPENAI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": 0.25,
            "n": 1,
        }
    else:
        url = f"{OPENAI_API_BASE.rstrip('/')}/completions"
        payload = {
            "model": OPENAI_MODEL,
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": 0.25,
            "n": 1,
        }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            response_payload = json.load(resp)
    except urllib.error.HTTPError as exc:
        warn_openai_disabled(f"HTTP error {exc.code} from OpenAI API")
        return None
    except urllib.error.URLError as exc:
        warn_openai_disabled(f"network error contacting OpenAI API ({exc.reason})")
        return None
    except Exception as exc:
        warn_openai_disabled(f"error contacting OpenAI API ({exc.__class__.__name__})")
        return None
    text = ""
    if use_responses_endpoint:
        output_text = response_payload.get("output_text")
        if isinstance(output_text, list) and output_text:
            text = "\n".join(str(part) for part in output_text if part)
        if not text:
            output = response_payload.get("output")
            if isinstance(output, list):
                for block in output:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "message":
                        continue
                    content = block.get("content", [])
                    if isinstance(content, list):
                        for chunk in content:
                            if not isinstance(chunk, dict):
                                continue
                            chunk_type = chunk.get("type")
                            if chunk_type not in {"text", "output_text"}:
                                continue
                            text = chunk.get("text", "") or ""
                            if text:
                                break
                    if text:
                        break
    else:
        choices = response_payload.get("choices")
        if not choices:
            warn_openai_disabled("OpenAI API returned no choices")
            return None
        if use_chat_endpoint:
            message = choices[0].get("message", {})
            if isinstance(message, dict):
                text = message.get("content", "") or ""
        else:
            text = choices[0].get("text", "")
    if text:
        announce_openai_ready()
    return text.strip()


def get_keywords_from_prompt(
    prompt: str,
    desired_count: int,
    fallback_factory: Callable[[], List[str]],
    max_new_tokens: int = 96,
) -> List[str]:
    """Generate keywords via GPT-2 if possible; otherwise use the fallback factory."""
    generated = generate_with_openai(prompt, max_new_tokens=max_new_tokens)
    if not generated:
        generated = generate_with_gpt2(prompt, max_new_tokens=max_new_tokens)
    keywords: List[str] = []
    if generated:
        keywords = filter_descriptors(parse_keywords(generated))
        if not keywords:
            keywords = filter_descriptors(fallback_keywords_from_text(generated, desired_count * 2))
    if len(keywords) < max(1, desired_count // 2):
        fallback_words = filter_descriptors(fallback_factory())
        existing = {normalize_term(k) for k in keywords}
        for word in fallback_words:
            norm = normalize_term(word)
            if not norm or norm in existing:
                continue
            keywords.append(word)
            existing.add(norm)
            if len(keywords) >= desired_count:
                break
    if not keywords:
        keywords = filter_descriptors(fallback_factory())
    return keywords[:desired_count]


def generate_secondary_keywords(statement: str) -> List[str]:
    """Produce up to 10 descriptive keywords for the user statement."""
    statement_for_prompt = truncate_for_prompt(statement, MAX_STATEMENT_PROMPT_TOKENS)
    prompt = (
        "Generate exactly 10 concise keywords that someone would naturally associate with "
        "the statement below. Favor synonyms, hypernyms, physical attributes, contexts, "
        "functions, causes, or consequences tied directly to the subject, especially those "
        "aligned with the database entries. If the statement is a single noun, provide "
        "closely related nouns (e.g., synonyms or category names) that may match entries. "
        "Each keyword must be 1-3 alphabetic words, comma-separated, with no instructions "
        "or filler language.\n"
        f"Statement: \"{statement_for_prompt}\"\n"
        "Keywords:"
    )
    return get_keywords_from_prompt(
        prompt,
        desired_count=10,
        fallback_factory=lambda: fallback_keywords_from_text(statement, 10),
        max_new_tokens=80,
    )


def generate_tertiary_keywords(unmatched_terms: Sequence[str]) -> List[str]:
    """Expand unmatched secondary keywords into tertiary descriptors."""
    if not unmatched_terms:
        return []
    joined_terms = truncate_for_prompt(", ".join(unmatched_terms), MAX_SUGGESTION_PROMPT_TOKENS)
    prompt = (
        "The following keywords returned no database matches: "
        f"{joined_terms}. "
        "Generate up to 15 new descriptors (1-3 alphabetic words each) that clarify their "
        "synonyms, categories, symbolic meanings, contexts, or downstream effects—especially "
        "phrases that could match related entries. Do not repeat the original keywords "
        "and do not include instructions. Return a single comma-separated list."
    )
    tertiary = get_keywords_from_prompt(
        prompt,
        desired_count=15,
        fallback_factory=lambda: fallback_keywords_from_terms(unmatched_terms, 15),
        max_new_tokens=120,
    )
    return exclude_terms(tertiary, unmatched_terms)


def has_negation(text: str) -> bool:
    tokens = tokenize_for_overlap(text)
    return any(token in NEGATION_TOKENS for token in tokens)


def calculate_similarity_features(term: str, candidate: str) -> Tuple[float, int]:
    tokens_term = tokenize_for_overlap(term)
    tokens_candidate = tokenize_for_overlap(candidate)
    overlap = len(tokens_term & tokens_candidate)
    ratio = SequenceMatcher(None, term, candidate).ratio()
    return ratio, overlap


def is_close_match(
    term: str,
    candidate: str,
    entry_tokens: Set[str] | None = None,
    min_token_overlap: int = 1,
) -> bool:
    """Return True if term and candidate are similar enough."""
    if not term or not candidate:
        return False

    term_has_neg = has_negation(term)
    candidate_has_neg = has_negation(candidate)
    if term_has_neg != candidate_has_neg:
        return False

    ratio, overlap = calculate_similarity_features(term, candidate)
    if overlap >= min_token_overlap:
        return True
    term_tokens = tokenize_for_overlap(term)
    if entry_tokens:
        entry_meaningful_tokens = entry_tokens - BANNED_GENERIC_TOKENS
        meaningful_overlap = len(entry_meaningful_tokens & term_tokens)
        if meaningful_overlap >= min_token_overlap:
            return True
    if len(term) < 5 or len(candidate) < 5:
        return False
    return ratio >= SIMILARITY_THRESHOLD


def search_database_with_terms(
    terms: Iterable[str],
    data: Database,
    statement_context: str | None = None,
    min_term_token_overlap_ratio: float = 0.0,
    min_term_tokens: int = 0,
) -> Tuple[MatchDict, Set[str]]:
    """Search the database for any entry containing the provided terms."""
    matches: MatchDict = {}
    matched_terms: Set[str] = set()
    for term in terms:
        normalized_term = normalize_term(term)
        if not normalized_term:
            continue
        term_tokens = tokenize_for_overlap(normalized_term)
        if not term_tokens:
            continue
        if min_term_tokens and len(term_tokens) < min_term_tokens:
            continue
        total_term_token_count = len(term_tokens)
        effective_min_ratio = min_term_token_overlap_ratio
        if effective_min_ratio > 0 and total_term_token_count <= 3:
            effective_min_ratio = 0.0
        candidate_keys = _candidate_keys_for_tokens(term_tokens, data)
        term_word_count = len(normalized_term.split())
        for key in candidate_keys:
            entry = data[key]
            db_string = normalize_term(str(entry.get("string", "")))
            entry_tokens = entry_tokens_for(entry["string"])
            min_overlap = 1
            entry_word_count = len(db_string.split())
            if term_word_count > 1 and entry_word_count > 1:
                min_overlap = 2
                if min(len(term_tokens), len(entry_tokens)) <= 2:
                    min_overlap = 1
            overlap = len(term_tokens & entry_tokens)
            if overlap < min_overlap:
                continue
            if (
                effective_min_ratio > 0
                and total_term_token_count > 0
                and overlap / total_term_token_count < effective_min_ratio
            ):
                continue
            if is_close_match(normalized_term, db_string, entry_tokens, min_overlap):
                if statement_context and not contexts_align(statement_context, entry["string"]):
                    continue
                matches[key] = entry
                matched_terms.add(normalized_term)
    return matches, matched_terms


def search_database_exact(term: str, data: Database) -> MatchDict:
    """Return matches whose string matches the term exactly (after normalization)."""
    normalized_term = normalize_term(term)
    if not normalized_term:
        return {}
    matches: MatchDict = {}
    for key, entry in data.items():
        db_string = normalize_term(str(entry.get("string", "")))
        if db_string == normalized_term:
            matches[key] = entry
    return matches


def search_database_near_exact(term: str, data: Database, similarity_threshold: float = 0.93) -> MatchDict:
    """Return matches that are near-identical to the term (minor pluralization/typos)."""
    normalized_term = normalize_term(term)
    if not normalized_term:
        return {}
    term_tokens = tokenize_for_overlap(normalized_term)
    if not term_tokens:
        return {}
    candidate_keys = _candidate_keys_for_tokens(term_tokens, data)
    matches: MatchDict = {}
    for key in candidate_keys:
        entry = data[key]
        db_string = normalize_term(str(entry.get("string", "")))
        if not db_string:
            continue
        effective_threshold = similarity_threshold
        if len(normalized_term.split()) <= 4:
            effective_threshold = max(0.8, similarity_threshold - 0.1)
        ratio = SequenceMatcher(None, normalized_term, db_string).ratio()
        if ratio < effective_threshold:
            continue
        entry_tokens = entry_tokens_for(entry["string"])
        min_overlap = min(len(term_tokens), len(entry_tokens))
        overlap = len(term_tokens & entry_tokens)
        if min_overlap <= 2 and overlap >= 1:
            matches[key] = entry
            continue
        if overlap >= max(1, min_overlap):
            matches[key] = entry
    return matches


def search_database_by_names(
    names: Iterable[str],
    data: Database,
    statement_context: str | None = None,
) -> MatchDict:
    """Return matches for database names (exact normalized string)."""
    matches: MatchDict = {}
    for name in names:
        normalized = normalize_term(name)
        key = DATABASE_NORMALIZED_TO_KEY.get(normalized)
        if key and key in data:
            entry = data[key]
            if statement_context and not contexts_align(statement_context, entry["string"]):
                continue
            matches[key] = entry
    return matches


def search_adap_map(
    terms: Iterable[str],
    data: Dict[str, Dict[str, object]],
) -> MatchDict:
    """Match terms against the ADAP Map of Consciousness (name only)."""
    matches: MatchDict = {}
    if not terms:
        return matches
    for term in terms:
        normalized_term = normalize_term(term)
        if not normalized_term:
            continue
        for key, entry in data.items():
            entry_string = str(entry.get("string", "")).strip()
            if not entry_string:
                continue
            entry_tokens = tokenize_for_overlap(entry_string)
            if is_close_match(normalized_term, normalize_term(entry_string), entry_tokens, min_token_overlap=1):
                match_entry = dict(entry)
                match_entry["matched_fields"] = ["name"]
                matches[key] = match_entry
    return matches


def _filter_statement_similarity_matches(statement: str, matches: MatchDict) -> MatchDict:
    """Tighten the statement-similarity layer by requiring strong overlap."""
    statement_tokens = tokenize_for_overlap(statement)
    if len(statement_tokens) < 2:
        return {}
    normalized_statement = normalize_term(statement)
    filtered: MatchDict = {}
    for key, entry in matches.items():
        entry_text = str(entry.get("string", ""))
        entry_tokens = entry_tokens_for(entry_text)
        overlap = len(statement_tokens & entry_tokens)
        if overlap < 2:
            continue
        coverage = overlap / max(1, len(statement_tokens))
        normalized_entry = normalize_term(entry_text)
        ratio = SequenceMatcher(None, normalized_statement, normalized_entry).ratio()
        if coverage < 0.5 and ratio < 0.8:
            continue
        filtered[key] = entry
    return filtered


def _heuristic_adap_map_match(statement: str) -> Tuple[str, str]:
    """Fallback: choose the best ADAP Map entry using token overlap scoring."""
    tokens = tokenize_for_overlap(statement)
    hint_loc = _detect_adap_hint_loc(statement)

    best_key = None
    best_score = -1.0
    for key in ADAP_MOC_ORDERED_KEYS:
        entry = ADAP_MAP_OF_CONSCIOUSNESS[key]
        score = 0.0
        string_tokens = tokenize_for_overlap(entry["string"])
        overlap = len(tokens & string_tokens)
        score += overlap
        if hint_loc is not None:
            proximity = max(0.0, 1.0 - abs(entry["calibration"] - hint_loc) / 300.0)
            score += proximity
        if score > best_score:
            best_score = score
            best_key = key
    reason = "heuristic intent similarity" if best_score > 0 else "default fallback (no overlap)"
    return best_key or ADAP_MOC_ORDERED_KEYS[-1], reason


def _rank_adap_candidates(statement: str, extra_terms: Iterable[str]) -> List[Tuple[str, float, List[str]]]:
    """Score ADAP entries against all terms to build a candidate list."""
    combined_text = " ".join([statement, *extra_terms])
    tokens = tokenize_for_overlap(combined_text)
    hint_loc = _detect_adap_hint_loc(combined_text)

    candidates: List[Tuple[str, float, List[str]]] = []
    for key in ADAP_MOC_ORDERED_KEYS:
        entry = ADAP_MAP_OF_CONSCIOUSNESS[key]
        matched_fields: List[str] = []
        score = 0.0
        field_tokens = tokenize_for_overlap(entry["string"])
        overlap = len(tokens & field_tokens)
        if overlap:
            matched_fields.append("name")
        score += overlap
        if hint_loc is not None:
            proximity = max(0.0, 1.0 - abs(entry["calibration"] - hint_loc) / 300.0)
            score += proximity
        candidates.append((key, score, matched_fields))
    candidates.sort(key=lambda t: (t[1], ADAP_MAP_OF_CONSCIOUSNESS[t[0]]["calibration"]), reverse=True)
    return candidates


def _select_best_adap_match(matches: MatchDict) -> Tuple[str | None, str | None, List[str]]:
    """Pick a single ADAP entry from token-overlap matches."""
    best_key: str | None = None
    best_reason: str | None = None
    best_fields: List[str] = []
    best_score = -1.0
    for key, entry in matches.items():
        fields = list(entry.get("matched_fields", []))
        score = float(len(fields))
        if score > best_score or (
            score == best_score
            and ADAP_MAP_OF_CONSCIOUSNESS[key]["calibration"]
            > ADAP_MAP_OF_CONSCIOUSNESS.get(best_key or key, {}).get("calibration", 0)  # type: ignore[arg-type]
        ):
            best_key = key
            best_fields = fields
            best_score = score
    if best_key is not None:
        if best_fields:
            best_reason = f"overlap on {', '.join(best_fields)}"
        else:
            best_reason = "token overlap"
    return best_key, best_reason, best_fields


def evaluate_adap_map_with_gpt(statement: str) -> Tuple[str | None, str | None]:
    """Use GPT to pick the best ADAP Map entry for the statement's intent."""
    statement_for_prompt = truncate_for_prompt(statement, MAX_STATEMENT_PROMPT_TOKENS)
    prompt = (
        "ADAP Map of Consciousness entries:\n"
        f"{ADAP_MOC_PROMPT_ENTRIES}\n\n"
        f"Statement: \"{statement_for_prompt}\"\n"
        "Select the single best-fitting entry whose intention/crown/3rd eye/heart most closely align with the explicit "
        "or implicit intent of the statement. Capitalization is semantically important (e.g., 'Presence' as a divine "
        "proper noun differs from generic 'presence'); only choose a capitalized divine term if the statement clearly "
        "points to the same divine concept. The explanation must describe how the statement embodies or aspires to the "
        "selected qualities—avoid inverse or negation-based rationales (e.g., do not justify with \"lacks\" or \"implies "
        "absence\"). Statements about unfairness, hypocrisy, bias, or double standards generally align with lower levels "
        "of consciousness (below 200); do not map them to higher integrative states merely because fairness is desirable. "
        "Respond ONLY with JSON like "
        '{"loc": 350, "intention": "improvement", "reason": "short explanation"}'
    )
    text = language_model_completion(prompt, max_new_tokens=200)
    if not text:
        return None, None
    json_text = text.strip()
    if not json_text.startswith("{"):
        try:
            start = json_text.index("{")
            end = json_text.rindex("}") + 1
            json_text = json_text[start:end]
        except ValueError:
            pass
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return None, None
    if not isinstance(data, dict):
        return None, None
    loc_val = data.get("loc")
    intention_val = str(data.get("intention", "")) if data.get("intention") else None
    reason = str(data.get("reason", "")).strip() if data.get("reason") else None
    loc_num: int | None = None
    try:
        loc_num = int(loc_val) if loc_val is not None else None
    except (TypeError, ValueError):
        loc_num = None
    selected_key: str | None = None
    if loc_num is not None:
        for key, entry in ADAP_MAP_OF_CONSCIOUSNESS.items():
            if int(entry["calibration"]) == loc_num:
                selected_key = key
                break
    if not selected_key and intention_val:
        norm_intent = normalize_term(intention_val)
        for key, entry in ADAP_MAP_OF_CONSCIOUSNESS.items():
            if normalize_term(str(entry.get("string", ""))) == norm_intent:
                selected_key = key
                break
    return selected_key, reason


def match_adap_map(statement: str, extra_terms: Iterable[str]) -> MatchDict:
    """Return the single most likely ADAP map match, preferring GPT then overlap then heuristic."""
    overlap_matches = search_adap_map([statement, *extra_terms], ADAP_MAP_OF_CONSCIOUSNESS)
    overlap_key, overlap_reason, overlap_fields = _select_best_adap_match(overlap_matches)

    gpt_key, gpt_reason = evaluate_adap_map_with_gpt(statement)
    selected_key = gpt_key or overlap_key
    selected_reason = None
    if gpt_key:
        selected_reason = gpt_reason or "model suggested intent"
    else:
        selected_reason = overlap_reason
    selected_fields: List[str] = []

    if selected_key is None:
        selected_key, selected_reason = _heuristic_adap_map_match(statement)
        selected_fields = []
    else:
        selected_fields = overlap_fields if selected_key == overlap_key else []
        if selected_reason is None:
            selected_reason = "intent match"

    # If we detected a desire/fear/apathy hint, force alignment to the closest LoC when
    # the chosen entry had no lexical overlap (pure intent guess).
    hint_loc = _detect_adap_hint_loc(statement)
    if hint_loc is not None and not selected_fields:
        current_loc = int(ADAP_MAP_OF_CONSCIOUSNESS[selected_key]["calibration"])
        if abs(current_loc - hint_loc) > 50:
            closest_key = _closest_adap_key_to_loc(hint_loc)
            selected_key = closest_key
            selected_reason = f"aligned to hinted intent near LoC {hint_loc}"
            selected_fields = []

    entry = dict(ADAP_MAP_OF_CONSCIOUSNESS[selected_key])
    entry["matched_fields"] = selected_fields
    entry["reason"] = selected_reason
    entry["is_adap_map"] = True
    return {selected_key: entry}


def average_calibration(matches: MatchDict) -> float | None:
    """Return the average calibration for the provided matches."""
    if len(matches) < 3:
        return None
    calibrations = [float(entry["calibration"]) for entry in matches.values()]
    weights = [entry_weight(entry) for entry in matches.values()]
    return geometric_mean_values(calibrations, weights)


def geometric_mean_values(values: Iterable[float], weights: Iterable[float] | None = None) -> float | None:
    values = list(values)
    if not values:
        return None
    weights_list = None if weights is None else list(weights)
    if weights_list is not None and len(weights_list) != len(values):
        weights_list = None
    log_sum = 0.0
    weight_sum = 0.0
    for idx, value in enumerate(values):
        w = weights_list[idx] if weights_list is not None else 1.0
        if value <= 0:
            return float(mean(values))
        log_sum += math.log(value) * w
        weight_sum += w
    if weight_sum == 0:
        return None
    return math.exp(log_sum / weight_sum)


def entry_weight(entry: Dict[str, object]) -> float:
    """Compute a simple weight to favor stronger matches in averaging."""
    weight = 1.0
    matched_fields = entry.get("matched_fields") or []
    weight += 0.3 * len(matched_fields)
    entry_type = entry.get("type")
    if entry_type == "ADAP":
        weight += 0.2
    return weight


def geometric_mean_entries(matches: MatchDict) -> float | None:
    calibrations = []
    weights = []
    for entry in matches.values():
        calibrations.append(float(entry["calibration"]))
        weights.append(entry_weight(entry))
    return geometric_mean_values(calibrations, weights)


def calibration_range(matches: MatchDict) -> Tuple[float, float] | None:
    """Return the min/max calibration values across matches."""
    if not matches:
        return None
    calibrations = [float(entry["calibration"]) for entry in matches.values()]
    if not calibrations:
        return None
    return min(calibrations), max(calibrations)


def run_pipeline(statement: str, data: Database) -> Tuple[MatchDict, List[Tuple[str, MatchDict]], List[str], List[str], List[str]]:
    """Execute the lookup pipeline in ordered layers, returning on first hit."""
    stage_records: List[Tuple[str, MatchDict]] = []
    consolidated_matches: MatchDict = {}
    secondary_keywords: List[str] = []
    tertiary_keywords: List[str] = []
    gpt_suggestions_with_reason: List[Tuple[str, str]] = []

    # 1. Direct statement
    direct_matches = search_database_exact(statement, data)
    stage_records.append(("Direct statement", direct_matches))
    if direct_matches:
        consolidated_matches.update(direct_matches)
        return consolidated_matches, stage_records, secondary_keywords, tertiary_keywords, gpt_suggestions_with_reason

    # 2. Near exact statement
    near_matches = search_database_near_exact(statement, data)
    stage_records.append(("Near exact statement", near_matches))
    if near_matches:
        consolidated_matches.update(near_matches)
        return consolidated_matches, stage_records, secondary_keywords, tertiary_keywords, gpt_suggestions_with_reason

    # 3. Statement similarity
    statement_similarity_matches, _ = search_database_with_terms(
        [statement],
        data,
        statement_context=statement,
        min_term_token_overlap_ratio=0.6,
        min_term_tokens=2,
    )
    statement_similarity_matches = _filter_statement_similarity_matches(statement, statement_similarity_matches)
    stage_records.append(("Statement similarity", statement_similarity_matches))
    if statement_similarity_matches:
        consolidated_matches.update(statement_similarity_matches)
        return consolidated_matches, stage_records, secondary_keywords, tertiary_keywords, gpt_suggestions_with_reason

    # 4. Database suggestions
    gpt_suggestions_with_reason = generate_database_suggestions(statement)
    gpt_suggestions = [name for name, _ in gpt_suggestions_with_reason]
    suggestion_matches = search_database_by_names(gpt_suggestions, data, statement) if gpt_suggestions else {}
    stage_matches = dict(suggestion_matches)
    stage_records.append(("Database suggestions", stage_matches))
    if stage_matches:
        consolidated_matches.update(stage_matches)
        return consolidated_matches, stage_records, secondary_keywords, tertiary_keywords, gpt_suggestions_with_reason

    # 4b. ADAP Map fallback (only if no database suggestions matched)
    adap_map_matches = match_adap_map(statement, [statement])
    stage_records.append(("ADAP map fallback", adap_map_matches))
    if adap_map_matches:
        consolidated_matches.update(adap_map_matches)
        return consolidated_matches, stage_records, secondary_keywords, tertiary_keywords, gpt_suggestions_with_reason

    # 5. Secondary keywords + Map of Consciousness (fallback)
    secondary_keywords = generate_secondary_keywords(statement)
    secondary_matches, matched_secondary_terms = search_database_with_terms(secondary_keywords, data, statement)
    stage_matches = dict(secondary_matches)
    if not stage_matches:
        stage_matches = match_adap_map(statement, [statement, *secondary_keywords])
    stage_records.append(("Secondary keywords + ADAP map", stage_matches))
    if stage_matches:
        consolidated_matches.update(stage_matches)
        return consolidated_matches, stage_records, secondary_keywords, tertiary_keywords, gpt_suggestions_with_reason

    # 6. Tertiary keywords + Map of Consciousness (fallback)
    unmatched_secondary = [term for term in secondary_keywords if normalize_term(term) not in matched_secondary_terms]
    tertiary_keywords = generate_tertiary_keywords(unmatched_secondary)
    tertiary_matches, _ = search_database_with_terms(tertiary_keywords, data, statement)
    stage_matches = dict(tertiary_matches)
    if not stage_matches:
        stage_matches = match_adap_map(statement, [statement, *secondary_keywords, *tertiary_keywords])
    stage_records.append(("Tertiary keywords + ADAP map", stage_matches))
    consolidated_matches.update(stage_matches)

    return consolidated_matches, stage_records, secondary_keywords, tertiary_keywords, gpt_suggestions_with_reason


def format_entry(entry: Dict[str, object]) -> str:
    """Format a database entry for human-readable output."""
    entry_type = entry.get("type", "ADAP")
    if entry.get("is_adap_map"):
        loc = entry.get("calibration")
        matched_fields = entry.get("matched_fields")
        matched_text = ""
        if matched_fields:
            matched_text = f" (matched on {', '.join(matched_fields)})"
        reason = entry.get("reason")
        reason_text = f" — {reason}" if reason else ""
        return f"ADAP Map of Consciousness — {entry['string']} (LoC {loc}){matched_text}{reason_text}"
    return f"{entry['string']} (calibration={entry['calibration']}, type={entry_type})"


def main() -> None:
    """CLI entry point."""
    statement = input("Please enter the statement you want to analyze: ").strip()
    if not statement:
        print("No statement provided. Exiting.")
        return

    matches, stages, secondary_keywords, tertiary_keywords, gpt_suggestions_with_reason = run_pipeline(statement, database)

    direct_stage = next(
        (stage_matches for stage_name, stage_matches in stages if stage_name == "Direct statement" and stage_matches),
        None,
    )

    print("\n=== Lookup Results ===")
    if direct_stage:
        print("\nDirect statement match(es) found! Immediate calibrations:")
        for entry in direct_stage.values():
            print(f"  => {format_entry(entry)}")

    for stage_name, stage_matches in stages:
        print(f"\n{stage_name}:")
        if stage_name == "Secondary keywords" and secondary_keywords:
            print("  Keywords:", ", ".join(secondary_keywords))
        if stage_name == "Tertiary keywords" and tertiary_keywords:
            print("  Keywords:", ", ".join(tertiary_keywords))
        if stage_name == "Database suggestions" and gpt_suggestions_with_reason:
            formatted = ", ".join(
                f"{name} ({reason or 'no reasoning provided'})"
                for name, reason in gpt_suggestions_with_reason
            )
            print("  Suggestions:", formatted)
        if stage_matches:
            for entry in stage_matches.values():
                print(f"  - {format_entry(entry)}")
        else:
            print("  No matches.")

    avg = average_calibration(matches) or geometric_mean_entries(matches)
    cal_range = calibration_range(matches)
    if avg is None:
        print("\nNo calibration guesstimate could be made with the current database. This could be wrong.")
    else:
        qualifier = ""
        if len(matches) < 3:
            qualifier = " (only a few matches available)"
        range_text = ""
        if cal_range:
            lo, hi = cal_range
            range_text = f" between ({lo:.0f} - {hi:.0f})"
        print(
            f"\nGeometric-average calibration across {len(matches)} match(es): {avg:.2f}{range_text}{qualifier}. "
            "Remember the scale is logarithmic, so geometric means are used. This could be wrong."
        )


def contexts_align(statement: str, entry_text: str) -> bool:
    """Check whether the statement and entry share compatible context/polarity."""
    key = (normalize_term(statement), entry_text)
    cached = CONTEXT_ALIGNMENT_CACHE.get(key)
    if cached is not None:
        return cached

    statement_for_prompt = truncate_for_prompt(statement, MAX_CONTEXT_STATEMENT_TOKENS)
    entry_for_prompt = truncate_for_prompt(entry_text, MAX_CONTEXT_ENTRY_TOKENS)
    prompt = (
        "Determine whether the following statement and entry describe related contexts "
        "with similar intent, polarity, and level (e.g., pro/anti, positive/negative, high/low, "
        "affirmative/denial). Consider negations such as 'not', 'ban', 'resist', etc. Respond ONLY "
        "with a JSON object {\"related\": true/false, \"reason\": \"...\"}. Reply 'true' only if the "
        "statement and entry clearly align; otherwise reply 'false'.\n"
        f"Statement: {statement_for_prompt}\n"
        f"Entry: {entry_for_prompt}\n"
        "JSON:"
    )
    text = None
    if OPENAI_API_KEY:
        text = language_model_completion(prompt, max_new_tokens=200)
    related: bool | None = None
    if text:
        text = text.strip()
        json_text = text
        if not text.startswith("{"):
            try:
                start = text.index("{")
                end = text.rindex("}") + 1
                json_text = text[start:end]
            except ValueError:
                json_text = text
        try:
            data = json.loads(json_text)
            if isinstance(data, dict):
                val = data.get("related")
                if isinstance(val, bool):
                    related = val
                elif isinstance(val, str):
                    related = val.strip().lower() == "true"
        except json.JSONDecodeError:
            related = None

    heuristic_related = _heuristic_context_related(statement, entry_text)
    if related is None:
        related = heuristic_related
    elif related and not heuristic_related:
        related = False

    CONTEXT_ALIGNMENT_CACHE[key] = related
    return related
def heuristic_database_suggestions(statement: str, data: Database, top_n: int = 3, threshold: float = 0.45) -> List[str]:
    """Provide fallback suggestions using token overlap and string similarity."""
    normalized_statement = normalize_term(statement)
    statement_tokens = {
        token for token in normalized_statement.split() if len(token) > 2 and token not in STOPWORDS
    }
    scores: List[Tuple[float, str]] = []
    for entry in data.values():
        db_name = entry["string"]
        normalized_db = normalize_term(db_name)
        db_tokens = {
            token for token in normalized_db.split() if len(token) > 2 and token not in STOPWORDS
        }
        overlap = len(statement_tokens & db_tokens)
        char_score = SequenceMatcher(None, normalized_statement, normalized_db).ratio()
        combined = overlap + char_score * 0.5
        if overlap == 0 and char_score < threshold:
            continue
        scores.append((combined, db_name))
    scores.sort(reverse=True)
    return [name for _, name in scores[:top_n]]


def generate_database_suggestions(statement: str) -> List[Tuple[str, str]]:
    """Ask an LLM to suggest database entries similar to the statement, with reasoning."""
    statement_for_prompt = truncate_for_prompt(statement, MAX_STATEMENT_PROMPT_TOKENS)
    prompt = (
        "You are given the following database entries:\n"
        f"{DATABASE_REFERENCES}\n\n"
        f"Statement: \"{statement_for_prompt}\"\n"
        "Output a JSON array of up to 5 objects. Each object must have keys 'entry', 'reason', "
        "and 'confidence'. The 'entry' must exactly match one of the database entries. The "
        "'reason' should briefly explain the literal overlap between the statement and the entry. "
        "'confidence' must be a number between 0 and 1. Only include entries with a clear, "
        "literal relationship to the statement. If none qualify, return an empty JSON array []."
    )
    text = language_model_completion(prompt, max_new_tokens=400)
    suggestions: List[Tuple[str, str]] = []
    parsed = False
    statement_tokens = {
        token for token in re.findall(r"[A-Za-z]+", statement.lower()) if token not in STOPWORDS
    }
    if text:
        text = text.strip()
        json_text = text
        if not text.startswith("["):
            try:
                start = text.index("[")
                end = text.rindex("]") + 1
                json_text = text[start:end]
            except ValueError:
                json_text = text
        try:
            data = json.loads(json_text)
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("entry", "")).strip()
                    reason = str(item.get("reason", "")).strip()
                    try:
                        confidence_val = float(item.get("confidence", 0))
                    except (TypeError, ValueError):
                        confidence_val = 0.0
                    reason_tokens = {
                        token for token in re.findall(r"[A-Za-z]+", reason.lower()) if token not in STOPWORDS
                    }
                    if (
                        not name
                        or normalize_term(name) not in DATABASE_NORMALIZED_TO_KEY
                        or confidence_val < SUGGESTION_CONFIDENCE_THRESHOLD
                        or not statement_tokens.intersection(reason_tokens)
                        or not statement_tokens.intersection(entry_tokens_for(name))
                    ):
                        continue
                    suggestions.append((name, reason or "model suggested similarity"))
            parsed = True
        except json.JSONDecodeError:
            parsed = False

    if not suggestions and not parsed and text:
        # Fallback to line parsing if JSON was not honored.
        lines = [line.strip(" -*•\t") for line in text.splitlines() if line.strip()]
        if not lines and "," in text:
            lines = [part.strip() for part in text.split(",") if part.strip()]
        for line in lines:
            if not line or line.lower() == "none":
                continue
            if "—" in line:
                name_part, justification = line.split("—", 1)
            elif "-" in line:
                name_part, justification = line.split("-", 1)
            else:
                name_part, justification = line, ""
            name = name_part.strip(" \"'")
            justification = justification.strip(" \"'")
            reason_tokens = {
                token for token in re.findall(r"[A-Za-z]+", justification.lower()) if token not in STOPWORDS
            }
            if (
                not name
                or normalize_term(name) not in DATABASE_NORMALIZED_TO_KEY
                or not statement_tokens.intersection(reason_tokens)
                or not statement_tokens.intersection(entry_tokens_for(name))
            ):
                continue
            suggestions.append((name, justification or "model suggested similarity"))

    fallback = heuristic_database_suggestions(statement, database)
    seen = {normalize_term(name) for name, _ in suggestions}
    for name in fallback:
        normalized = normalize_term(name)
        if normalized not in seen and statement_tokens.intersection(entry_tokens_for(name)):
            suggestions.append((name, "heuristic similarity"))
            seen.add(normalized)
    return suggestions


if __name__ == "__main__":
    main()
