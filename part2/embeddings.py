# everything to turn tweets into embedding lookups: loading the pretrained embeddings
# and tokenising tweets the same way each embedding model was trained

import gzip
import html
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from nltk.tokenize import TweetTokenizer

MODELS = [
    "glove-wiki-gigaword-50",
    "glove-twitter-50",
    "fasttext-wiki-news-subwords-300",
    "conceptnet-numberbatch-17-06-300",
]
CACHE_DIR = Path.home() / ".cache" / "nlp1-embeddings"
EMB_DIR = Path(__file__).parent / "embeddings"   # zips for the tutorial loader (large, gitignored)
URL = "https://github.com/RaRe-Technologies/gensim-data/releases/download/{0}/{0}.gz"


# ---- embeddings ----
# gensim doesn't build on python 3.14, so we read the gensim-data files ourselves
# (plain word2vec text format: "N D" header, then "word v1 ... vD")

class Vectors:
    def __init__(self, words, matrix):
        self.index_to_key = list(words)
        self.key_to_index = {w: i for i, w in enumerate(words)}
        self.vectors = matrix
        self.vector_size = matrix.shape[1]

    def __contains__(self, w):
        return w in self.key_to_index

    def __getitem__(self, w):
        return self.vectors[self.key_to_index[w]]

    def __len__(self):
        return len(self.index_to_key)


def download(name):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    gensim_path = Path.home() / "gensim-data" / name / f"{name}.gz"
    if gensim_path.exists():
        return gensim_path
    path = CACHE_DIR / f"{name}.gz"
    if not path.exists():
        print("downloading", name)
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(URL.format(name)) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        tmp.rename(path)
    return path


def read_entries(name):
    # numberbatch is multilingual (/c/en/word, /c/de/...), keep english only
    nb = "numberbatch" in name
    with gzip.open(download(name), "rt", encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            word, _, rest = line.rstrip().partition(" ")  # fasttext lines end with a space
            rest = rest.strip()
            if nb:
                if not word.startswith("/c/en/"):
                    continue
                word = word[6:]
            if word and rest:
                yield word, rest


_loaded = {}

def load_kv(name):
    if name in _loaded:
        return _loaded[name]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    npy = CACHE_DIR / f"{name}.npy"
    vocab_file = CACHE_DIR / f"{name}.vocab"
    if npy.exists() and vocab_file.exists():
        words = vocab_file.read_text(encoding="utf-8").split("\n")
        matrix = np.load(npy)
    else:
        print("parsing", name)
        words, rows = [], []
        for w, rest in read_entries(name):
            words.append(w)
            rows.append(np.array(rest.split(), dtype=np.float32))
        matrix = np.vstack(rows)
        np.save(npy, matrix)
        vocab_file.write_text("\n".join(words), encoding="utf-8")
    _loaded[name] = Vectors(words, matrix)
    return _loaded[name]


def export_for_tutorial(name):
    # PretrainedEmbeddings from the tutorial wants a zipped text file with a header line
    EMB_DIR.mkdir(exist_ok=True)
    zip_path = EMB_DIR / f"{name}.zip"
    if not zip_path.exists():
        tmp = EMB_DIR / f"{name}.tmp"
        n, dims = 0, None
        with open(tmp, "w", encoding="utf-8") as out:
            for w, rest in read_entries(name):
                if any(c.isspace() for c in w):
                    continue
                out.write(f"{w} {rest}\n")
                n += 1
                dims = dims or len(rest.split())
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            with z.open(f"{name}.txt", "w") as f, open(tmp, "rb") as b:
                f.write(f"{n} {dims}\n".encode())
                shutil.copyfileobj(b, f, length=1 << 20)
        tmp.unlink()
    with zipfile.ZipFile(zip_path) as z, z.open(z.namelist()[0]) as f:
        dims = int(f.readline().split()[1])
    return str(zip_path), dims, True


def load_for_tutorial(name):
    import part2
    path, dims, skip_first = export_for_tutorial(name)
    return part2.clslib.PretrainedEmbeddings(path, dims, skip_first_line=skip_first)


# ---- tokenizer ----
# each embedding model was trained on differently tokenised text, so each model gets
# its own settings (checked against the vocabularies):
#   contractions: "split_nt"   don't -> do n't, i'm -> i 'm   (glove)
#                 "split_apos" don't -> don 't, i'm -> i 'm   (fasttext)
#                 "keep"       don't stays don't              (numberbatch)
#   numbers:      "tag"  2014 -> <number>   (glove twitter)
#                 "hash" 2014 -> ####       (numberbatch, only 2+ digits)
#                 "keep" 2014 stays 2014
#   twitter_tags: glove twitter's special tokens: @user -> <user>, links -> <url>,
#                 #tag -> <hashtag> tag, !!! -> ! <repeat>, sooo -> so <elong>,
#                 LOVE -> love <allcaps>, :) -> <smile>, <3 -> <heart>, ...
#                 (our approximation of stanford's preprocessing script)
#                 without it: mentions and links are dropped, #tag -> tag, sooooo -> sooo

TOKENIZATION = {
    "glove-twitter-50":                 dict(contractions="split_nt", numbers="tag", twitter_tags=True),
    "glove-twitter-200":                dict(contractions="split_nt", numbers="tag", twitter_tags=True),
    "glove-wiki-gigaword-50":           dict(contractions="split_nt", numbers="keep", twitter_tags=False),
    "word2vec-google-news-300":         dict(contractions="split_nt", numbers="keep", twitter_tags=False),
    "fasttext-wiki-news-subwords-300":  dict(contractions="split_apos", numbers="keep", twitter_tags=False),
    "conceptnet-numberbatch-17-06-300": dict(contractions="keep", numbers="hash", twitter_tags=False),
}

NT = re.compile(r"^(\w+)(n't)$", re.I)
CLITIC = re.compile(r"^(\w+)('(?:m|s|re|ve|ll|d))$", re.I)
APOS = re.compile(r"^(\w+)('(?:t|m|s|re|ve|ll|d))$", re.I)
NUMBER = re.compile(r"^[-+]?[.\d]*\d+[:,.\d]*(?:st|nd|rd|th|s)?$", re.I)
SMILE = re.compile(r"^(?:[8:=;]['`\-]?[)dD]+|[(]+['`\-]?[8:=;])$")
LOLFACE = re.compile(r"^[8:=;]['`\-]?[pP]+$")
SADFACE = re.compile(r"^(?:[8:=;]['`\-]?\(+|\)+['`\-]?[8:=;])$")
NEUTRALFACE = re.compile(r"^[8:=;]['`\-]?[/|lL*]$")
ELONG = re.compile(r"(\w)\1{2,}")


def clean_text(text):
    # tweet_eval has literal escapes like \u2019 and html entities like &amp;
    text = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), text)
    text = html.unescape(text)
    return text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')


def is_special(t):
    return t.startswith("<") and t.endswith(">") and len(t) > 2


class TweetTok:
    # same interface as the tutorial's BasicTokenizer (tokens, ids, vocab)
    # keep_punct=False for embed-and-pool, True for the rnn

    def __init__(self, vocab=None, keep_punct=True, contractions="split_nt", numbers="keep", twitter_tags=False):
        assert contractions in ("split_nt", "split_apos", "keep")
        assert numbers in ("tag", "hash", "keep")
        self.vocab = vocab
        self.keep_punct = keep_punct
        self.contractions = contractions
        self.numbers = numbers
        self.twitter_tags = twitter_tags
        # with twitter tags elongations become <elong>, otherwise nltk shortens sooooo -> sooo
        self.tt = TweetTokenizer(preserve_case=True, reduce_len=not twitter_tags)

    def split_contraction(self, t):
        m = None
        if self.contractions == "split_nt":
            m = NT.match(t) or CLITIC.match(t)
        elif self.contractions == "split_apos":
            m = APOS.match(t)
        return [m.group(1), m.group(2)] if m else [t]

    def number(self, t):
        if self.numbers == "tag" and NUMBER.match(t):
            return "<number>"
        if self.numbers == "hash" and sum(c.isdigit() for c in t) >= 2:
            return re.sub(r"\d", "#", t)
        return t

    def with_twitter_tags(self, t):
        if t.startswith("@") and len(t) > 1:
            return ["<user>"]
        if t.lower().startswith(("http", "www.")):
            return ["<url>"]
        if t == "<3":
            return ["<heart>"]
        for rx, tag in [(SMILE, "<smile>"), (LOLFACE, "<lolface>"), (SADFACE, "<sadface>"), (NEUTRALFACE, "<neutralface>")]:
            if rx.match(t):
                return [tag]
        if self.number(t) != t:
            return [self.number(t)]
        if t.startswith("#") and len(t) > 1:
            body = t[1:]
            parts = [body] if body.isupper() else re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", body)
            return ["<hashtag>"] + [self.number(p.lower()) for p in (parts or [body])]
        tail = []
        if len(t) > 1 and t.isalpha() and t.isupper():
            tail.append("<allcaps>")
        if ELONG.search(t):
            t = ELONG.sub(r"\1", t)
            tail.insert(0, "<elong>")
        return self.split_contraction(t.lower()) + tail

    def without_twitter_tags(self, t):
        # mentions and links have no vectors in these models
        if t.startswith("@") and len(t) > 1:
            return []
        if t.lower().startswith(("http", "www.")):
            return []
        t = self.number(t.lower())
        if t.startswith("#") and len(t) > 1 and t[1].isalpha():
            t = t[1:]
        return self.split_contraction(t)

    def mark_repeats(self, toks):
        # !!! -> ! <repeat>, ... -> . <repeat>
        out = []
        for t in toks:
            if t in ("!", "?", "."):
                if out and out[-1] == t:
                    out.append("<repeat>")
                    continue
                if len(out) >= 2 and out[-1] == "<repeat>" and out[-2] == t:
                    continue
            if len(t) > 1 and set(t) <= {".", "!", "?"}:
                out += [t[0], "<repeat>"]
                continue
            out.append(t)
        return out

    def tokens(self, text):
        toks = []
        for t in self.tt.tokenize(clean_text(text)):
            toks += self.with_twitter_tags(t) if self.twitter_tags else self.without_twitter_tags(t)
        if self.twitter_tags:
            toks = self.mark_repeats(toks)
        if not self.keep_punct:
            toks = [t for t in toks if any(c.isalnum() for c in t) or is_special(t)]
        return toks

    def ids(self, text):
        return [self.vocab.index(t) for t in self.tokens(text)]


def make_tokenizer(name, vocab=None, keep_punct=True):
    settings = TOKENIZATION.get(name, dict(contractions="split_nt", numbers="keep", twitter_tags=False))
    return TweetTok(vocab, keep_punct=keep_punct, **settings)