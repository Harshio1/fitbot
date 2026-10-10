"""
Train the fitness intent-classification neural network.

Pipeline:  text -> tokenize -> bag of words (unigrams + bigrams)
           -> Dense(64, ReLU) -> Dropout -> Dense(32, ReLU) -> Dropout
           -> Dense(num_intents, Softmax)

Only NumPy is needed:   pip install numpy
Run from this folder:   python train.py
Output:                 ../model/model.json  (loaded by the browser, no server needed)

IMPORTANT: tokenize() here must stay identical to tokenize() in app.js.
"""
import json
import os
import re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
H1, H2 = 64, 32
EPOCHS = 400
LR = 0.01
DROPOUT = 0.3

STOPWORDS = {"a", "an", "the", "is", "are", "am", "i", "me", "my", "to", "of",
             "do", "does", "you", "it", "be", "so", "that", "this", "please", "s"}


def tokenize(text):
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    out = []
    for w in text.split():
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]                      # very light plural stemming
        if w not in STOPWORDS:
            out.append(w)
    return out


def features(text):
    toks = tokenize(text)
    return toks + [a + "_" + b for a, b in zip(toks, toks[1:])]


# ---------------------------------------------------------------- data
with open(os.path.join(HERE, "intents.json"), encoding="utf-8") as f:
    intents = json.load(f)["intents"]

labels = [i["tag"] for i in intents]
responses = {i["tag"]: i["response"] for i in intents}

docs, y = [], []
for idx, it in enumerate(intents):
    for p in it["patterns"]:
        docs.append(features(p))
        y.append(idx)

vocab = sorted({w for d in docs for w in d})
w2i = {w: i for i, w in enumerate(vocab)}


def vectorize(feats):
    v = np.zeros(len(vocab), dtype=np.float32)
    for w in feats:
        if w in w2i:
            v[w2i[w]] = 1.0
    return v


X = np.stack([vectorize(d) for d in docs])
Y = np.eye(len(labels), dtype=np.float32)[y]
print(f"{len(docs)} training patterns, {len(labels)} intents, {len(vocab)} features")

# ---------------------------------------------------------------- model
rng = np.random.default_rng(SEED)


def he(i, o):
    return (rng.standard_normal((i, o)) * np.sqrt(2.0 / i)).astype(np.float32)


P = {"W1": he(len(vocab), H1), "b1": np.zeros(H1, np.float32),
     "W2": he(H1, H2), "b2": np.zeros(H2, np.float32),
     "W3": he(H2, len(labels)), "b3": np.zeros(len(labels), np.float32)}


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(x, train=False):
    z1 = x @ P["W1"] + P["b1"]
    a1 = np.maximum(z1, 0)
    m1 = None
    if train:
        m1 = (rng.random(a1.shape) > DROPOUT) / (1 - DROPOUT)
        a1 = a1 * m1
    z2 = a1 @ P["W2"] + P["b2"]
    a2 = np.maximum(z2, 0)
    m2 = None
    if train:
        m2 = (rng.random(a2.shape) > DROPOUT) / (1 - DROPOUT)
        a2 = a2 * m2
    out = softmax(a2 @ P["W3"] + P["b3"])
    return out, (x, z1, a1, m1, z2, a2, m2)


def predict(x):
    return forward(x, train=False)[0]


# Adam optimiser
m = {k: np.zeros_like(v) for k, v in P.items()}
v_ = {k: np.zeros_like(v) for k, v in P.items()}
b1_, b2_, eps = 0.9, 0.999, 1e-8
BATCH = 32
n = len(X)
step = 0
for epoch in range(1, EPOCHS + 1):
    order = rng.permutation(n)
    for s in range(0, n, BATCH):
        idx = order[s:s + BATCH]
        xb, yb = X[idx], Y[idx]
        out, (x, z1, a1, m1, z2, a2, m2) = forward(xb, train=True)
        d3 = (out - yb) / len(idx)
        g = {"W3": a2.T @ d3, "b3": d3.sum(0)}
        da2 = (d3 @ P["W3"].T) * m2 * (z2 > 0)
        g["W2"], g["b2"] = a1.T @ da2, da2.sum(0)
        da1 = (da2 @ P["W2"].T) * m1 * (z1 > 0)
        g["W1"], g["b1"] = x.T @ da1, da1.sum(0)
        step += 1
        for k in P:
            m[k] = b1_ * m[k] + (1 - b1_) * g[k]
            v_[k] = b2_ * v_[k] + (1 - b2_) * g[k] ** 2
            mh = m[k] / (1 - b1_ ** step)
            vh = v_[k] / (1 - b2_ ** step)
            P[k] -= LR * mh / (np.sqrt(vh) + eps)
    if epoch % 100 == 0 or epoch == 1:
        p = predict(X)
        loss = -np.log(p[np.arange(n), y] + 1e-9).mean()
        acc = (p.argmax(1) == np.array(y)).mean()
        print(f"epoch {epoch:4d}  loss {loss:.4f}  train acc {acc:.3f}")

# ---------------------------------------------------------------- evaluation
with open(os.path.join(HERE, "test_queries.json"), encoding="utf-8") as f:
    tests = json.load(f)

THRESHOLD = 0.60
MARGIN = 0.20
MIN_KNOWN = 0.50
correct = answered_wrong = rejected = 0
confs_right, confs_wrong = [], []
for text, tag in tests:
    feats = features(text)
    known = [w for w in feats if w in w2i]
    p = predict(vectorize(feats)[None, :])[0]
    pred, conf = labels[int(p.argmax())], float(p.max())
    top2 = np.sort(p)[-2]
    unsure = (not known) or conf < THRESHOLD or (conf - top2) < MARGIN \
        or (len(known) / max(1, len(feats))) < MIN_KNOWN
    if unsure:
        pred = "fallback"
    if pred == tag:
        correct += 1
        confs_right.append(conf)
    else:
        if pred == "fallback":
            rejected += 1
        else:
            answered_wrong += 1
        confs_wrong.append(conf)
        print(f"  MISS: '{text}' -> {pred} ({conf:.2f}) expected {tag}")

print(f"\nHeld-out test accuracy: {correct}/{len(tests)} = {correct / len(tests):.1%}")
print(f"Wrong answers given: {answered_wrong}   Safe fallbacks (asked to rephrase): {rejected}")

# ---------------------------------------------------------------- export
model = {
    "threshold": THRESHOLD, "margin": MARGIN, "minKnown": MIN_KNOWN,
    "labels": labels,
    "responses": responses,
    "stopwords": sorted(STOPWORDS),
    "vocab": vocab,
    "W1": np.round(P["W1"], 4).tolist(), "b1": np.round(P["b1"], 4).tolist(),
    "W2": np.round(P["W2"], 4).tolist(), "b2": np.round(P["b2"], 4).tolist(),
    "W3": np.round(P["W3"], 4).tolist(), "b3": np.round(P["b3"], 4).tolist(),
}
out_path = os.path.join(HERE, "..", "model", "model.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(model, f, separators=(",", ":"))
print(f"Saved {os.path.normpath(out_path)}  ({os.path.getsize(out_path) / 1024:.0f} KB)")
