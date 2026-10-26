"""Measure the trivial-baseline floor for any dataset.

If a classifier that sees ONLY length (and simple surface stats) matches or
beats a real model, the dataset carries no learnable signal beyond size and
the honest move is to fix the data, not tune the model.

Prints a floor report; exits non-zero when the floor is suspiciously high so
it can gate a pipeline.
"""
import json
import sys

import numpy as np


def load(path):
    return [json.loads(l) for l in open(path)]


def surface(text):
    t = text or ""
    return np.array([
        len(t),                                  # length (the confound)
        len(t.split()),
        t.count("\n"),
        t.count("traceback"),
    ], dtype=float)


def fit_logreg(X, y, lr=0.1, steps=4000):
    """Tiny logistic regression - no sklearn dependency on the cluster."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    Xs = np.hstack([Xs, np.ones((len(Xs), 1))])
    w = np.zeros(Xs.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-Xs @ w))
        w -= lr * (Xs.T @ (p - y)) / len(y)
    return lambda Xn: 1 / (1 + np.exp(-(((Xn - mu) / sd) @ w[:-1] + w[-1])))


def report(name, probs, y):
    pred = (probs >= 0.5).astype(int)
    acc = (pred == y).mean()
    base = max(y.mean(), 1 - y.mean())
    print(f"  {name:28s} acc {acc*100:5.1f}%   (majority-class {base*100:5.1f}%)"
          f"   lift {100*(acc-base):+5.1f} pts")
    return acc, base


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/noul_train.jsonl"
    ev = sys.argv[2] if len(sys.argv) > 2 else "data/noul_test.jsonl"
    tr, te = load(path), load(ev)
    Xtr = np.vstack([surface(r["prompt"]) for r in tr])
    ytr = np.array([r["label"] for r in tr])
    Xte = np.vstack([surface(r["prompt"]) for r in te])
    yte = np.array([r["label"] for r in te])

    print(f"train={len(tr)} test={len(te)}  test unsafe rate {yte.mean()*100:.1f}%")
    print("\n=== TRIVIAL-BASELINE FLOOR (train -> test) ===")
    m = fit_logreg(Xtr, ytr)
    acc, base = report("surface stats (len etc)", m(Xte), yte)

    # length alone
    ml = fit_logreg(Xtr[:, :1], ytr)
    acc_l, _ = report("length ONLY", ml(Xte[:, :1]), yte)

    print()
    if max(acc, acc_l) > base + 0.08:
        print(f"FAIL: the trivial floor is {max(acc, acc_l)*100:.1f}%, more than 8 points "
              f"above the majority class. The dataset is separable by surface form; "
              f"a model trained on it would 'learn' the artefact.")
        return 1
    print(f"OK: floor is {max(acc, acc_l)*100:.1f}% vs majority {base*100:.1f}% "
          f"(<=8pt lift). Any real model must beat this to count.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
