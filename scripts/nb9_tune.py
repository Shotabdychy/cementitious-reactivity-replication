"""PART 1.4 -- hyperparameter tuning of the notebook-9 network with K-fold cross-validation
on the TRAINING rows only.

The paper cites Keras Tuner (ref. 95) but the released notebook has fixed hyperparameters and no
tuner call, so this search is mine. It is run after the per-gram fix of the mix columns.

Protocol
  1. One outer train/test split, fixed by SPLIT_SEED. The test rows are loaded but never read.
  2. Inside the training rows: StratifiedKFold (k folds) on the sound stratification key.
  3. Trial score = mean over folds of the best validation value of a FIXED criterion:
        mean over the three targets of the masked MSE in standardized units (equal weights).
     The criterion is fixed so that trials stay comparable while the loss weights are tuned.
     Early stopping monitors the same validation quantity.
  4. The winning configuration is written to results/nb9_tuned_config.json and is re-evaluated
     later by nb9_final.py in the same 11-seed protocol as every other row of the report.

usage: python nb9_tune.py [--trials 18] [--folds 5] [--max-epochs 500]
"""
import os
import sys
import time
import argparse

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (TARGETS, FEATURES, RESULTS, LOGS, CACHE, SPLIT_SEED, load_r3, make_split,
                    prepare, sound_stratify_key, build_model, ys_dict, WholeSetMaskedValLoss,
                    jdump, set_seed)

ap = argparse.ArgumentParser()
ap.add_argument("--trials", type=int, default=18)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--max-epochs", type=int, default=500)
ap.add_argument("--impute", default="train", choices=["train", "all"])
args = ap.parse_args()

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)


HEADS = ['Heat_Release_Output', 'CH_Consumption_Output', 'Bound_Water_Output']
EQ = {h: 1.0 / 3.0 for h in HEADS}          # fixed monitoring / scoring weights

r3 = load_r3()
tr, te, how = make_split(r3)
log(f"outer split: {how}; train {len(tr)} test {len(te)} (test rows are NOT used in this script)")
X, mask, y_std, mu, sd, _ = prepare(r3, tr, log=lambda m: None, impute=args.impute)
key_tr = sound_stratify_key(r3[TARGETS].iloc[tr])
log(f"label counts in training rows: "
    f"{ {t: int(r3[TARGETS].iloc[tr][t].notna().sum()) for t in TARGETS} }")

folds = list(StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SPLIT_SEED)
             .split(np.arange(len(tr)), key_tr))
for i, (f, v) in enumerate(folds):
    cnt = {t: int(r3[TARGETS].iloc[tr[v]][t].notna().sum()) for t in TARGETS}
    log(f"  fold {i}: fit {len(f)} val {len(v)}  validation labels {cnt}")

WIDTHS = {"256-256-16": (256, 256, 16), "256-128-32": (256, 128, 32),
          "512-256-32": (512, 256, 32), "128-128-16": (128, 128, 16),
          "256-256-64-16": (256, 256, 64, 16)}


def hp_space(hp):
    return dict(
        lr=hp.Float("lr", 1e-4, 3e-3, sampling="log"),
        dropout=hp.Choice("dropout", [0.0, 0.1, 0.2, 0.3]),
        l2=hp.Choice("l2", [0.0, 1e-5, 1e-4, 1e-3]),
        widths=hp.Choice("widths", list(WIDTHS)),
        batch=hp.Choice("batch", [32, 50, 128]),
        ch_mult=hp.Choice("ch_mult", [1.0, 3.0, 5.0, 10.0]),
        weighting=hp.Choice("weighting", ["notebook", "equal", "inverse_count"]),
        patience=hp.Choice("patience", [40, 70]),
    )


def head_weights(cfg, y_fit):
    """Training loss weights. 'notebook' = (total-count_i)/total with the Ca(OH)2 head x ch_mult."""
    c = {t: int(np.count_nonzero(~np.isnan(np.asarray(y_fit[t], dtype=float)))) for t in TARGETS}
    total = sum(c.values())
    if cfg["weighting"] == "notebook":
        w = {t: (total - c[t]) / total for t in TARGETS}
    elif cfg["weighting"] == "equal":
        w = {t: 1.0 for t in TARGETS}
    else:
        w = {t: total / max(c[t], 1) for t in TARGETS}
        s = sum(w.values())
        w = {t: 3 * v / s for t, v in w.items()}
    out = dict(zip(HEADS, [w['heat release'], w['CH consumption'], w['Bound water']]))
    out['CH_Consumption_Output'] *= cfg["ch_mult"]
    return out


def fit_one(cfg, fit_idx, val_idx, seed, max_epochs):
    """Train on fit_idx, early-stop on val_idx, return the best fixed-criterion value."""
    set_seed(seed)
    y_fit = y_std.iloc[fit_idx]
    lw = head_weights(cfg, y_fit)
    model = build_model(X.shape[1], lw, widths=WIDTHS[cfg["widths"]], dropout=cfg["dropout"],
                        l2=cfg["l2"], lr=cfg["lr"], seed=seed)
    cb = WholeSetMaskedValLoss([X[val_idx], mask[val_idx]], ys_dict(y_std, val_idx), EQ,
                               patience=cfg["patience"], restore=False)
    model.fit([X[fit_idx], mask[fit_idx]], ys_dict(y_std, fit_idx), epochs=max_epochs,
              batch_size=int(cfg["batch"]), verbose=0, callbacks=[cb])
    tf.keras.backend.clear_session()
    return cb.best, cb.best_epoch, len(cb.history)


class CVTuner(kt.RandomSearch):
    def run_trial(self, trial, **kwargs):
        cfg = hp_space(trial.hyperparameters)
        scores, epochs = [], []
        for i, (f, v) in enumerate(folds):
            s, be, ran = fit_one(cfg, tr[f], tr[v], seed=1000 + i, max_epochs=args.max_epochs)
            scores.append(s)
            epochs.append(be)
        m = float(np.mean(scores))
        log(f"trial {trial.trial_id}: score {m:.5f} (folds "
            f"{[round(s, 4) for s in scores]}, best epochs {epochs}) cfg "
            f"{ {k: (round(v, 6) if isinstance(v, float) else v) for k, v in cfg.items()} }")
        return {"cv_score": m, "mean_best_epoch": float(np.mean(epochs))}


tuner_dir = os.path.join(CACHE, "kt")
tuner = CVTuner(
    hypermodel=lambda hp: None,
    objective=kt.Objective("cv_score", direction="min"),
    max_trials=args.trials,
    seed=SPLIT_SEED,
    directory=tuner_dir,
    project_name=f"nb9_cv_{args.folds}fold",
    overwrite=True,
)
tuner.search()

best = tuner.oracle.get_best_trials(num_trials=min(5, args.trials))
log("\ntop configurations by cross-validated score:")
rows = []
for t in best:
    cfg = {k: t.hyperparameters.values[k] for k in t.hyperparameters.values}
    rows.append(dict(trial=t.trial_id, cv_score=t.score,
                     mean_best_epoch=t.metrics.get_last_value("mean_best_epoch"), **cfg))
    log(f"  {t.trial_id}  cv_score {t.score:.5f}  {cfg}")

import pandas as pd
pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "nb9_tuning_top5.csv"), index=False)

all_rows = []
for t in tuner.oracle.get_best_trials(num_trials=args.trials):
    all_rows.append(dict(trial=t.trial_id, cv_score=t.score,
                         mean_best_epoch=t.metrics.get_last_value("mean_best_epoch"),
                         **t.hyperparameters.values))
pd.DataFrame(all_rows).to_csv(os.path.join(RESULTS, "nb9_tuning_all_trials.csv"), index=False)

win = best[0]
cfg = dict(win.hyperparameters.values)
cfg["widths"] = list(WIDTHS[cfg["widths"]])
cfg["cv_score"] = float(win.score)
cfg["mean_best_epoch"] = float(win.metrics.get_last_value("mean_best_epoch"))
cfg["folds"] = args.folds
cfg["trials"] = args.trials
cfg["max_epochs"] = args.max_epochs
cfg["impute"] = args.impute
cfg["split_seed"] = SPLIT_SEED
cfg["selected_on"] = "mean over folds of equal-weight masked validation MSE (standardized); test rows never used"
jdump(cfg, os.path.join(RESULTS, "nb9_tuned_config.json"))
log(f"\nwrote {os.path.join(RESULTS, 'nb9_tuned_config.json')}: {cfg}")
log(f"total {time.time() - t0:.0f}s")
