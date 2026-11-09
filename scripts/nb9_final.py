"""PARTS 1.5 - 1.8 -- final notebook-9 training under the sound protocol.

For a given configuration ('notebook' = the released architecture and loss weights, or 'tuned' =
the winner of nb9_tune.py) this trains N seeds, forms the seed-mean ensemble, and reports
per-seed R2/RMSE, mean, SD and a bootstrap 95% CI on the test set.

Everything that selects anything -- early stopping, the ensemble members, the tuned
configuration -- uses validation rows only. The test rows are read exactly once per run, at the
end, to produce the reported numbers.

usage:
  python nb9_final.py --config notebook --seeds 11 --tag sound_notebook
  python nb9_final.py --config tuned    --seeds 11 --tag sound_tuned
  python nb9_final.py --config tuned    --seeds 11 --grouped material --tag grouped_material
  python nb9_final.py --config tuned    --seeds 3  --impute all --tag sens_impute_all
"""
import os
import sys
import time
import json
import pickle
import argparse

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (TARGETS, FEATURES, SHORT, PAPER, RESULTS, LOGS, ROOT, SPLIT_SEED, load_r3,
                    make_split, inner_val_split, prepare, build_model, ys_dict,
                    WholeSetMaskedValLoss, masked_metrics, bootstrap_ci, unstandardize,
                    jdump, set_seed)

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="notebook", choices=["notebook", "tuned"])
ap.add_argument("--seeds", type=int, default=11)
ap.add_argument("--grouped", default="none", choices=["none", "material", "doi"])
ap.add_argument("--impute", default="train", choices=["train", "all"])
ap.add_argument("--max-epochs", type=int, default=2000)
ap.add_argument("--tag", default=None)
ap.add_argument("--split-seed", type=int, default=SPLIT_SEED)
args = ap.parse_args()
grouped = None if args.grouped == "none" else args.grouped
TAG = args.tag or f"{args.config}_{args.grouped}"
MODELS = os.path.join(ROOT, "models", TAG)
os.makedirs(MODELS, exist_ok=True)

t0 = time.time()


def log(m):
    print(f"[{TAG} | {time.time() - t0:6.0f}s] {m}", flush=True)


HEADS = ['Heat_Release_Output', 'CH_Consumption_Output', 'Bound_Water_Output']
EQ = {h: 1.0 / 3.0 for h in HEADS}      # fixed early-stopping criterion (validation only)

# ------------------------------------------------------------------ configuration
if args.config == "notebook":
    CFG = dict(lr=1e-3, dropout=0.1, l2=0.0, widths=[256, 256, 16], batch=50,
               ch_mult=5.0, weighting="notebook", patience=200)
    log("configuration: the released notebook-9 network (256-256-16, dropout 0.1, Adam 1e-3, "
        "batch 50, notebook loss weights with the Ca(OH)2 head x5, patience 200)")
else:
    p = os.path.join(RESULTS, "nb9_tuned_config.json")
    CFG = json.load(open(p))
    log(f"configuration: tuned, from {p} (cv_score {CFG.get('cv_score')})")
log(f"CFG = { {k: v for k, v in CFG.items() if k in ('lr','dropout','l2','widths','batch','ch_mult','weighting','patience')} }")

# ------------------------------------------------------------------ data and split
r3 = load_r3()
tr, te, how = make_split(r3, split_seed=args.split_seed, grouped=grouped)
log(f"outer split: {how} (split_seed {args.split_seed}); train {len(tr)} test {len(te)}")
lab_tr = {t: int(r3[TARGETS].iloc[tr][t].notna().sum()) for t in TARGETS}
lab_te = {t: int(r3[TARGETS].iloc[te][t].notna().sum()) for t in TARGETS}
log(f"labelled rows  train {lab_tr}  test {lab_te}")
X, mask, y_std, mu, sd, scaler = prepare(r3, tr, log=lambda m: None, impute=args.impute)
log(f"feature imputation: {args.impute}-rows; scaler and target standardization fitted on the "
    f"{len(tr)} training rows only")


def head_weights(y_fit):
    c = {t: int(np.count_nonzero(~np.isnan(np.asarray(y_fit[t], dtype=float)))) for t in TARGETS}
    total = sum(c.values())
    if CFG["weighting"] == "notebook":
        w = {t: (total - c[t]) / total for t in TARGETS}
    elif CFG["weighting"] == "equal":
        w = {t: 1.0 for t in TARGETS}
    else:
        w = {t: total / max(c[t], 1) for t in TARGETS}
        s = sum(w.values())
        w = {t: 3 * v / s for t, v in w.items()}
    out = dict(zip(HEADS, [w['heat release'], w['CH consumption'], w['Bound water']]))
    out['CH_Consumption_Output'] *= CFG["ch_mult"]
    return out


# ------------------------------------------------------------------ train the seeds
SEEDS = [42] + list(range(1, args.seeds)) if args.seeds > 1 else [42]
SEEDS = SEEDS[:args.seeds]
runs = []
pred_std_test = []        # standardized predictions on the test rows, per seed
pred_std_all = []         # on every R3 row (for the parity plot of the training rows)

for seed in SEEDS:
    set_seed(seed)
    fit_idx, va_idx = inner_val_split(r3, tr, seed, val_frac=0.15, grouped=grouped)
    lab_va = {t: int(r3[TARGETS].iloc[va_idx][t].notna().sum()) for t in TARGETS}
    lw = head_weights(y_std.iloc[fit_idx])
    model = build_model(X.shape[1], lw, widths=tuple(CFG["widths"]), dropout=CFG["dropout"],
                        l2=CFG["l2"], lr=CFG["lr"], seed=seed)
    cb = WholeSetMaskedValLoss([X[va_idx], mask[va_idx]], ys_dict(y_std, va_idx), EQ,
                               patience=int(CFG["patience"]), restore=True)
    h = model.fit([X[fit_idx], mask[fit_idx]], ys_dict(y_std, fit_idx),
                  epochs=args.max_epochs, batch_size=int(CFG["batch"]), verbose=0, callbacks=[cb])
    p_all = [np.asarray(a).flatten() for a in model(
        [tf.convert_to_tensor(X), tf.convert_to_tensor(mask)], training=False)]
    pred_std_all.append(p_all)
    pred_std_test.append([a[te] for a in p_all])

    def ev(rows):
        out = {}
        for i, t in enumerate(TARGETS):
            pu = unstandardize(p_all[i][rows], mu[t], sd[t])
            yu = unstandardize(y_std[t].values[rows], mu[t], sd[t])
            out[t] = masked_metrics(yu, pu)
        return out

    # Part 1.6: collapse rule, defined on validation behaviour only.
    collapsed = (not np.isfinite(cb.best)) or cb.best_epoch <= 3
    rec = dict(seed=seed, n_fit=len(fit_idx), n_val=len(va_idx), val_labels=lab_va,
               epochs_run=len(cb.history), best_epoch=cb.best_epoch,
               val_criterion=float(cb.best), collapsed=bool(collapsed),
               loss_weights={k: float(v) for k, v in lw.items()},
               train=ev(fit_idx), val=ev(va_idx), test=ev(te))
    runs.append(rec)
    log(f"seed {seed:3d}: epochs {len(cb.history):4d} best {cb.best_epoch:4d} "
        f"val_crit {cb.best:.5f}{' COLLAPSED' if collapsed else ''} | test R2 " +
        " ".join(f"{SHORT[t]} {rec['test'][t]['R2']:+.3f}" for t in TARGETS))
    model.save(os.path.join(MODELS, f"seed{seed}.keras"))
    tf.keras.backend.clear_session()

# ------------------------------------------------------------------ ensemble + statistics
ens_all = [np.mean([p[i] for p in pred_std_all], axis=0) for i in range(3)]
good = [i for i, r in enumerate(runs) if not r["collapsed"]]
ens_good = [np.mean([pred_std_all[i][k] for i in good], axis=0) for k in range(3)] if good else None


def report_block(pred, rows, name, boot=True):
    out = {}
    for i, t in enumerate(TARGETS):
        pu = unstandardize(pred[i][rows], mu[t], sd[t])
        yu = unstandardize(y_std[t].values[rows], mu[t], sd[t])
        m = masked_metrics(yu, pu)
        if boot:
            m.update(bootstrap_ci(yu, pu, n_boot=2000, seed=7))
        out[t] = m
    log(f"{name}: " + " | ".join(
        f"{SHORT[t]} R2 {out[t]['R2']:+.3f}"
        + (f" [{out[t]['R2_lo']:+.3f},{out[t]['R2_hi']:+.3f}]" if boot else "")
        + f" RMSE {out[t]['RMSE']:.2f}" for t in TARGETS))
    return out


summary = dict(tag=TAG, config=CFG, grouped=args.grouped, impute=args.impute,
               split_seed=args.split_seed, split_how=how,
               n_train=len(tr), n_test=len(te), labels_train=lab_tr, labels_test=lab_te,
               seeds=SEEDS, runs=runs,
               collapse_rule="validation criterion non-finite, or restored epoch <= 3",
               n_collapsed=int(len(runs) - len(good)))

per_seed = {t: np.array([r["test"][t]["R2"] for r in runs], dtype=float) for t in TARGETS}
per_seed_rmse = {t: np.array([r["test"][t]["RMSE"] for r in runs], dtype=float) for t in TARGETS}
summary["per_seed_mean_sd"] = {
    t: dict(R2_mean=float(np.nanmean(per_seed[t])), R2_sd=float(np.nanstd(per_seed[t], ddof=1)) if len(runs) > 1 else float('nan'),
            R2_min=float(np.nanmin(per_seed[t])), R2_max=float(np.nanmax(per_seed[t])),
            RMSE_mean=float(np.nanmean(per_seed_rmse[t])),
            RMSE_sd=float(np.nanstd(per_seed_rmse[t], ddof=1)) if len(runs) > 1 else float('nan'))
    for t in TARGETS}
log("per-seed test R2 mean +- SD: " + " | ".join(
    f"{SHORT[t]} {summary['per_seed_mean_sd'][t]['R2_mean']:+.3f} +- "
    f"{summary['per_seed_mean_sd'][t]['R2_sd']:.3f} "
    f"(min {summary['per_seed_mean_sd'][t]['R2_min']:+.3f}, max {summary['per_seed_mean_sd'][t]['R2_max']:+.3f})"
    for t in TARGETS))
if len(runs) > len(good):
    sub = {t: np.array([runs[i]["test"][t]["R2"] for i in good]) for t in TARGETS}
    summary["per_seed_mean_sd_noncollapsed"] = {
        t: dict(n=len(good), R2_mean=float(np.nanmean(sub[t])),
                R2_sd=float(np.nanstd(sub[t], ddof=1)) if len(good) > 1 else float('nan'))
        for t in TARGETS}
    log(f"non-collapsed runs only ({len(good)}/{len(runs)}): " + " | ".join(
        f"{SHORT[t]} {summary['per_seed_mean_sd_noncollapsed'][t]['R2_mean']:+.3f}" for t in TARGETS))

summary["ensemble_test"] = report_block(ens_all, te, f"ENSEMBLE of {len(runs)} seeds, test")
summary["ensemble_train"] = report_block(ens_all, tr, f"ENSEMBLE of {len(runs)} seeds, train", boot=False)
if good and len(good) < len(runs):
    summary["ensemble_test_noncollapsed"] = report_block(
        ens_good, te, f"ENSEMBLE of {len(good)} non-collapsed seeds, test")

log("paper (test): " + " | ".join(f"{SHORT[t]} R2 {PAPER[t]['R2']} RMSE {PAPER[t]['RMSE']}" for t in TARGETS))
summary["paper"] = PAPER
summary["secs"] = time.time() - t0
jdump(summary, os.path.join(RESULTS, f"nb9_{TAG}.json"))

# per-row predictions of the ensemble, for the parity plots and for reuse
out = pd.DataFrame({'row': np.arange(len(r3)), 'material_name': r3.material_name, 'doi': r3.doi,
                    'split': np.where(np.isin(np.arange(len(r3)), te), 'test', 'train')})
for i, t in enumerate(TARGETS):
    out[f'y_{SHORT[t]}'] = unstandardize(y_std[t].values, mu[t], sd[t])
    out[f'pred_{SHORT[t]}'] = unstandardize(ens_all[i], mu[t], sd[t])
out.to_csv(os.path.join(RESULTS, f"nb9_{TAG}_predictions.csv"), index=False)

pickle.dump(dict(scaler=scaler, mu=mu, sd=sd, tr=tr, te=te, cfg=CFG, seeds=SEEDS,
                 impute=args.impute, grouped=args.grouped),
            open(os.path.join(MODELS, "bundle.pkl"), "wb"))
log(f"wrote results/nb9_{TAG}.json, results/nb9_{TAG}_predictions.csv, models/{TAG}/")
log("done")
