"""Notebook 9, re-run FAITHFULLY (cell by cell, quirks included), plus a 'clean' variant.

faithful : exactly what the released notebook does --
             scaler and target standardization fitted on all 1,849 rows before the split;
             stratification bins taken from the unstandardized targets but np.digitize applied
             to the standardized ones; train/test split random_state=42 as in cell 14;
             validation = the last 15% of the training arrays (validation_split=0.15);
             early stopping on val_CH_Consumption_Output_custom_nan_nrmse, patience 200.
clean    : same architecture, loss weights and early-stopping monitor, but no test-row
             information in the scaler or the target statistics, stratification applied on the
             same scale it was built on, and an explicit random validation split.

The seed changes only the TensorFlow/NumPy initialization and, for 'clean', the validation
split. The train/test split is random_state=42 in both variants, so all seeds share one test
set of 370 rows -- the same size as the authors' saved output.

Neither variant ever uses test rows for early stopping or model selection.

usage: python nb9_faithful.py <faithful|clean> <seed> [output suffix]

The optional suffix keeps a second environment's results beside the main ones, e.g.
`nb9_faithful.py faithful 42 legacy` writes results/nb9_faithful_seed42_legacy.json.
"""
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (TARGETS, FEATURES, SHORT, RESULTS, ROOT, load_r3, load_df_imputed,
                    build_nb9_model, stratify_key_quirk, masked_metrics, unstandardize,
                    jdump, set_seed)

variant, seed = sys.argv[1], int(sys.argv[2])
assert variant in ("faithful", "clean")
SUFFIX = ("_" + sys.argv[3]) if len(sys.argv) > 3 else ""
SPLIT_RS = 42                      # notebook cell 14: random_state=42
MODELS = os.path.join(ROOT, "models", f"nb9_{variant}{SUFFIX}")
os.makedirs(MODELS, exist_ok=True)
t0 = time.time()


def log(m):
    print(f"[{variant} seed {seed} | {time.time() - t0:5.0f}s] {m}", flush=True)


set_seed(seed)
r3 = load_r3()
X = r3[FEATURES]
y = r3[TARGETS].copy()
log(f"R3 rows {len(r3)}  labels {y.notna().sum().to_dict()}")
input_mask = X.notnull().astype(int).values
X_imputed = load_df_imputed(lambda m: None)[FEATURES]

mean, std = y.mean(), y.std()           # cell 9: over ALL rows

if variant == "faithful":
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)          # cell 6: fitted on ALL rows
    y_std = (y - mean) / std                            # cell 9: ALL rows
    key = stratify_key_quirk(y, y_std)                  # cell 14 quirk
    X_train, X_test, mask_train, mask_test, y_train, y_test = train_test_split(
        X_scaled, input_mask, y_std, test_size=0.2, random_state=SPLIT_RS, stratify=key)
    strat = "stratified (notebook key: bins on unstandardized, digitize on standardized)"
    mu, sd = mean, std
    fit_kwargs = dict(validation_split=0.15)            # last 15% of the training arrays
    tr_rows = None
else:
    def bins(c):
        return np.digitize(y[c], np.linspace(y[c].min(), y[c].max(), 5))
    key = bins('heat release') + bins('CH consumption') * 5 + bins('Bound water') * 25
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SPLIT_RS, stratify=key)
    strat = "stratified (same scale for bins and digitize)"
    scaler = StandardScaler().fit(X_imputed.iloc[tr])
    X_scaled = scaler.transform(X_imputed)
    mu, sd = y.iloc[tr].mean(), y.iloc[tr].std()
    y_std = (y - mu) / sd
    X_train, X_test = X_scaled[tr], X_scaled[te]
    mask_train, mask_test = input_mask[tr], input_mask[te]
    y_train, y_test = y_std.iloc[tr], y_std.iloc[te]
    rel = np.random.RandomState(seed).permutation(len(tr))
    nval = int(round(0.15 * len(tr)))
    va, fi = rel[:nval], rel[nval:]
    fit_kwargs = dict(validation_data=([X_train[va], mask_train[va]],
                                       {'Heat_Release_Output': y_train['heat release'].values[va],
                                        'CH_Consumption_Output': y_train['CH consumption'].values[va],
                                        'Bound_Water_Output': y_train['Bound water'].values[va]}))
    tr_rows = (fi, va)
log(f"{strat}; train {len(X_train)} test {len(X_test)}")
log(f"test labels {y_test.notna().sum().to_dict()}")

model, w = build_nb9_model(X_scaled.shape[1], y_train)
log(f"loss weights {dict((k, round(v, 4)) for k, v in w.items())}")
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_CH_Consumption_Output_custom_nan_nrmse', patience=200, mode='min',
    restore_best_weights=True)

if variant == "faithful":
    xs, ysel = [X_train, mask_train], slice(None)
else:
    fi = tr_rows[0]
    xs, ysel = [X_train[fi], mask_train[fi]], fi
ys = {'Heat_Release_Output': y_train['heat release'].values[ysel],
      'CH_Consumption_Output': y_train['CH consumption'].values[ysel],
      'Bound_Water_Output': y_train['Bound water'].values[ysel]}

hist = model.fit(xs, ys, epochs=10000, batch_size=50, callbacks=[early_stopping], verbose=0,
                 **fit_kwargs)
mon = np.array(hist.history['val_CH_Consumption_Output_custom_nan_nrmse'])
finite = np.isfinite(mon)
best_ep = int(np.nanargmin(np.where(finite, mon, np.inf))) if finite.any() else 0
best_val = float(mon[best_ep])
log(f"stopped after {len(mon)} epochs; monitored value at epoch 1 = {mon[0]:.6g}; "
    f"restored epoch {best_ep + 1}; monitored value there {best_val:.6g}; "
    f"non-finite epochs {int((~finite).sum())}/{len(mon)}")

# PART 1.6 collapse rule -- validation behaviour only, never test performance
collapsed = (not np.isfinite(best_val)) or (best_ep + 1) <= 3 or best_val > 1.0
reason = ("monitor non-finite" if not np.isfinite(best_val) else
          "restored epoch <= 3" if (best_ep + 1) <= 3 else
          "monitored nRMSE > 1 (range term degenerate)" if best_val > 1.0 else "")
if collapsed:
    log(f"COLLAPSED early stopping: {reason}")


def ev(Xa, ma, ya):
    p = model.predict([Xa, ma], verbose=0)
    out = {}
    for i, t in enumerate(TARGETS):
        pu = unstandardize(p[i].flatten(), mu[t], sd[t])
        yu = unstandardize(ya[t].values, mu[t], sd[t])
        out[t] = masked_metrics(yu, pu)
    return out


res = dict(variant=variant, seed=seed, split_random_state=SPLIT_RS, strat=strat,
           r3_rows=len(r3), n_train=len(X_train), n_test=len(X_test),
           epochs=len(mon), best_epoch=best_ep + 1, monitor_at_best=best_val,
           monitor_epoch1=float(mon[0]), n_nonfinite_epochs=int((~finite).sum()),
           collapsed=bool(collapsed), collapse_reason=reason,
           weights={k: float(v) for k, v in w.items()},
           train=ev(X_train, mask_train, y_train), test=ev(X_test, mask_test, y_test),
           secs=time.time() - t0)
for t in TARGETS:
    log(f"{t:15s} train R2={res['train'][t]['R2']:+.3f}  test R2={res['test'][t]['R2']:+.3f} "
        f"RMSE={res['test'][t]['RMSE']:.2f} (n={res['test'][t]['n']})")
res["python"] = sys.version.split()[0]
res["tensorflow"] = tf.__version__
res["pandas"] = pd.__version__
jdump(res, os.path.join(RESULTS, f"nb9_{variant}_seed{seed}{SUFFIX}.json"))
try:
    model.save(os.path.join(MODELS, f"seed{seed}.keras"))
except Exception as e:
    # the .keras v3 format needs TensorFlow >= 2.13; the pinned legacy environment of Part 1.9
    # is older, and the metrics above are already written
    log(f"could not save the model in the .keras format ({e}); metrics are unaffected")
log("done")
