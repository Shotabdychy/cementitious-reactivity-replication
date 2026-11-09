"""Shared pieces for the replication of notebooks 9, 10 and 11 of
olivettigroup/Cementitious-reactivity-prediction-and-discovery.

Two kinds of code live here and are labelled:
  FAITHFUL  - copied from the authors' notebooks, quirks included.
  SOUND     - my replacement, used only in the "improved" pipeline. Every SOUND
              choice is documented in the report. No SOUND choice ever looks at
              the test set.
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.environ.get("REPL_DATA", os.path.join(ROOT, "data", "supplementary"))
GARD = os.environ.get("REPL_GARD", os.path.join(ROOT, "data", "gard"))
LOGS = os.path.join(ROOT, "logs")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
# REPL_CACHE lets a second environment (e.g. the pinned legacy stack of Part 1.9) keep its own
# cached tables: a DataFrame pickled by pandas 3 cannot be read back by pandas 1.5.
CACHE = os.environ.get("REPL_CACHE") or os.path.join(ROOT, "data_cache")
for _d in (LOGS, RESULTS, FIGURES, CACHE):
    os.makedirs(_d, exist_ok=True)

# ----------------------------------------------------------------- column names (notebook order)
FEATURES = ['SiO2', 'Al2O3', 'Fe2O3', 'CaO', 'MgO', 'SO3', 'TiO2', 'MnO', 'K2O', 'Na2O', 'LOI',
            'S gravity', 'Amorphous content', 'd50', 'Ca(OH)2', 'CaCO3', 'Water', 'K2SO4', 'KOH',
            'curing temp', 'age']
TARGETS = ['heat release', 'CH consumption', 'Bound water']
OXIDES = ['SiO2', 'Al2O3', 'Fe2O3', 'CaO', 'MgO', 'SO3', 'TiO2', 'MnO', 'K2O', 'Na2O', 'LOI']
MIX = ['Ca(OH)2', 'CaCO3', 'Water', 'K2SO4', 'KOH']
RENAME = {"Specific gravity (1)": "S gravity", "Amorphous content (%)": "Amorphous content",
          "d50 (micrometer)": "d50", "Ca(OH)2 (g)": "Ca(OH)2", "CaCO3 (g)": "CaCO3",
          "Water (g)": "Water", "K2SO4 (g)": "K2SO4", "KOH (g)": "KOH", "age (days)": "age",
          "heat release J/g": "heat release", "CH consumption g/100 g": "CH consumption",
          "Bound water  g/100g": "Bound water"}
# paper, Table 1 / Fig. 2 (test set)
PAPER = {'heat release': dict(R2=0.93, RMSE=28.20), 'CH consumption': dict(R2=0.85, RMSE=12.17),
         'Bound water': dict(R2=0.89, RMSE=1.47)}
SHORT = {'heat release': 'heat', 'CH consumption': 'CH', 'Bound water': 'BW'}


# ----------------------------------------------------------------- data
def load_r3(rows="scm"):
    """Dataset 4, standing in for the authors' missing R3.csv.

    DEVIATION (documented): Dataset 4 stores the mix in raw grams, so the mix columns are
    divided by 'SCM (g)'. The authors' saved R3.csv rows show Ca(OH)2 = 3.0 per gram of SCM,
    which is what this division produces.
    """
    df = pd.read_csv(os.path.join(DATA, "Dataset 4.csv"), encoding="utf-8-sig").rename(columns=RENAME)
    keep_name = df['Material name'].astype(str).str.strip()
    keep_doi = df['DOI'].astype(str).str.strip()
    for c in FEATURES + TARGETS + ['SCM (g)']:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in MIX:
        df[c] = df[c] / df['SCM (g)']
    df['material_name'] = keep_name
    df['doi'] = keep_doi
    if rows == "scm":
        df = df[df['SCM (g)'].notna()]
    return df.reset_index(drop=True)


# ----------------------------------------------------------------- FAITHFUL: notebook 10/11 imputation
def lightgbm_impute(data, target_column, features):
    known = data[data[target_column].notna()]
    unknown = data[data[target_column].isna()]
    if unknown.shape[0] == 0:
        return data
    X_train = known[features].values
    y_train = known[target_column].values
    X_test = unknown[features].values
    if issubclass(data[target_column].dtype.type, np.integer):
        model = lgb.LGBMClassifier(verbose=-1)
    else:
        model = lgb.LGBMRegressor(verbose=-1)
    model.fit(X_train, y_train)
    predicted_values = model.predict(X_test)
    data.loc[data[target_column].isna(), target_column] = predicted_values
    return data


def impute_notebook(df, log=print):
    """One sequential in-place pass over the columns, as in notebooks 10 and 11."""
    columns = df.columns.tolist()
    for column in columns:
        if df[column].isna().any():
            df = lightgbm_impute(df, column, [col for col in columns if col != column])
            log(f"  imputed {column}")
    return df


# ----------------------------------------------------------------- FAITHFUL: notebook 9 losses
def custom_nan_mse(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.is_nan(y_true))
    mask = tf.cast(mask, dtype=tf.float32)
    y_true = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
    squared_difference = tf.square(y_true - y_pred) * mask
    epsilon = 1e-7
    return tf.reduce_sum(squared_difference) / (tf.reduce_sum(mask) + epsilon)


def custom_nan_nrmse(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.is_nan(y_true))
    mask = tf.cast(mask, dtype=tf.float32)
    y_true = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
    squared_difference = tf.square(y_true - y_pred) * mask
    mse = tf.reduce_sum(squared_difference) / tf.reduce_sum(mask)
    rmse = tf.sqrt(mse)
    y_true_masked_min = tf.where(mask > 0, y_true, tf.fill(tf.shape(y_true), tf.float32.max))
    y_true_masked_max = tf.where(mask > 0, y_true, tf.fill(tf.shape(y_true), tf.float32.min))
    y_true_min = tf.reduce_min(y_true_masked_min)
    y_true_max = tf.reduce_max(y_true_masked_max)
    y_range = y_true_max - y_true_min
    epsilon = 1e-7
    return rmse / (y_range + epsilon)


CUSTOM_OBJECTS = {'custom_nan_mse': custom_nan_mse, 'custom_nan_nrmse': custom_nan_nrmse}


def unstandardize(values, mean, std):
    return values * std + mean


def loss_weights_from(y_train):
    """FAITHFUL notebook 9 cell 13: w_i = (total - count_i) / total, with the Ca(OH)2 head x5."""
    c = {t: int(np.count_nonzero(~np.isnan(np.asarray(y_train[t], dtype=float)))) for t in TARGETS}
    total = sum(c.values())
    w = {t: (total - c[t]) / total for t in TARGETS}
    return {'Heat_Release_Output': w['heat release'],
            'CH_Consumption_Output': w['CH consumption'] * 5,
            'Bound_Water_Output': w['Bound water']}, c


def build_nb9_model(n_features, y_train):
    """FAITHFUL architecture of notebook 9 cell 13 (256-256-16, dropout 0.1, three heads)."""
    from tensorflow.keras.layers import Input, Dense, concatenate, Dropout
    from tensorflow.keras.models import Model
    input_features = Input(shape=(n_features,), name='Input_Features')
    input_masks = Input(shape=(n_features,), name='Input_Masks')
    combined_inputs = concatenate([input_features, input_masks])
    hidden1 = Dense(256, activation='relu')(combined_inputs)
    hidden2 = Dropout(0.1)(hidden1)
    hidden3 = Dense(256, activation='relu')(hidden2)
    hidden4 = Dense(16, activation='relu')(hidden3)
    out = [Dense(1, name='Heat_Release_Output')(hidden4),
           Dense(1, name='CH_Consumption_Output')(hidden4),
           Dense(1, name='Bound_Water_Output')(hidden4)]
    model = Model(inputs=[input_features, input_masks], outputs=out)
    lw, counts = loss_weights_from(y_train)
    model.compile(optimizer='adam',
                  loss={k: custom_nan_mse for k in lw},
                  loss_weights=lw,
                  metrics={k: [custom_nan_mse, custom_nan_nrmse] for k in lw})
    return model, lw


def stratify_key_quirk(y_unstd, y_std):
    """FAITHFUL notebook 9 cell 15: bins from the UNSTANDARDIZED target range, but np.digitize
    applied to the STANDARDIZED targets."""
    num_bins = 5
    b = {c: np.linspace(y_unstd[c].min(), y_unstd[c].max(), num_bins) for c in TARGETS}
    hr = np.digitize(y_std['heat release'], b['heat release'])
    ch = np.digitize(y_std['CH consumption'], b['CH consumption'])
    bw = np.digitize(y_std['Bound water'], b['Bound water'])
    return hr + (ch * num_bins) + (bw * (num_bins ** 2))


# ----------------------------------------------------------------- SOUND pieces
def sound_stratify_key(y):
    """SOUND (Part 1.3). The notebook's key is degenerate (6 classes, built on one scale and
    applied on another). This key is built and applied on the same, unstandardized values:

      availability pattern of the three targets (which are labelled)  x  tertile of heat release.

    Stratifying on availability is what guarantees that the test and validation sets contain
    labelled rows for all three targets -- the thing that breaks the authors' early stopping.
    """
    avail = (y['heat release'].notna().astype(int) * 4
             + y['CH consumption'].notna().astype(int) * 2
             + y['Bound water'].notna().astype(int))
    hr = y['heat release']
    q = pd.Series(0, index=y.index, dtype=int)
    ok = hr.notna()
    if ok.sum() > 10:
        q[ok] = pd.qcut(hr[ok], 3, labels=False, duplicates='drop') + 1
    return (avail.astype(str) + "_" + q.astype(str)).values


# The outer train/test split is fixed by this seed for every SOUND run, so that the 11 model
# seeds share one test set (needed for an ensemble) and so that tuning can never see it.
SPLIT_SEED = 0


def make_split(r3, split_seed=SPLIT_SEED, grouped=None, test_size=0.2):
    """SOUND (Parts 1.2, 1.3, 1.7). Outer train/test split only; the validation split is drawn
    from the training rows later, per model seed.

    grouped=None        : row-level split, stratified by sound_stratify_key.
    grouped='material'  : all rows of one material name in the same side (GroupShuffleSplit).
    grouped='doi'       : all rows of one source paper in the same side.
    """
    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    y = r3[TARGETS]
    idx = np.arange(len(r3))
    if grouped is None:
        key = sound_stratify_key(y)
        tr, te = train_test_split(idx, test_size=test_size, random_state=split_seed, stratify=key)
        how = "row-level, stratified by target-availability x heat tertile"
    else:
        groups = r3['material_name' if grouped == 'material' else 'doi'].values
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=split_seed)
        tr, te = next(gss.split(idx, groups=groups))
        how = f"grouped by {grouped} ({len(np.unique(groups))} groups)"
    return np.sort(tr), np.sort(te), how


def inner_val_split(r3, tr, seed, val_frac=0.15, grouped=None):
    """SOUND (Part 1.2). Validation rows drawn at random from the TRAINING rows, stratified by
    the same key, so every target has labelled validation rows. Never the last 15% of the array."""
    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    if grouped is None:
        key = sound_stratify_key(r3[TARGETS].iloc[tr])
        fit, va = train_test_split(np.arange(len(tr)), test_size=val_frac,
                                   random_state=seed, stratify=key)
    else:
        groups = r3['material_name' if grouped == 'material' else 'doi'].values[tr]
        gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        fit, va = next(gss.split(np.arange(len(tr)), groups=groups))
    return tr[np.sort(fit)], tr[np.sort(va)]


def masked_metrics(y_true, y_pred):
    """R2 and RMSE on the labelled rows only, in physical units."""
    from sklearn.metrics import r2_score, mean_squared_error
    ok = ~np.isnan(y_true)
    if ok.sum() < 2:
        return dict(n=int(ok.sum()), R2=float('nan'), RMSE=float('nan'))
    return dict(n=int(ok.sum()),
                R2=float(r2_score(y_true[ok], y_pred[ok])),
                RMSE=float(np.sqrt(mean_squared_error(y_true[ok], y_pred[ok]))))


def bootstrap_ci(y_true, y_pred, n_boot=2000, seed=0):
    """SOUND (Part 1.5): percentile bootstrap 95% CI for R2 and RMSE over the labelled rows."""
    from sklearn.metrics import r2_score, mean_squared_error
    ok = ~np.isnan(y_true)
    yt, yp = y_true[ok], y_pred[ok]
    n = len(yt)
    if n < 5:
        return dict(R2_lo=float('nan'), R2_hi=float('nan'), RMSE_lo=float('nan'), RMSE_hi=float('nan'))
    rng = np.random.default_rng(seed)
    r2s, rmses = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(yt[idx])) < 2:
            r2s[b], rmses[b] = np.nan, np.nan
            continue
        r2s[b] = r2_score(yt[idx], yp[idx])
        rmses[b] = np.sqrt(mean_squared_error(yt[idx], yp[idx]))
    return dict(R2_lo=float(np.nanpercentile(r2s, 2.5)), R2_hi=float(np.nanpercentile(r2s, 97.5)),
                RMSE_lo=float(np.nanpercentile(rmses, 2.5)), RMSE_hi=float(np.nanpercentile(rmses, 97.5)))


class WholeSetMaskedValLoss(tf.keras.callbacks.Callback):
    """SOUND (Part 1.1). Replacement for the notebook's early-stopping monitor.

    The notebook monitors 'val_CH_Consumption_Output_custom_nan_nrmse': a normalized RMSE
    averaged over validation BATCHES of 50 rows. With 192 Ca(OH)2 labels in 1,849 rows a batch
    often holds 0 or 1 label, so the quantity is NaN or ~1e6 and early stopping restores
    whatever epoch that artefact favoured.

    This callback evaluates the whole validation set in one pass and reports
    'val_masked' = sum over the three heads of loss_weight_i * masked MSE_i (standardized units),
    i.e. the same objective the model is trained on, measured on all validation rows at once.
    """

    def __init__(self, val_x, val_y, loss_weights, patience, restore=True, verbose=0,
                 min_epochs=0):
        super().__init__()
        self.min_epochs = min_epochs
        self.val_x = [tf.convert_to_tensor(np.asarray(a, dtype='float32')) for a in val_x]
        self.val_y = {k: np.asarray(v, dtype=float) for k, v in val_y.items()}
        self.lw = loss_weights
        self.patience = patience
        self.restore = restore
        self.verbose = verbose
        self.history = []
        # masks and labels as plain arrays, prepared once
        self._ok = {k: ~np.isnan(v) for k, v in self.val_y.items()}
        self._yv = {k: v[self._ok[k]] for k, v in self.val_y.items()}

    def on_train_begin(self, logs=None):
        self.best = np.inf
        self.best_epoch = 0
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0

    def _value(self):
        # one forward pass over the whole validation set (no batching, no predict() overhead)
        p = self.model(self.val_x, training=False)
        p = [np.asarray(a).flatten() for a in p]
        total, parts = 0.0, {}
        for arr, (head, w), t in zip(p, self.lw.items(), TARGETS):
            ok = self._ok[head]
            mse = float(np.mean((self._yv[head] - arr[ok]) ** 2)) if ok.any() else 0.0
            parts[t] = mse
            total += w * mse
        return total, parts

    def on_epoch_end(self, epoch, logs=None):
        v, parts = self._value()
        logs = logs if logs is not None else {}
        logs['val_masked'] = v
        for t, m in parts.items():
            logs[f'val_mse_{SHORT[t]}'] = m
        self.history.append(v)
        if np.isfinite(v) and v < self.best - 1e-9:
            self.best, self.best_epoch, self.wait = v, epoch + 1, 0
            if self.restore:
                self.best_weights = [w.copy() for w in self.model.get_weights()]
        else:
            self.wait += 1
            if self.wait >= self.patience and epoch + 1 >= self.min_epochs:
                self.stopped_epoch = epoch + 1
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.restore and self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def build_model(n_features, loss_weights, widths=(256, 256, 16), dropout=0.1, l2=0.0,
                lr=1e-3, seed=None):
    """SOUND (Part 1.4) parameterized version of the notebook architecture.
    widths=(256,256,16), dropout=0.1, l2=0, lr=1e-3 reproduces the notebook's network exactly."""
    from tensorflow.keras.layers import Input, Dense, concatenate, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2 as L2
    reg = L2(l2) if l2 else None
    init = 'glorot_uniform'
    xf = Input(shape=(n_features,), name='Input_Features')
    xm = Input(shape=(n_features,), name='Input_Masks')
    h = concatenate([xf, xm])
    for i, w in enumerate(widths):
        h = Dense(w, activation='relu', kernel_regularizer=reg, kernel_initializer=init)(h)
        if i == 0 and dropout:
            h = Dropout(dropout, seed=seed)(h)
    out = [Dense(1, name='Heat_Release_Output')(h),
           Dense(1, name='CH_Consumption_Output')(h),
           Dense(1, name='Bound_Water_Output')(h)]
    model = Model(inputs=[xf, xm], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss={k: custom_nan_mse for k in loss_weights},
                  loss_weights=loss_weights)
    return model


def ys_dict(y_df, idx=None):
    sel = y_df if idx is None else y_df.iloc[idx]
    return {'Heat_Release_Output': sel['heat release'].values.astype(float),
            'CH_Consumption_Output': sel['CH consumption'].values.astype(float),
            'Bound_Water_Output': sel['Bound water'].values.astype(float)}


def impute_fit_apply(df, tr, log=print):
    """SOUND. LightGBM feature imputation whose models are fitted on TRAINING rows only and then
    applied to every row. Same column order and same estimator as impute_notebook; the only
    difference is that rows outside `tr` never contribute to a fitted model.
    (impute_notebook fits on all rows, so test rows inform the imputed values.)
    """
    out = df.copy()
    columns = out.columns.tolist()
    tr = np.asarray(tr)
    for column in columns:
        if not out[column].isna().any():
            continue
        feats = [c for c in columns if c != column]
        known_tr = out.index[out[column].notna()].intersection(out.index[tr])
        if len(known_tr) < 20:
            out.loc[out[column].isna(), column] = out.loc[out.index[tr], column].median()
            log(f"  imputed {column} (median; only {len(known_tr)} training observations)")
            continue
        m = lgb.LGBMRegressor(verbose=-1)
        m.fit(out.loc[known_tr, feats].values, out.loc[known_tr, column].values)
        miss = out[column].isna()
        out.loc[miss, column] = m.predict(out.loc[miss, feats].values)
        log(f"  imputed {column} (fit on {len(known_tr)} training rows)")
    return out


def prepare(r3, tr, log=print, impute="train"):
    """SOUND (Part 1.3). Scaler and target standardization fitted on the TRAINING rows only.

    The notebook fits both on all rows before the split, so test-row statistics leak into the
    inputs. Here nothing outside `tr` is seen by either transform.
    impute='train' : feature imputation models fitted on training rows only (SOUND default).
    impute='all'   : the notebook's imputation over all rows (sensitivity row in the report).
    Returns X (scaled, all rows), mask, y_std, mu, sd, scaler.
    """
    from sklearn.preprocessing import StandardScaler
    mask = r3[FEATURES].notnull().astype(int).values
    if impute == "all":
        Xi = load_df_imputed(log)[FEATURES]
    else:
        import hashlib, pickle
        h = hashlib.md5(np.asarray(tr).tobytes()).hexdigest()[:12]
        p = os.path.join(CACHE, f"imputed_trainonly_{h}.pkl")
        if os.path.exists(p):
            Xi = pickle.load(open(p, "rb"))
        else:
            Xi = impute_fit_apply(r3[FEATURES].copy(), tr, log=log)
            Xi[Xi < 0] = 0
            pickle.dump(Xi, open(p, "wb"))
    sc = StandardScaler().fit(Xi.iloc[tr])
    X = sc.transform(Xi).astype('float32')
    y = r3[TARGETS]
    mu, sd = y.iloc[tr].mean(), y.iloc[tr].std()
    y_std = (y - mu) / sd
    return X, mask.astype('float32'), y_std, mu, sd, sc


def evaluate_preds(pred_std, y_std, mu, sd, rows):
    """R2/RMSE in physical units on `rows`, per target."""
    out = {}
    for i, t in enumerate(TARGETS):
        pu = unstandardize(np.asarray(pred_std[i]).flatten()[rows], mu[t], sd[t])
        yu = unstandardize(y_std[t].values[rows], mu[t], sd[t])
        out[t] = masked_metrics(yu, pu)
    return out


def load_df_imputed(log=print, path=None):
    """Substitute for the missing df_imputed.pkl: notebook-style LightGBM imputation of the R3
    features, clipped at 0 as in notebook 10. Cached so notebooks 9/10/11 share one table."""
    import pickle
    path = path or os.path.join(CACHE, "df_imputed_r3.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    r3 = load_r3()
    Ximp = impute_notebook(r3[FEATURES].copy(), log=log)
    Ximp[Ximp < 0] = 0
    out = pd.concat([Ximp, r3[TARGETS]], axis=1)
    pickle.dump(out, open(path, "wb"))
    return out


def key_name(s):
    """Material name reduced to lowercase alphanumerics, for matching across sources."""
    import re
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def _affix_match(t, p):
    """True if one key is the other with an extra prefix or suffix, but never when the extra part
    would extend a number: 'snm1' must not match 'snm14' (digit follows), while 'pc1' matches
    'pc1cemi425r' and 'snm1' matches 'mortarsnm1'."""
    if t == p:
        return True
    short, long_ = (t, p) if len(t) < len(p) else (p, t)
    if not short:
        return False
    if long_.startswith(short):
        return not long_[len(short)].isdigit() or not short[-1].isdigit()
    if long_.endswith(short):
        before = long_[-len(short) - 1]
        return not (before.isdigit() and short[0].isdigit())
    return False


def pair_materials(true_keys, pred_keys):
    """Pair material names across two sources: strict equality first, then a prefix/suffix
    fallback for naming-convention differences (see _affix_match). A candidate is accepted only
    when exactly one unused candidate remains. Returns (pairs, number of strict matches)."""
    pairs, used_t, used_p = [], set(), set()
    for k in true_keys:
        if k and k in pred_keys:
            pairs.append((k, k))
            used_t.add(k)
            used_p.add(k)
    n_strict = len(pairs)
    for t in true_keys:
        if t in used_t or not t:
            continue
        cands = [p for p in pred_keys if p and p not in used_p and _affix_match(t, p)]
        if len(cands) == 1:
            pairs.append((t, cands[0]))
            used_t.add(t)
            used_p.add(cands[0])
    return pairs, n_strict


def jdump(obj, path):
    def d(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(str(type(o)))
    json.dump(obj, open(path, "w"), indent=1, default=d)


def set_seed(seed):
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
