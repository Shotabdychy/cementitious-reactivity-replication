# Replication report

**Paper.** S. Mahjoubi, V. Venugopal, I. B. Manav, H. Azarijafari, R. Kirchain, E. Olivetti,
*Data-driven material screening of secondary and natural cementitious precursors*,
Communications Materials **6**, 99 (2025).
**Code.** `olivettigroup/Cementitious-reactivity-prediction-and-discovery` (14 notebooks).
**This replication.** {{DATE}} · Windows 11, Python 3.11.9, 8 CPU cores, no GPU ·
all scripts, logs and result files in the accompanying folder.

## 1. Summary

{{TABLE_CLAIMS}}

**What was run.** Notebooks 9, 10 and 11 were re-run end to end: faithfully (copying the
released code, quirks included) and again under a protocol that fixes the parts of the released
code that make a single run unreliable, with 5-fold cross-validated tuning on the training rows,
11 seeds, a seed ensemble and bootstrap confidence intervals. Notebooks 1-8 cannot be run as
released; the extraction and classification steps they perform were reproduced on a sample of
tables with a substitute language model.

**What matched.** {{WHAT_MATCHED}} The qualitative picture of Fig. 3 reproduces: metakaolin is
the most reactive category, fly ashes are pozzolanic, slags are hydraulic, lime species are
inert. The literature-mining step reproduces well on the tables that could be scored: of
{{FP_CELLS}} oxide values extracted from {{FP_NTABLES}} tables, {{FP_EXACT}} agree exactly with the
authors' manual ground truth (a fourth table could not be scored: its ground truth belongs to a
different table).

**What did not match.** Every count that depends on the authors' trained weights
(`90andupnewones.keras`, `90andupnewonesch.keras`) or on an unpublished decision rule. The
natural-rock screening is the largest gap: the notebook as released scores
{{NB11_REACTIVE_ALL}} of 1,022,092 samples as reactive against the paper's 50,569.

**Why.** Five files the notebooks load are not in the repository, and three decisions the paper
reports are not in the code: which rock rows are screened, how rock names are grouped, and how a
reactive rock is called pozzolanic or hydraulic. Section 9 lists what would be needed.



## 2. Scope, and what had to be substituted

| Stage | Notebooks | What happened here |
|---|---|---|
| Literature filtering, table and sentence retrieval | 1-3, 6 | Not runnable. `MongoClient().predsynth.papers_v2` is the group's private licensed corpus. |
| Table extraction, extraction evaluation | 4, 5 | Reproduced on 6 tables with a substitute extractor (Part 4). |
| Descriptor extraction, material typing | 7, 8 | Reproduced on 40 unseen Dataset 3 rows with a substitute classifier (Part 4). |
| Reactivity network | 9 | Re-run: faithful, clean, and sound-protocol with tuning, 11 seeds and an ensemble. |
| Literature materials | 10 | Re-run with the ensemble on all 14,429 materials. |
| Natural rocks | 11 | Re-run with the ensemble on all 1,022,092 samples, plus an 8-variant ablation. |

**Substitutions, each forced by a missing file.**

1. `R3.csv` → **Dataset 4**. Dataset 4 stores the mix in raw grams, so the five mix columns are
   divided by `SCM (g)`; the authors' saved `R3.csv` rows show Ca(OH)₂ = 3.0 per gram, which is
   what this division gives. 1,849 of 1,875 rows have a usable SCM mass (the authors' `R3.csv`
   had 1,848).
2. `df_imputed.pkl` → rebuilt with the notebook's own LightGBM routine on the R3 features,
   clipped at zero. Whether the authors' pickle also held imputed *targets* is unknowable; the
   stratification bins of notebook 9 read targets from it.
3. `natural_rocks.csv` → rebuilt by merging the Gard et al. (2019) tables on their own keys
   (`sample` ⋈ `major` ⋈ `rockgroup` ⋈ `country` ⋈ `reference`), keeping all 1,022,092 samples.
   The row order matches the authors': the first rows come from `ozchem`, the last from
   Mikhalsky et al. (2006).
4. `90andupnewones.keras`, `90andupnewonesch.keras` → retrained. Every comparison against a
   value the authors left in a saved notebook cell is therefore **not testable**, not failed.
5. The extractor and classifier endpoints → **Claude Opus 5**, stated wherever it matters.

**One structural fact about Dataset 4 that shapes everything in Part 1.** The three targets are
mutually exclusive: 1,319 rows carry only heat release, 192 only Ca(OH)₂ consumption, 277 only
bound water, and 61 carry no label. There are 277 distinct material names and 28 distinct DOIs
in 1,849 rows, and 162 of those materials appear with more than one target. A validation split
that is not stratified on *which* target is labelled can therefore leave a head with almost no
validation labels — which is exactly what happens in the released notebook.



## 3. Notebook 9: the reactivity model

### 3.1 Accuracy

{{TABLE_ACCURACY}}

No row chooses anything on its test set: early stopping, the ensemble members and the tuned
configuration are selected on validation rows only. Within each split the test set is fixed, so
the seeds of one row are directly comparable and can be ensembled — "faithful" and "clean" split
with `random_state=42`, as notebook 9 cell 14 does, and the sound rows use a fixed split seed.
The two grouped rows necessarily use a different test set from the row-level rows, which is the
point of §3.4; their sizes are given there.

### 3.2 Why the released run is unstable, and what fixing it costs

The notebook stops on `val_CH_Consumption_Output_custom_nan_nrmse`: a normalized RMSE for the
Ca(OH)₂ head, averaged over validation *batches* of 50 rows, taken from the last 15 % of the
training array. With 192 Ca(OH)₂ labels among 1,849 rows, a 50-row batch usually holds none or
one. With none the metric is NaN; with exactly one the range term is zero and the metric is of
order 10⁶. {{COLLAPSE_SENTENCE}} The collapse rule is defined only on validation behaviour —
monitored value non-finite, or restored epoch ≤ 3, or monitored nRMSE > 1, which means the error
exceeds the whole range of the labels in that batch. The epoch such a run restores is
essentially the epoch at which one arbitrary Ca(OH)₂ label happened to be closest.

The sound protocol replaces that monitor with the masked validation loss evaluated over the
**whole** validation set, and it also changes three other things at once: the validation rows
are drawn at random, stratified on which target is labelled; the scaler, target standardization
and feature imputation are fitted on training rows only; and the outer split is stratified on
target availability. Together these move Ca(OH)₂ from a test R² of {{FAITHFUL42_CH_R2}}
(faithful, seed 42) to {{SOUND_CH_R2}} (ensemble). **The contribution of each change was not
isolated**: no run changes the monitor alone. The "clean" variant, which keeps the notebook's
monitor but fixes the leakage and the validation split, remains unstable ({{CLEAN_COLLAPSED}} of
11 runs collapse), which is consistent with the monitor being an important contributor but does
not prove it. §3.3 shows that tuning adds nothing on top of the combined change.

### 3.3 Tuning

The paper cites Keras Tuner but the released notebook has fixed hyperparameters and no tuner
call, so this search is mine. {{TUNE_TRIALS}} random-search trials, {{TUNE_FOLDS}}-fold
cross-validation **inside the training rows**, scored on a criterion that is fixed across trials
(mean over the three targets of the masked validation MSE in standardized units) so that tuning
the loss weights cannot game the score. The test rows are not read by the search.

Selected: {{TUNE_CFG}}. Mean best epoch across folds {{TUNE_EPOCH}}; cross-validated score
{{TUNE_SCORE}} against {{TUNE_WORST}} for the worst trial. The full trial list is in
`results/nb9_tuning_all_trials.csv`.

{{TUNE_OUTCOME}}

### 3.4 How much of the accuracy is the split

{{GROUPED_PARA}}

### 3.5 Library versions

{{TABLE_LEGACY}}

<!--pagebreak-->

## 4. Figures

![Figure 2a-c reproduced](figures/{{FIG_PARITY}})

![Figure 2d-f reproduced](figures/{{FIG_IMPORTANCE}})

{{IMPORTANCE_PARA}}

![Figure 2g-i reproduced](figures/{{FIG_SHAP}})

![Figure 3 reproduced](figures/{{FIG_LIT}})

![Figure 4a reproduced](figures/{{FIG_ROCKS}})

The paper's own figures are not redrawn or reproduced here: it is published under CC BY-NC-ND.
Each panel instead carries the paper's reported values as an annotation, and the numbers behind
every panel are in `results/fig*.csv`.



## 5. Notebook 10: the literature-extracted materials

Scored with the {{N_SEEDS}}-seed ensemble on all 14,429 materials.

{{NB10_PARA}}

{{TABLE_NB10}}

## 6. Notebook 11: natural rocks

{{NB11_PARA}}

{{TABLE_ABLATION}}

{{NB11_AMORPH}}

**Pozzolanic against hydraulic.** The released code contains no rule. The split below is mine,
and is reported only to show how sensitive the answer is to a threshold the paper does not give;
the paper reports about 46,700 pozzolanic and about 3,800 hydraulic.

{{TABLE_SWEEP}}

**Rock types.** The paper groups roughly 1,200 rock types and reports 25-38 with more than 5 %
reactive. The grouping is not documented and none was invented here: using the raw `rock_name`
values, {{N_ROCKNAMES}} distinct names occur, {{N_ROCKNAMES_20}} with at least 20 samples, of
which {{N_ROCKNAMES_5PCT}} have more than 5 % reactive. Figure 4a above compares the types the
paper names. {{FIG4_PARA}}



## 7. Part 4: running the pipeline on a few papers

Notebooks 1-8 need a private corpus and two fine-tuned OpenAI endpoints. What they can still be
tested on is the data the paper publishes: **Dataset 2** holds 200 tables with the manually
curated extraction for each, and **Dataset 1** holds what the pipeline produced, with a DOI and
a table caption. Dataset 2 has no DOI column, so the two were linked on the normalized caption:
{{N_CAPTION_MATCHED}} of the 200 Dataset 2 tables match a Dataset 1 caption with a single DOI.
Four of those were chosen, plus two tables with no Dataset 1 match. Working from the Dataset 2
table text means the extractor receives exactly the input the released pipeline receives.

The extractor was **Claude Opus 5**, with the notebook-4 prompt verbatim, at temperature 0 in
effect. The replies were written before any ground truth was read in the session; the order of
work is recorded in `results/few_papers/EXTRACTOR_NOTE.md`.

{{TABLE_FEWPAPERS}}

{{FEWPAPERS_PARA}}

**Material typing.** On a 40-row sample of Dataset 3 drawn before its labels were read, the
substitute classifier agrees with the manual top-level material type on {{D3_AGREE}} of 40
({{D3_ACC}} %), against the paper's reported 97 %. The four disagreements are label-scheme
ambiguity rather than misreadings: a limestone-like sample at 78.94 % CaO is labelled `Others`,
a clean-coal by-product is labelled `Byproduct of coal` rather than `Ash`.

**Predicted reactivity of what was extracted.** {{FEWPAPERS_PRED}}

<!--pagebreak-->

## 8. Observations that affect reproducibility

These are properties of the released material, recorded because each one changes what a
re-runner gets. None is a judgement about the science.

**Released files.** Five files the notebooks open are absent: `R3.csv`, `natural_rocks.csv`,
`df_imputed.pkl`, `90andupnewones.keras` (loaded by notebook 10) and `90andupnewonesch.keras`
(loaded by notebook 11, a different file).

**Notebook 9.**

- The stratification key is built from the *unstandardized* targets in `df_imputed` but
  `np.digitize` is applied to the *standardized* `y`, which cell 9 has already overwritten in
  place. The key has 6 classes, of sizes 9, 61, 90, 187, 192 and 1,310.
- The scaler (cell 6) and the target standardization (cell 9) are fitted on all 1,849 rows
  before the train/test split in cell 14.
- The early-stopping monitor behaves as described in §3.2.
- Validation is the last 15 % of the training array (`validation_split=0.15`), not a random draw.
- The saved notebook runs out of order: the data cells carry execution counts 1-34 while the
  model cells carry 1,044-1,052, and cell 12 uses `y_train`, which cell 14 creates. As saved,
  the notebook depends on state from an earlier session. About 1,050 executions are recorded and
  the model-selection protocol is not specified.
- `BatchNormalization` is imported and not used. Keras Tuner is cited in the paper (ref. 95);
  the released notebook contains no tuner call.

**Notebooks 10 and 11.**

- Notebook 11 scores every row, including {{N_NO_OXIDE}} rock samples with no measured oxide at
  all, whose descriptors are therefore entirely imputed. How many of those come out reactive
  depends on which retrained model is used — {{PCT_NO_OXIDE_REACTIVE}} % under the ensemble
  reported in §6, nearly all of them under a single weaker model — so the released code gives
  no stable answer for a third of the database.
- Neither a pozzolanic/hydraulic rule nor a rock-name grouping is present, although the paper
  reports both.
- The mix constants used for screening (CaCO₃ 0, water 3.6, KOH 0.151, 50 °C, 10 days) are not
  the standard R³ mix that Dataset 4 records (CaCO₃ 0.5, water 5.4, K₂SO₄ 0.108, KOH 0.0216,
  40 °C, 7 days). §6 reports what changing them does.
- The per-category `d50` assignments are written as chained assignments
  (`new_df['d50'][mask] = v`), which are silent no-ops in pandas 3.

**Notebook 5's metrics, which Supplementary Table G3 reports.** `calculate_precision_recall_f1`
returns `precision[0], recall[0], f1[0]` — the figure for the first encoded class only, not an
average over classes. Material names are label-encoded over the union of truth and prediction
and zero-padded to equal length, so "Materials - F1" is agreement on one arbitrary class.
Numeric values are flattened with missing replaced by 0 and then label-encoded, so
"Compositions - F1" is dominated by agreement on *which cells are empty*, and the companion
RMSE counts a missing value as zero. Part 4 reports both that metric, for comparability with
Table G3, and a standard set-based metric.

**The published datasets.** Six specific defects were found in the four tables examined and in
the label schemes; they are listed in Appendix C.

## 9. What would be needed from the authors

1. `R3.csv`, `natural_rocks.csv`, `df_imputed.pkl`, `90andupnewones.keras`,
   `90andupnewonesch.keras`.
2. The rule that decides whether a reactive rock is pozzolanic or hydraulic.
3. The mapping from the database's `rock_name` values to the paper's rock types.
4. The filter applied to rock samples with no measured composition, if any.
5. The train/test split seed of notebook 9, and the protocol by which a trained model was kept.
6. Confirmation of the Dataset 3 label scheme, and of the Dataset 3 / Dataset 4 row counts.

## 10. Runtime, verification and limits

{{TABLE_RUNTIME}}

**Verification.** {{VERIFY_PARA}}

**Limits of this replication.** The substitute extractor is not the authors' model, so Part 4
measures the task, not their system. The imputation of notebooks 10 and 11 is transductive by
construction — it fills the screening rows using the R³ rows — and that was kept as released.
Permutation importance and SHAP are computed on the test rows; they report the fitted model and
feed no decision. The grouped splits in §3.4 are mine: the paper does not say how its split was
formed, so the drop they show is a property of this data, not a correction to a stated number.



## Appendix A. Files behind every number

{{TABLE_FILES}}

## Appendix B. Per-seed results and the full accuracy table

{{TABLE_PERSEED}}

{{TABLE_ACCURACY_FULL}}

## Appendix C. Defects found in the published datasets

Each was found while doing Part 4 and is checkable from the files in `results/few_papers/`.

- **Dataset 2 row 191** — the material name contains an unquoted comma (`mortar SNM, 1`), so
  every data line holds 13 fields against a 12-field header. Parsed, the name is lost (pandas
  takes the first field as the row index); the oxide values stay aligned.
  (`results/few_papers/dataset2_integrity.csv`)
- **Dataset 2 row 105** — the `Winner` does not correspond to its own `Prompt`: the prompt is a
  cement-kiln-dust oxide table with 8 samples, the ground truth a single `Cement` row.
- **Dataset 3's `Category` is hierarchical free text** — 178 distinct labels and 65 distinct
  top-level strings in 318 rows, including `clinker` beside `Clinker`, not the 19 predefined
  types the Supplementary Information describes. Dataset 1's `Category` is a different, flat
  vocabulary of 21 names.
- **Row counts** — the Supplementary Information describes Dataset 3 as 200 material
  descriptions (the file has 318 rows) and Dataset 4 as 318 materials and 1,850 data points
  (the file has 1,875 rows, 1,849 of them with a usable SCM mass).
- **Statistics extracted as materials** — `Mean` and an unnamed material appear as materials in
  Dataset 1 for the kiln-dust table; and the 14 distinct cements of another table were collapsed
  into 5 EN 197-1 designations, six rows sharing one composition descriptor.
- **A transcription error that propagated** — for mortar sample SNM 7 the source table gives
  Na₂O 1.87 and MgO 1.57; both the Dataset 2 manual ground truth and the Dataset 1 row record
  MgO as 1.87. Three of the 16 mortar samples present in the Dataset 2 ground truth are also
  absent from Dataset 1.

## Appendix D. The rock ablation in full

Both row filters and the single-model rows, for every imputation variant:

{{TABLE_ABLATION_FULL}}
