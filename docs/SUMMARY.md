# Summary: what was done, what was not, and why

**Task.** Run the code behind Mahjoubi et al. (Communications Materials 6:99, 2025), get its outputs, and replicate the results on a few papers. The 5–6 page report is `REPLICATION_REPORT_FINAL.docx` (PDF copy alongside); a longer technical report is `REPLICATION_REPORT_TECHNICAL.pdf`. This page is the short version.

## What was done, and why

| Part | What | Why |
|---|---|---|
| Notebook 9 (reactivity model) | Ran it as released (11 seeds), then under a modified protocol: 5-fold CV tuning on training rows only, 11 seeds, ensemble, bootstrap CIs, grouped splits | A single released run is unreliable, so the comparison was needed to know what the code can do |
| Notebooks 10 and 11 | Re-ran both with the ensemble; 8-variant, one-factor-at-a-time ablation on 1,022,092 rock samples | An earlier comparison used a weak model, which confounded it |
| Figures | Parity, permutation importance, SHAP, literature scatter, rock types | To compare with the paper's figures (CC BY-NC-ND, so not redrawn) |
| Notebooks 1–8 proxy | Extracted 4 composition tables (+2 non-composition) with Claude Opus 5 instead of the authors' fine-tuned GPT-3.5; scored against Datasets 1, 2 and 3 | The private corpus and the authors' endpoints are unavailable |

## Main results

- **Released notebook 9 is unstable.** Its early-stopping monitor averages over 50-row validation batches that often hold 0 or 1 Ca(OH)₂ label. All 11 faithful runs hit a collapse rule defined on validation behaviour, and Ca(OH)₂ test R² was 0.087 ± 0.130.
- **Modified protocol.** The ensemble reaches test R² of 0.947 (heat release; paper 0.93), 0.764 (Ca(OH)₂; paper 0.85) and 0.830 (bound water; paper 0.89). The last two rest on 38 and 56 test labels, and their 95% intervals include the published values; that shows the test sets are too small to separate the two, not that they are equivalent.
- **What caused the improvement is not isolated.** The protocol changes four things at once (whole-set monitor, random stratified validation split, train-only preprocessing, stratified outer split). A "clean" variant that keeps the original monitor stays unstable (7 of 11 runs collapse), which is consistent with the monitor mattering but does not prove it.
- **Tuning did not help.** The notebook's own hyperparameters scored slightly higher on test (0.956 / 0.776 / 0.844). The cross-validation-selected configuration was kept; switching to the better test score would be selection on the test set.
- **Grouped split.** With no material shared between train and test, Ca(OH)₂ R² is −0.061 and heat release 0.645, while training fit stays above 0.95. Row-level accuracy therefore mostly measures interpolation between ages of materials already seen.
- **Rock screening.** The released notebook gives 141,548 reactive samples against the paper's 50,569 (+180%), varying from 92,639 to 462,245 across seeds. Adding Dataset 5 and using total iron gives 64,239 (+27%); the R³ standard mix gives about 570,000. No combination matches, so no single factor explains the gap.
- **Few papers (substitute model).** Relevance was judged correctly on all 6 tables. 240 of 241 oxide cells from **3** tables agree exactly with the manual ground truth; the fourth composition table (Dataset 2 row 105) was excluded because its ground truth belongs to a different table. Against the authors' Dataset 1 rows, 192 of 193 comparable cells agree (25 of 43 extracted materials could be paired by name). Classification agreed on 36 of 40 unseen rows (90%) against the paper's 97%, but on a different label scheme. The report follows one material (HL1, a high-lime fly ash). Provenance: table text and manual values from Dataset 2; DOI, category and the authors output from Dataset 1; title and journal from Crossref. The report author confirmed the table appears in Tokyay (1999) end to end: extracted oxides, comparison with both reference datasets, imputed inputs and predictions.
- **Verification.** A clean-start re-run reproduced six checked outputs identically; the full 48-row rock ablation was regenerated after being overwritten and also matched the first run.

## What was not done, and why

| Not done | Why |
|---|---|
| Notebooks 1–3, 6 | They need a private MongoDB corpus |
| The authors' fine-tuned models | They sit on the authors' OpenAI account; a substitute is labelled everywhere, so Part 4 measures the task, not their system |
| Matching the authors' saved outputs (964 materials, 50,569 rocks, row values) | Needs `R3.csv`, `natural_rocks.csv`, `df_imputed.pkl` and two `.keras` files, none released; marked "not testable" |
| Python 3.8 | Only 3.11 is installed; an older library stack on 3.11 was used instead and the report says it is not a 3.8 test |
| A second run of everything | The re-run covered headline outputs; not the grouped or sensitivity runs, the tuning search, the figures or the legacy run |
| SHAP on all models and rows | KernelExplainer is slow: 3 models, 60 test rows, as the caption states |

## Mistakes found and corrected

- The first extraction summary said 156 of 157 cells "from four tables". They came from only two tables because a name-matching rule was too strict; after fixing it the figure is 240 of 241 cells from three tables.
- The verification script overwrote the full rock ablation with a base-only file. It was regenerated and matched the first run, and the script was changed.
- Earlier text credited the early-stopping monitor alone for the improvement. That was not isolated and has been reworded.
- A wrong F1 count, a wrong row-filter statement and a SHAP caption naming the wrong explainer were fixed.
- The report mixed two network configurations without saying so: the faithful runs use the released 256–256–16 network and the tuned ensemble uses 512–256–32. Each table row and the workflow figure now state which applies.
- The rock ablation table showed 5 of the 8 variants; it now shows all 8.
- The two extraction comparisons (240/241 against the manual ground truth, 192/193 against the authors' output) cover different tables and are now labelled and explained as such; an independent recount from the raw records agrees with both.

## Caveats

- Ca(OH)₂ and bound-water test sets are small, so single R² values are noisy.
- The grouped splits are my own addition; the paper does not say how its split was formed.
- The substitute extractor ran interactively, so no timing comparison is made.
