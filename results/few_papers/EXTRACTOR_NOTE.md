# How the extraction in `responses/` was produced

**Substitute model.** The authors' table extractor is a fine-tuned GPT-3.5 endpoint on their own
OpenAI account (`ft:gpt-3.5-turbo-1106:personal::8faiLnzJ`, notebook 4) and their material-type
classifier is a second one (`ft:gpt-3.5-turbo-1106:personal::8kzjZwlt`, notebook 8). Neither is
reachable, and no OpenAI key is available here. **Claude Opus 5 (`claude-opus-5`) was used in
their place.** Every number in Part 4 of the report therefore measures *a* current LLM on the
authors' task, not their model, and no claim is made about reproducing their model's behaviour.

**Input.** Exactly the notebook-4 prompt, reproduced verbatim in `selection.json`
(`prompt_system`, `prompt_user_template`), filled with the table text from Dataset 2. Working
from Dataset 2 means the extractor sees the same input the released pipeline sees; the paper
corpus it would otherwise come from is a private MongoDB.

**Order of work, so that the test is not circular.**

1. `few_papers_select.py` wrote `prompts/d2rowNNN.txt`. It deliberately does not read, print or
   write the Dataset 2 `Winner` column or any Dataset 1 oxide value.
2. The prompts were read and the replies written to `responses/d2rowNNN.csv`. At that point no
   ground truth for these tables had been read in the session.
3. Only then did `few_papers_score.py` open the `Winner` column and the Dataset 1 rows.

The same order was kept for the classification step: `my_categories.csv` and
`d3_sample2_my_labels.csv` were written from the notebook-8 prompt inputs (material name, table
caption, descriptor) and the label vocabulary, before the corresponding `Category` column was
read. `d3_sample_inputs.csv` (seed 0) was scored first against the Dataset 1 flat categories;
once its labels had been seen, a second, disjoint sample (`d3_sample2_inputs.csv`, seed 1) was
drawn for the clean top-level-type measurement reported in the report.

**Timing.** The paper's Table G3 reports an average end-to-end time of 6.87 s per table for the
fine-tuned GPT-3.5. No comparable timing is reported here: the replies were produced inside an
interactive session rather than by a timed API loop, so any number would not be comparable.
