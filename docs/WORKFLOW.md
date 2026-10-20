# Workflow and input schemas

These commands describe interfaces in the retained scripts, not a verified
historical command log. They are **not** run by installation or the offline
tests. Read [DATA_ACCESS.md](DATA_ACCESS.md) first.

Use authorized data in a protected working directory. The following names are
placeholders; no clinical files are provided. Create `outputs/` before running.
Use separate outputs and working copies: several scripts retain in-place
defaults or resume files.

## 1. Education and personas

```bash
mkdir -p outputs
python preprocessing/education_generation.py \
  --input data/patients.csv.gz --output outputs/patients_education.csv \
  --seed 2025 --alpha 300
python preprocessing/persona_collection.py \
  --admissions data/admissions.csv.gz --drgcodes data/drgcodes.csv.gz \
  --patients_edu outputs/patients_education.csv --output outputs/persona.csv
```

The patient table needs `subject_id`, `gender`, `anchor_age`.
Education is **synthetically sampled**. Admission/persona fields include
`subject_id`, `hadm_id`, `marital_status`, `insurance`, `language`,
`admission_location`, `admission_type`, `drg_severity`, `gender`, `anchor_age`,
and `education_level`.

Known retained edge case: two adult age-bin names in `choose_age_bin` do not
match `AGE_PRIOR` keys and therefore fall back to `BASE_PRIOR`. Dirichlet
sampling also requires positive concentrations; the retained zero-valued
minor-age priors are not safe with noise enabled. MIMIC adult-cohort assumptions
must be checked rather than extending this script to children unchanged.
These scientific choices have not been silently modified in the release.

## 2. EHR extraction and sampling

```bash
python preprocessing/EHR_extraction.py \
  --train_csv data/diagnoses.csv \
  --persona_csv outputs/persona.csv \
  --discharge data/discharge.csv.gz \
  --output outputs/ehr_prepared.csv
python preprocessing/sampling_data.py \
  --input outputs/ehr_prepared.csv --output outputs/ehr_subset.csv \
  --n 10000 --seed 42
```

`data/diagnoses.csv` is an **external prerequisite** with
`subject_id, hadm_id, report, icd11_title`. The discharge file needs
`subject_id, hadm_id, text` (the code also recognizes documented aliases).
The extraction output is
`subject_id, hadm_id, icd11_title, report, persona, text`.

The main extraction route selects the longest note per admission, truncates
before specified examination/results headers, removes the part before the
Chief Complaint heading when found, and keeps up to **700 whitespace-delimited
words**. This is not an LLM tokenizer limit and is not a validated guarantee
that the retained text contains only patient-stated evidence.

The sampling implementation seeks disease/persona coverage greedily. Do not
interpret it as a proven fixed-quota stratification algorithm.

## 3. Generate narratives

Set `OPENAI_API_KEY` securely in the shell. The next commands send clinical
content to the configured provider and incur API costs; run them only with
the appropriate authorizations.

```bash
python generation/report_generation2.py \
  --input outputs/ehr_subset.csv --output outputs/generated_gpt4o.csv \
  --model gpt-4o-mini
python generation/report_generation.py \
  --input outputs/ehr_subset.csv --output outputs/generated_gpt5.csv \
  --model gpt-5-mini
```

Both read `persona` and `text` and preserve the input columns. The generated
columns are respectively `report_gpt4o_mini` and `report_gpt5_mini`.
Default worker count is 8. Use `--help` to inspect checkpoint/resume/limit
options; never reuse a checkpoint from another input or model.

Provider identifiers here are the retained script defaults, not a promise
that each model remains available. No model replacement is performed
automatically by this release.

## 4. Explicitly prepare and grade a report

```bash
python tools/prepare_evaluation_input.py \
  --input outputs/generated_gpt4o.csv \
  --output outputs/grading_input_gpt4o.csv \
  --report-column report_gpt4o_mini
cp outputs/grading_input_gpt4o.csv outputs/gpt4o_judged_by_gpt.csv
python evaluation/evaluate_single.py \
  --input outputs/gpt4o_judged_by_gpt.csv --model gpt-4.1
# Requires ANTHROPIC_API_KEY and authorized processing:
cp outputs/grading_input_gpt4o.csv outputs/gpt4o_judged_by_claude.csv
python evaluation/evaluation_single_claude.py \
  --input outputs/gpt4o_judged_by_claude.csv \
  --model claude-3-7-sonnet-latest
```

The retained single-report judges read their output working file on a fresh
run. Therefore, the examples first create a copy and then grade it in place;
do not pass a nonexistent path to `--output`. Only make these copies for a
fresh run: do not overwrite existing grading results or resume files.
Repeat with `report_gpt5_mini` and distinct paths if grading the second
generator. The adapter refuses an existing output or `source_report` collision,
missing required fields, and empty selected narratives. It does not silently
drop rows or impute failed generations.

The judges read `persona`, `text`, and **`report`**. They produce
`ClinicalCorrectness`, `Faithfulness`, `Persona`, `Realism`, and `overall`;
Claude also retains a `score_error` field. The prompts specify an overall
weighting of 0.40/0.30/0.20/0.10 in that order. The retained implementation
trusts the returned weighted overall rather than independently recomputing
or fully range-validating it.

## 5. Pairwise comparison

Prepare a private **working copy** containing both generated columns, matched
by the same admission identifiers. The existing merge helper overwrites
`--base`, so first copy it deliberately:

```bash
cp outputs/generated_gpt4o.csv outputs/pairwise_working.csv
python tools/other.py \
  --base outputs/pairwise_working.csv --src outputs/generated_gpt5.csv
python evaluation/compare.py \
  --input outputs/pairwise_working.csv --model gpt-4.1 --seed 42
python evaluation/compare2.py \
  --input outputs/pairwise_working.csv \
  --model claude-3-7-sonnet-latest --seed 42
```

The comparison scripts write to the supplied working CSV. They randomize
A/B presentation and write `cmp_winner` / `cmp_winner2`. In the retained
implementation, `tie` can also encode missing reports, API failures, or
unparseable responses; it must not automatically be interpreted as a genuine
judge preference tie.

Keep all working copies and analysis exports private. The analysis directory
contains study-specific filenames and column assumptions; adapt paths
explicitly and record any changes when conducting a new experiment.
