# Persona-Conditioned Patient Self-Reports

Research code for **Persona-Conditioned Generation of Patient Self-Reports from EHRs**, by **Yuexin Wu, Jianming Wei, Shiqi Wang, and Vasile Rus**, published at **LREC 2026**.

**[Paper (PDF)](https://aclanthology.org/2026.lrec-1.845.pdf)** · [ACL Anthology](https://aclanthology.org/2026.lrec-1.845/) · [Citation](#citation)

## Overview

This project studies how language models can transform clinical records into first-person, patient-style narratives while conditioning the wording on a patient persona. It combines EHR preprocessing, structured persona construction, report generation, and evaluation of clinical correctness, source faithfulness, persona consistency, and realism.

The repository contains the project's preprocessing, GPT-based generation, GPT/Claude judging, pairwise comparison, and analysis scripts. The generated narratives are **synthetic reconstructions from EHRs**, not reports collected directly from patients. This is research software, not a clinical tool.

**Code only:** no MIMIC records, patient-level generated text, identifiers, API credentials, model weights, or private results are distributed here. Access to the underlying data and permission to process them with any model provider must be obtained separately.

## Repository layout

```text
preprocessing/  education attributes, persona joins, EHR extraction, sampling
generation/    separate GPT-4o-mini and GPT-5-mini generation entrypoints
evaluation/    single-report rubric scoring and randomized pairwise judging
analysis/      agreement, score distributions, linguistic and subgroup analyses
tools/         explicit grading-input preparation and a legacy merge utility
legacy/        alternate extraction routes and earlier experimental scripts
tests/         offline tests using synthetic fixtures only
docs/          workflow, data requirements, code map, and release notes
```

See [the workflow](docs/WORKFLOW.md) for commands and schemas, [the code map](docs/CODE_MAP.md) for script roles, and [release notes](docs/RELEASE_NOTES.md) for implementation caveats.

## Installation and safe checks

Use Python 3.10 or later in an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Optional plotting / linguistic analysis dependencies:
python -m pip install -r requirements-analysis.txt
```

These dependency lists are installation guidance, **not a recovered lockfile from the paper's experiments**. Provider model availability and SDK compatibility must be checked before a new run.

Run the offline checks without clinical data or credentials:

```bash
python tools/check_release.py
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

Use environment variables `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` for credentials; never put real keys in scripts or commits. No `.env` loader is included. Avoid passing keys on the command line, where shell history or process listings may expose them.

## Workflow

1. Prepare authorized MIMIC-IV patient/admission/DRG tables and MIMIC-IV-Note discharge records locally.
2. Generate **synthetic** education attributes and merge the persona fields.
3. Join the separately prepared diagnosis table with personas and discharge text; apply the retained extraction rules.
4. Select records using the existing disease/persona-coverage sampling routine.
5. Generate reports with `generation/report_generation2.py` (GPT-4o-mini) or `generation/report_generation.py` (GPT-5-mini).
6. Explicitly choose the generated report column for single-report grading, or prepare a working copy containing both columns for pairwise judging.
7. Analyze the grading outputs.

**Important:** generation writes `report_gpt4o_mini` or `report_gpt5_mini`, while the single-report judges read `report`. Use the new release utility below; do not accidentally grade the upstream diagnosis-table `report`:

```bash
python tools/prepare_evaluation_input.py \
  --input outputs/generated_gpt4o.csv \
  --output outputs/grading_input_gpt4o.csv \
  --report-column report_gpt4o_mini
```

The utility preserves an existing `report` as `source_report`, refuses overwrites and ambiguous inputs, and does not generate or change narrative text. Full examples are in [WORKFLOW.md](docs/WORKFLOW.md).

## Data access and responsible use

Read [DATA_ACCESS.md](docs/DATA_ACCESS.md) before running any data or API workflow. Credentials for an API do **not** by themselves authorize sending clinical records to that provider. Comply with the source data agreement, institutional approvals, provider terms, and approved data-handling environment.

Education is sampled by the code, not measured in the EHR. Header truncation and prompt instructions do not establish that every retained statement was patient-reported. Outputs may contain unsupported claims or stereotypes and require appropriate review.

The `legacy/` directory is **not part of the recommended workflow**. In particular, the old grader omits the candidate narrative, and the negative-symptom heuristic can invent denials from diagnosis cues. See the code map before using these historical scripts.

## Reproducibility status

This release organizes the retained research code and documents its interfaces. It preserves generation prompts and research scoring logic, with credential removal, portable paths, import guards, a proxy-keyword compatibility correction, and offline tooling.

The source folder did not contain a frozen environment or a run manifest linking every script to the published tables. Consequently, this repository does not claim bit-for-bit reproduction of the paper's results. No generation, API evaluation, or patient-data experiment was rerun while preparing this release. See [release notes](docs/RELEASE_NOTES.md) for known limitations and validation scope.

## Citation

The author list below follows the linked published PDF.

```bibtex
@inproceedings{wu-etal-2026-persona,
  title = {Persona-Conditioned Generation of Patient Self-Reports from EHRs},
  author = {Wu, Yuexin and Wei, Jianming and Wang, Shiqi and Rus, Vasile},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference},
  month = may,
  year = {2026},
  address = {Palma de Mallorca, Spain},
  publisher = {ELRA Language Resource Association},
  pages = {10792--10801},
  doi = {10.63317/34prj6o9qfrk},
  url = {https://aclanthology.org/2026.lrec-1.845/}
}
```

## License

No software license was present in the retained source. This release does not assign a new license or grant rights to the underlying datasets. See [LICENSE_STATUS.md](LICENSE_STATUS.md).
