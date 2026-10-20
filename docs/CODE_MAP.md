# Code map

Filenames are retained to make the release traceable to the working scripts;
directory grouping is new. Run commands from the repository root. Relative
data paths remain relative to the current working directory.

| Area | Files | Purpose |
|---|---|---|
| Preprocessing | `education_generation.py` | Sample synthetic education attributes from age-conditioned priors |
| Preprocessing | `persona_collection.py` | Merge admissions, APR DRG severity, and patient attributes |
| Preprocessing | `EHR_extraction.py` | Join diagnosis/persona records with truncated discharge text |
| Preprocessing | `sampling_data.py` | Greedy disease/persona coverage sampling |
| Generation | `report_generation2.py` | GPT-4o-mini report generation; writes `report_gpt4o_mini` |
| Generation | `report_generation.py` | GPT-5-mini generation with compatibility fallback paths; writes `report_gpt5_mini` |
| Evaluation | `evaluate_single.py` | GPT single-report rubric scoring |
| Evaluation | `evaluation_single_claude.py` | Claude single-report rubric scoring |
| Evaluation | `compare.py`, `compare2.py` | GPT/Claude pairwise comparison with randomized A/B presentation |
| Tools | `prepare_evaluation_input.py` | New, explicit generated-column-to-`report` bridge for single-report graders |
| Tools | `other.py` | Existing identifier-based merge of a generated column; overwrites `--base`, so use a copy |
| Analysis | `overall_grade1.py`, `overall_grade2.py` | Aggregate grading analyses |
| Analysis | `consistency.py`, `consistency2.py`, `paired_tests.py` | Agreement and paired statistical analyses |
| Analysis | `grade2_2.py`, `group_graph.py`, `group2.py` | Persona-group analyses/figures (`group2.py` came from the original `grade/` folder) |
| Analysis | `score_distribution.py`, `score_distribution_small.py`, `diff_distirbution.py` | Score-distribution figures; spelling of historical filename retained |
| Analysis | `case_study.py`, `sentence_pref_analysis.py`, `stat.py` | Linguistic, readability, and preference analyses |

Analysis scripts retain study-specific column/file assumptions. Some use
editable constants rather than complete CLIs. Review those constants before
execution; the release does not ship their input files. The linguistic script
requires spaCy's English model and NLTK resources; its `main` workflow may fetch
NLTK resources when explicitly run.

## Legacy code: not the recommended pipeline

| File | Why separate |
|---|---|
| `persona_ehr.py` | Alternate extraction route, without the main route's complete trimming procedure |
| `ehr_extraction_traindata.py` | Alternate balanced-sampling route; not interchangeable with `sampling_data.py` |
| `chatgpt_report_generation.py` | Earlier limited-row generation workflow and different prompt |
| `chatgpt_report_grade.py` | Earlier, incomplete grader: does not supply the candidate report; not a valid replacement for the main judges |
| `negative_symptom_generation.py` | Heuristic denials inferred from diagnosis/specialty cues, not verified patient statements; overwrites its input |

Empty `oo.py`, scratch `tt.py`, private reconstruction notes, and all non-code
patient/result files are intentionally excluded.
