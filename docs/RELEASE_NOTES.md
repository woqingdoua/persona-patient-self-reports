# Code release notes — 2026-10-20

## Scope

This release reorganizes retained research scripts into stage directories.
Original filenames are preserved (including `diff_distirbution.py`); the
original source directory is not modified.

Publication-only changes:

- Replace embedded credential defaults with provider environment variables.
- Replace workstation-specific absolute paths with relative placeholders.
- Remove preprocessing dataframe previews that print patient-level content.
- Place previously top-level workflows behind explicit script entrypoints.
- Use `httpx.Client(proxy=...)` in place of the deprecated plural argument.
- Add project documentation, citation metadata, dependency lists, `.gitignore`,
  a release checker, and synthetic tests.
- Add an explicit, non-generative grading-input adapter that preserves the
  upstream `report` while selecting the intended generated narrative.

Prompts, research scoring formulas, model defaults, and sampling choices are
not retuned. No training, generation, provider API call, or clinical-data
evaluation was run to prepare this release.

## Validation scope

The offline checker compiles Python source without importing workflows,
checks for accidental secret/private-path patterns and disallowed data
artifacts, and validates local Markdown links. Unit tests use wholly synthetic
CSV fixtures. These checks do not establish clinical validity or exact
reproduction of published results.

No frozen dependency lockfile or run manifest connecting each retained script
to the published tables was available. Model aliases, stochastic remote
generation, retries, and provider behavior can change results. Analysis scripts
may require manual path/schema adaptation and optional linguistic resources.

## Retained implementation caveats

- Synthetic education is not a measured patient attribute. Adult age-bin key
  fallbacks and zero-concentration minor-age priors are documented in the workflow.
- Header-based EHR trimming and instructions in prompts do not prove that every
  retained or generated statement came from patient-reported evidence.
- Single-report judges trust model-returned scores and weighted overall.
- Pairwise `tie` also covers some missing/failed/unparseable cases.
- Some scripts overwrite inputs by default, and resume logic assumes compatible
  files. Use new, explicit output paths and keep protected backups.
- The retained single-report judges read the output working file on a fresh
  run. The workflow explicitly prepares a copy and grades it in place, instead
  of passing a nonexistent output path.
- `analysis/consistency2.py` contains a compatibility branch involving the older
  SciPy `cumtrapz` name; it is not a promise of a tested scientific environment.
- Legacy scripts are separated, not promoted to a validated replacement pipeline.
  In particular, the earlier grader omits the candidate report, and the negative
  symptom heuristic can invent denials from diagnosis cues.

## Citation provenance

The linked published PDF lists Yuexin Wu, Jianming Wei, Shiqi Wang, and Vasile
Rus. The README and citation file follow that PDF's author list.

## Excluded material

Raw and derived clinical data, generated patient narratives, private grading
results and plots, spreadsheets/documents, backups, credentials, and scratch
files are not in this release. Data access and software licensing are separate;
see `DATA_ACCESS.md` and the root `LICENSE_STATUS.md`.
