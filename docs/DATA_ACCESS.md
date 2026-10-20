# Data access and publication boundary

This is a code-only release. It deliberately excludes raw clinical tables,
patient identifiers, discharge notes, generated narratives, grading CSVs,
spreadsheets, private prompts/documents, backups, and model artifacts.

Obtain the required MIMIC-IV and MIMIC-IV-Note data through their authorized
access process. The upstream diagnosis table expected by the extraction scripts
is a separate prerequisite; this repository neither downloads nor reconstructs
that table. Any ontology resources also require an appropriate license.

Before model generation or judging:

1. Confirm that the source data agreement and your institution permit the
   proposed processing and any external API transfer.
2. Use only an approved provider, account, environment, and retention policy.
3. Keep input, intermediate, output, logs, and backup files in protected storage.
4. Inspect the command and destination. Generation and evaluation can make paid
   API calls, use parallel workers, and save patient-level data.
5. Do not publish derived records merely because identifiers have been removed.

The scripts' `.bak` and `.tmp` files are sensitive too. `.gitignore` is a
convenience, not a comprehensive data-loss-prevention system. Inspect any
proposed commit before publication.

Use generic synthetic fixtures for software tests. Do not paste real records,
keys, or generated patient narratives into public issues.
