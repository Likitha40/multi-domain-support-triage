# Multi-Domain Support Triage Agent

This is a terminal-based support triage agent for HackerRank, Claude, and Visa tickets. It uses only local files as its support corpus and does not call external services.

## Expected files

- `sample_support_tickets.csv` for examples and optional evaluation
- `support_tickets.csv` for the unlabeled challenge input
- support corpus files under `corpus/`, `support_corpus/`, `docs/`, or `knowledge_base/`

Corpus files can be `.txt`, `.md`, `.html`, `.htm`, or `.csv`. Put company names such as `hackerrank`, `claude`, or `visa` in the file path when possible so retrieval can route documents to the right ecosystem.

## Run

```powershell
python triage_agent.py --input support_tickets.csv --output triage_output.csv
```

To pass corpus locations explicitly:

```powershell
python triage_agent.py --corpus .\corpus --input support_tickets.csv --output triage_output.csv
```

To sanity-check against the sample labels:

```powershell
python triage_agent.py --eval --sample sample_support_tickets.csv --input support_tickets.csv
```

## Output

The output CSV contains exactly:

- `status`
- `product_area`
- `response`
- `justification`
- `request_type`

The agent escalates sensitive account, billing, fraud, card, access, and unsupported cases, and replies only when it has enough grounding from the local corpus or the request is clearly out of scope.
