# Shows I Saw Ingestion Workflow

This folder contains the review-only ingestion tools for the `Shows I Saw` archive.

Core rule: AI proposes. Human approves.

The ingestion script never updates the live site automatically. It only creates review artifacts for you to inspect and manually apply.

## What the tool does

`ingest_ticket.py` will:

- let you choose a scan from `archive-ingest/incoming/`
- run OCR on the ticket image
- suggest structured ticket metadata
- prompt you to confirm or correct the important fields
- generate a draft JSON review package in `archive-ingest/review/draft-json/`
- generate a human-readable notes file in `archive-ingest/review/notes/`
- generate a draft share page in `archive-ingest/review/share-pages/`
- include provenance/confidence labels for ticket fields
- include research-enrichment placeholders that can plug into real research later
- generate a paste-ready ticket object for `tickets.js`

It does not:

- modify `tickets.js`
- copy files into the live site
- overwrite existing review artifacts on purpose
- deploy anything

## Requirements

You mentioned these tools are part of your local setup:

- Python
- Tesseract OCR
- `pytesseract`
- OpenCV
- Pillow
- NumPy

The script currently expects Tesseract here:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

If your Tesseract install lives somewhere else, update the path near the top of `ingest_ticket.py`.

## Folder flow

1. Put a new scan in `archive-ingest/incoming/`.
2. Run the ingest command from the `shows-ive-seen/` folder.
3. Review the generated outputs.
4. Manually decide what to publish.

## Exact command to run

From the `shows-ive-seen/` folder:

```powershell
python .\archive-ingest\ingest_ticket.py
```

If your machine uses the Python launcher instead:

```powershell
py .\archive-ingest\ingest_ticket.py
```

## What gets generated

For each reviewed scan, the script creates artifacts in the existing review folders:

- `archive-ingest/review/draft-json/<slug>.json`
- `archive-ingest/review/notes/<slug>.txt`
- `archive-ingest/review/share-pages/<slug>.html`

The JSON review package now includes:

- `ticket`: the proposed ticket data
- `provenance`: field-by-field confidence/provenance labels
- `research`: structured placeholder suggestions for later research integration
- `ticketObjectForTicketsJs`: a paste-ready object block for manual use in `tickets.js`

## Confidence labels

The draft outputs use three labels:

- `confirmed_from_ticket`
- `inferred_from_research`
- `generated_suggestion`

These labels are there to help you review what came directly from the ticket, what should eventually be confirmed with outside research, and what is only a generated draft suggestion.

## Review checklist

Before publishing anything manually:

- confirm artist billing and opener roles
- confirm venue naming and exact date
- confirm any setlist references or tour context before using them
- verify editorial copy and tags
- manually paste the approved object into `tickets.js`
- manually copy the approved share page into `share/<slug>/index.html`
- manually place approved image assets in the live site root
- test locally before deployment

## Future research plug-in

`research_enrichment.py` is intentionally built as a placeholder-friendly module.

Right now it creates structured research suggestions without live web lookup. Later, a real research provider can be plugged into that module without changing the review artifact contract.
