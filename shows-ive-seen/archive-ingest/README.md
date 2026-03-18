# Shows I Saw Ingestion Workflow

This folder contains the review-only ingestion tools for the `Shows I Saw` archive.

Core rule: AI proposes. Human approves.

The tools in this folder never update the live site automatically. They only create review artifacts for you to inspect and manually apply.

## Tools in this folder

### Single-ticket prototype

`process_ticket.py` is the current simplified prototype for already-cropped single-ticket JPGs.

It will:

- accept one cropped ticket image
- run deterministic cleanup for readability and OCR
- normalize paper background without redrawing text
- run OCR on cleaned variants
- propose artist, venue, year, and exact date when possible
- generate the canonical slug format `artist-venue-year`
- ask before committing canonical filenames
- save low-confidence cases into `review/needs-review/`
- write a draft JSON package instead of touching the live site

### Batch orchestration

`ingest_scan_batch.py` will:

- scan `archive-ingest/incoming/` for supported source images
- run the existing split workflow for each scan
- run the existing image-processing workflow for each split candidate
- run the metadata draft workflow for each processed archive candidate
- write a structured manifest for each ticket candidate
- write a batch summary log for the entire run
- keep every output inside `archive-ingest/`
- never update `tickets.js`, never copy files into the live site, and never publish automatically

### Multi-ticket scan splitting

`split_ticket_scan.py` will:

- let you choose a scan from `archive-ingest/incoming/`
- detect multiple likely ticket regions conservatively
- prefer fewer broader split candidates over fragmenting a single ticket
- use broad first-stage region detection first
- attempt a second-stage internal split only when a broad region looks oversized and internal separator bands are clear enough after density smoothing
- export each detected region as a separate working image in `archive-ingest/working/split/`
- preserve padding around each split candidate
- generate a preview overlay with numbered ticket boxes
- generate a contact sheet for review
- generate a notes file summarizing first-stage and final detected regions, bounds, confidence, and ambiguity

This stable splitter is best suited for clearly separated tickets and simple vertical stacks. It can also handle some simple horizontal arrangements when internal gaps are obvious.

Final splitter architecture:

- Stage 1 - Broad region detection
  Conservative first-pass detection finds likely full-ticket groupings and prefers broad safe regions over aggressive fragmentation.
- Stage 2 - Internal gap segmentation
  Oversized broad regions are checked for smoothed internal separator bands so simple stacked tickets can split cleanly.
- Validation - Reject fragments
  Candidate child regions must pass minimum height, minimum area ratio, reasonable ticket-like aspect ratio, and non-fragment checks.
- Acceptance - Keep all valid ticket-sized children
  If two or more candidate child regions pass validation, the splitter keeps all valid children instead of collapsing them because one separator is weaker than another.

The splitter intentionally prefers "fewer valid regions over fragmented tickets."

### Image processing

`process_ticket_image.py` will:

- let you choose a source image from `incoming/`, `originals/`, or existing working files
- detect ticket bounds using a conservative combination of contour and mask-based image analysis
- auto-straighten when the detected contour is strong enough to rotate safely
- crop with padding so the ticket is not trimmed flush to the edge
- apply subtle enhancement to improve readability without redrawing text
- generate an archive image candidate that feels closer to a ready-to-review archive object
- generate a 1200x630 share image candidate with a cleaner editorial presentation
- generate a detection preview image
- generate a contact sheet for human review
- generate an image review notes file that explains the detection and framing decisions

### Metadata ingest

`ingest_ticket.py` will:

- let you choose a scan from `archive-ingest/incoming/`
- run OCR on the ticket image
- suggest structured ticket metadata
- prompt you to confirm or correct the important fields
- run a validation checkpoint before slug generation
- generate a draft JSON review package in `archive-ingest/review/draft-json/`
- generate a human-readable notes file in `archive-ingest/review/notes/`
- generate a draft share page in `archive-ingest/review/share-pages/`
- include provenance/confidence labels for ticket fields
- include research-enrichment placeholders that can plug into real research later
- generate a paste-ready ticket object for `tickets.js`

The new `process_ticket.py` prototype is the preferred starting point for already-cropped single-ticket images because it combines cleanup, OCR, slug proposal, and review-state saving in one step.

## What the tools do not do

These tools do not:

- modify `tickets.js`
- copy files into the live site
- overwrite originals
- publish automatically
- invent missing printed details
- redraw text or substitute fonts

## Requirements

You mentioned these tools are part of your local setup:

- Python
- Tesseract OCR
- `pytesseract`
- OpenCV
- Pillow
- NumPy

The metadata script currently expects Tesseract here:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

If your Tesseract install lives somewhere else, update the path near the top of `ingest_ticket.py`.

## Recommended folder flow

### For a multi-ticket scan

1. Put the scan in `archive-ingest/incoming/`.
2. Run the batch controller or the splitter manually.
3. Review the numbered split candidates, preview overlay, contact sheet, and split notes.
4. Review each processed archive/share candidate, metadata draft, and manifest.
5. Manually decide what to publish.

### For a single-ticket scan

1. Put the scan in `archive-ingest/incoming/`.
2. Run the image processing step first.
3. Review the archive candidate, share candidate, preview overlay, contact sheet, and notes.
4. Run the metadata ingest step.
5. Review the draft JSON, notes, and share page draft.
6. Manually decide what to publish.

## Exact commands to run

From the `shows-ive-seen/` folder:

### Batch controller

```powershell
python .\archive-ingest\ingest_scan_batch.py
```

If your machine uses the Python launcher instead:

```powershell
py .\archive-ingest\ingest_scan_batch.py
```

### Multi-ticket splitter

```powershell
python .\archive-ingest\split_ticket_scan.py
```

If your machine uses the Python launcher instead:

```powershell
py .\archive-ingest\split_ticket_scan.py
```

### Image processing step

```powershell
python .\archive-ingest\process_ticket_image.py
```

If your machine uses the Python launcher instead:

```powershell
py .\archive-ingest\process_ticket_image.py
```

### Metadata ingest step

```powershell
python .\archive-ingest\ingest_ticket.py
```

If your machine uses the Python launcher instead:

```powershell
py .\archive-ingest\ingest_ticket.py
```

### Simplified single-ticket prototype

```powershell
python .\archive-ingest\process_ticket.py .\archive-ingest\incoming\my-ticket.jpg
```

Or choose from `incoming/` interactively:

```powershell
python .\archive-ingest\process_ticket.py
```

If you already trust the proposed slug and want to skip the confirmation prompt for high-confidence cases:

```powershell
python .\archive-ingest\process_ticket.py .\archive-ingest\incoming\my-ticket.jpg --yes
```

## Batch controller outputs

The batch controller creates review-only outputs in these folders:

- `archive-ingest/working/split/`
  Split ticket candidates exported from each source scan.
- `archive-ingest/working/cropped/`
  Processed archive image candidates for each split ticket.
- `archive-ingest/working/enhanced/`
  Share image candidates and preview overlays for each split ticket.
- `archive-ingest/review/contact-sheets/`
  Split and image-processing contact sheets.
- `archive-ingest/review/notes/`
  Split notes, image notes, and metadata notes.
- `archive-ingest/review/draft-json/`
  Metadata review packages.
- `archive-ingest/review/share-pages/`
  Draft share page files for manual review only.
- `archive-ingest/review/manifests/`
  One structured handoff manifest per ticket candidate, including split path, processed image paths, processing notes, contact sheet, metadata draft paths, ticket object block, warnings, and review status.
- `archive-ingest/review/batch-logs/`
  One summary log per batch run, including scans processed, split candidates generated, metadata drafts generated, failures, and ambiguous results.

What still must be done manually before publishing:

- review every split candidate and reject bad fragments
- review every archive/share image candidate
- review metadata drafts, provenance, and research placeholders
- paste approved ticket objects into `tickets.js` manually
- copy approved images into the live site manually
- copy approved share page drafts into live share folders manually
- test locally before any deployment

## Splitter outputs

The splitter creates review-only outputs in these existing folders:

- `archive-ingest/working/split/`
  Numbered split image candidates.
- `archive-ingest/review/contact-sheets/`
  Full-scan preview overlays and split contact sheets.
- `archive-ingest/review/notes/`
  Human-readable split review notes.

The split review notes call out:

- scan summary
- first-stage region summary
- final exported region summary
- second-stage segmentation notes
- child region evaluation table
- first-stage broad regions detected
- whether second-stage segmentation was attempted
- whether second-stage segmentation succeeded
- whether smoothed separator bands were found inside oversized stacked regions
- child-by-child confidence, height ratio, area ratio, parent area ratio, accepted/rejected status, and rejection reason when discarded
- whether proposed child regions were rejected for being too small or not ticket-like
- how many proposed child regions were kept versus discarded
- final exported region count
- bounds and confidence estimates
- ambiguity and conservative detection notes

## Image processing outputs

The image processing step creates review candidates in these existing folders:

- `archive-ingest/working/cropped/`
  Archive image candidates.
- `archive-ingest/working/enhanced/`
  Share image candidates and detection preview images.
- `archive-ingest/review/contact-sheets/`
  Side-by-side visual review sheets.
- `archive-ingest/review/notes/`
  Human-readable image review notes.

The image review notes call out:

- detection method used
- confidence score
- whether fallback was used
- whether rotation was applied or skipped
- crop padding values
- why the archive and share framing were chosen

## Metadata ingest outputs

The metadata step creates review artifacts in these folders:

- `archive-ingest/review/draft-json/<slug>.json`
- `archive-ingest/review/notes/<slug>.txt`
- `archive-ingest/review/share-pages/<slug>.html`

The JSON review package includes:

- `ticket`: the proposed ticket data
- `provenance`: field-by-field confidence/provenance labels
- `research`: structured placeholder suggestions for later research integration
- `ticketObjectForTicketsJs`: a paste-ready object block for manual use in `tickets.js`

## Confidence labels

The metadata draft outputs use three labels:

- `confirmed_from_ticket`
- `inferred_from_research`
- `generated_suggestion`

These labels help separate what came from the ticket, what should eventually be confirmed with outside research, and what is still a generated draft suggestion.

## Image authenticity rules

The splitter and image-processing steps are designed to respect these rules:

- do not crop flush to the ticket edge
- preserve padding around the ticket
- preserve torn edges
- preserve paper texture and imperfections
- improve faded text by adjusting existing pixels only
- do not redraw text
- do not substitute fonts
- do not invent missing printed details

## Manual review checklist

Before publishing anything manually:

- confirm each split contains one full ticket rather than a fragment
- confirm the archive crop still feels authentic
- confirm torn edges and paper texture are preserved
- confirm the share image framing looks balanced
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
