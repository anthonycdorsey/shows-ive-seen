# Spreadsheet Publishing

This repo now has one simple publish step for the live site.

Flow:

1. Confirm your cleaned Excel workbook is ready.
2. Confirm each ticket image named by `image_filename` already exists in the live site folder.
3. Run the publish script.
4. Review the console summary.

## Command

From [shows-ive-seen](C:\Users\Antho\OneDrive\Documents\Website\Shows Ive Seen\shows-ive-seen):

```powershell
powershell -ExecutionPolicy Bypass -File .\publish-site-from-spreadsheet.ps1 -WorkbookPath "C:\path\to\master.xlsx" -WorksheetName "Sheet1"
```

Dry run:

```powershell
powershell -ExecutionPolicy Bypass -File .\publish-site-from-spreadsheet.ps1 -WorkbookPath "C:\path\to\master.xlsx" -WorksheetName "Sheet1" -DryRun
```

Include rows marked `needs_review = true`:

```powershell
powershell -ExecutionPolicy Bypass -File .\publish-site-from-spreadsheet.ps1 -WorkbookPath "C:\path\to\master.xlsx" -WorksheetName "Sheet1" -ForceNeedsReview
```

## What The Script Does

- reads the workbook through Excel on Windows
- validates required fields row by row
- skips `needs_review = true` unless `-ForceNeedsReview` is passed
- reports missing image files
- generates unique slugs
- rewrites [tickets.js](C:\Users\Antho\OneDrive\Documents\Website\Shows Ive Seen\shows-ive-seen\tickets.js)
- creates or updates `share/<slug>/index.html`
- creates a timestamped backup of `tickets.js` before overwrite

## Notes

- The script assumes row 1 contains the spreadsheet headers.
- `exact_date` should be `YYYY-MM-DD` when present.
- For US shows, `state` should be filled in.
- If a dedicated `<slug>-share.jpg` exists, the share page will use it.
- If no dedicated share image exists, the share page falls back to the main `image_filename`.
- Fields that are not in the spreadsheet today, like companions, photos, and rotation, default to empty arrays or `0deg`.
