param(
    [Parameter(Mandatory = $true)]
    [string]$WorkbookPath,

    [string]$WorksheetName,

    [string]$SiteRoot,

    [string]$ImageRoot,

    [switch]$DryRun,

    [switch]$ForceNeedsReview,

    [switch]$IncludeNeedsReview
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$SiteBaseUrl = "https://www.anthonycdorsey.com/shows-ive-seen"

if ([string]::IsNullOrWhiteSpace($SiteRoot)) {
    $SiteRoot = $PSScriptRoot
}

if ([string]::IsNullOrWhiteSpace($SiteRoot)) {
    throw "SiteRoot is blank."
}

if ([string]::IsNullOrWhiteSpace($WorkbookPath)) {
    throw "WorkbookPath is blank."
}

if ([string]::IsNullOrWhiteSpace($ImageRoot)) {
    $ImageRoot = $SiteRoot
}

if ([string]::IsNullOrWhiteSpace($ImageRoot)) {
    throw "ImageRoot is blank."
}

$TicketsJsPath = Join-Path $SiteRoot "tickets.js"
$ShareRoot = Join-Path $SiteRoot "share"

function Get-TrimmedString {
    param([object]$Value)

    if ($null -eq $Value) {
        return ""
    }

    return ([string]$Value).Trim()
}

function ConvertTo-BoolValue {
    param([object]$Value)

    $text = (Get-TrimmedString $Value).ToLowerInvariant()
    return $text -in @("1", "true", "yes", "y", "x")
}

function ConvertTo-Slug {
    param([string]$Value)

    $text = (Get-TrimmedString $Value).ToLowerInvariant()
    $text = $text -replace "&", " and "
    $text = $text -replace "/", " "
    $text = $text -replace "[^a-z0-9\s-]", ""
    $text = $text -replace "\s+", "-"
    $text = $text -replace "-+", "-"
    return $text.Trim("-")
}

function Get-RequiredColumnMap {
    return @{
        image_filename   = $true
        artist           = $true
        venue            = $true
        city             = $true
        state            = $false
        country          = $false
        year             = $false
        exact_date       = $false
        price            = $false
        "opener(s)"      = $false
        tour             = $false
        associated_album = $false
        setlist_fm_url   = $false
        youtube_url      = $false
        tags             = $false
        short_description = $false
        notes            = $false
        confidence       = $false
        needs_review     = $false
    }
}

function Get-WorksheetRows {
    param(
        [string]$WorkbookFile,
        [string]$RequestedWorksheetName
    )

    $excel = $null
    $workbook = $null
    $worksheet = $null
    $usedRange = $null

    try {
        $excel = New-Object -ComObject Excel.Application
        $excel.Visible = $false
        $excel.DisplayAlerts = $false

        $resolvedWorkbookPath = (Resolve-Path $WorkbookFile).Path
        $workbook = $excel.Workbooks.Open($resolvedWorkbookPath)

        if ($RequestedWorksheetName) {
            $worksheet = $workbook.Worksheets.Item($RequestedWorksheetName)
        }
        else {
            $worksheet = $workbook.Worksheets.Item(1)
        }

        $usedRange = $worksheet.UsedRange
        $rowCount = [int]$usedRange.Rows.Count
        $columnCount = [int]$usedRange.Columns.Count

        if ($rowCount -lt 2) {
            throw "The worksheet does not contain a header row plus data rows."
        }

        $headers = @()
        for ($columnIndex = 1; $columnIndex -le $columnCount; $columnIndex++) {
            $headers += (Get-TrimmedString $usedRange.Cells.Item(1, $columnIndex).Text)
        }

        $headerMap = @{}
        for ($columnIndex = 0; $columnIndex -lt $headers.Count; $columnIndex++) {
            $headerName = $headers[$columnIndex]
            if (-not $headerName) {
                continue
            }

            if ($headerMap.ContainsKey($headerName)) {
                throw "Duplicate spreadsheet header detected: '$headerName'"
            }

            $headerMap[$headerName] = $columnIndex + 1
        }

        # Keep the spreadsheet contract explicit so the workbook can stay the
        # single source of truth without hidden field name assumptions.
        $requiredColumns = Get-RequiredColumnMap
        $missingColumns = @(
            $requiredColumns.Keys |
                Where-Object { $requiredColumns[$_] -and -not $headerMap.ContainsKey($_) }
        )

        if ($missingColumns.Count -gt 0) {
            throw "Workbook is missing required columns: $($missingColumns -join ', ')"
        }

        $rows = @()
        for ($rowIndex = 2; $rowIndex -le $rowCount; $rowIndex++) {
            $row = [ordered]@{
                _rowNumber = $rowIndex
            }

            foreach ($headerName in $headerMap.Keys) {
                $columnIndex = [int]$headerMap[$headerName]
                $row[$headerName] = Get-TrimmedString $usedRange.Cells.Item($rowIndex, $columnIndex).Text
            }

            $hasContent = $false
            foreach ($key in $row.Keys) {
                if ($key -eq "_rowNumber") {
                    continue
                }

                if ($row[$key]) {
                    $hasContent = $true
                    break
                }
            }

            if ($hasContent) {
                $rows += [pscustomobject]$row
            }
        }

        return $rows
    }
    finally {
        if ($workbook) {
            $workbook.Close($false)
        }

        if ($excel) {
            $excel.Quit()
        }

        foreach ($comObject in @($usedRange, $worksheet, $workbook, $excel)) {
            if ($null -ne $comObject) {
                [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($comObject)
            }
        }

        [GC]::Collect()
        [GC]::WaitForPendingFinalizers()
    }
}

function ConvertTo-TagArray {
    param([string]$Value)

    if (-not $Value) {
        return @()
    }

    $parts = $Value -split "[,;|]"
    $tags = New-Object System.Collections.Generic.List[string]
    foreach ($part in $parts) {
        $tag = (Get-TrimmedString $part)
        if ($tag -and -not $tags.Contains($tag)) {
            $tags.Add($tag)
        }
    }

    return @($tags)
}

function ConvertTo-JsString {
    param([string]$Value)

    $text = Get-TrimmedString $Value
    $text = $text.Replace("\", "\\")
    $text = $text.Replace('"', '\"')
    $text = $text.Replace("`r", "\r")
    $text = $text.Replace("`n", "\n")
    return $text
}

function Format-JsArray {
    param([string[]]$Values)

    if ($null -eq $Values -or @($Values).Count -eq 0) {
        return "[]"
    }

    $escaped = $Values | ForEach-Object { '"' + (ConvertTo-JsString $_) + '"' }
    return "[" + ($escaped -join ", ") + "]"
}

function Get-YearValue {
    param(
        [string]$YearText,
        [string]$ExactDateText
    )

    if ($YearText) {
        return $YearText
    }

    if ($ExactDateText -match "^(\d{4})-\d{2}-\d{2}$") {
        return $Matches[1]
    }

    return ""
}

function Get-ShareImageFileName {
    param(
        [string]$Slug,
        [string]$ImageFileName,
        [string]$ResolvedSiteRoot
    )

    $shareFileName = "$Slug-share.jpg"
    $shareImagePath = Join-Path $ResolvedSiteRoot $shareFileName
    if (Test-Path $shareImagePath) {
        return $shareFileName
    }

    return $ImageFileName
}

function New-TicketSlug {
    param(
        [pscustomobject]$Row,
        [hashtable]$UsedSlugs
    )

    $year = Get-YearValue -YearText $Row.year -ExactDateText $Row.exact_date
    $baseSlug = ConvertTo-Slug "$($Row.artist) $($Row.venue) $year"
    if (-not $baseSlug) {
        $baseSlug = ConvertTo-Slug ([System.IO.Path]::GetFileNameWithoutExtension($Row.image_filename))
    }

    $candidate = $baseSlug
    if ($UsedSlugs.ContainsKey($candidate) -and $Row.exact_date -match "^\d{4}-\d{2}-\d{2}$") {
        $candidate = ConvertTo-Slug "$($Row.artist) $($Row.venue) $($Row.exact_date)"
    }

    if (-not $candidate) {
        $candidate = "ticket"
    }

    $suffix = 2
    $uniqueSlug = $candidate
    while ($UsedSlugs.ContainsKey($uniqueSlug)) {
        $uniqueSlug = "$candidate-$suffix"
        $suffix++
    }

    $UsedSlugs[$uniqueSlug] = $true
    return $uniqueSlug
}

function Get-ValidationErrors {
    param(
        [pscustomobject]$Row,
        [string]$YearValue,
        [string]$ImagePath
    )

    $errors = New-Object System.Collections.Generic.List[string]

    if (-not $Row.artist) {
        $errors.Add("artist is blank")
    }

    if (-not $Row.venue) {
        $errors.Add("venue is blank")
    }

    if (-not $Row.image_filename) {
        $errors.Add("image_filename is blank")
    }

    if (-not $Row.city) {
        $errors.Add("city is blank")
    }

    if (-not $YearValue) {
        $errors.Add("year/exact_date is blank")
    }

    if ($Row.exact_date -and $Row.exact_date -notmatch "^\d{4}-\d{2}-\d{2}$") {
        $errors.Add("exact_date must be YYYY-MM-DD")
    }

    $country = if ($Row.country) { $Row.country } else { "USA" }
    if (($country -in @("USA", "US", "United States", "United States of America")) -and -not $Row.state) {
        $errors.Add("state is blank for a US row")
    }

    if (-not (Test-Path $ImagePath)) {
        $errors.Add("image file is missing: $($Row.image_filename)")
    }

    return @($errors.ToArray())
}

function Convert-RowToTicketObject {
    param(
        [pscustomobject]$Row,
        [string]$Slug,
        [string]$ResolvedSiteRoot
    )

    $year = Get-YearValue -YearText $Row.year -ExactDateText $Row.exact_date
    $country = if ($Row.country) { $Row.country } else { "USA" }
    $copyText = if ($Row.short_description) { $Row.short_description } elseif ($Row.notes) { $Row.notes } else { "$($Row.artist) at $($Row.venue), $year." }
    $extendedNotes = if ($Row.notes) { $Row.notes } elseif ($Row.short_description) { $Row.short_description } else { "$($Row.artist) at $($Row.venue), $year." }
    $shareImageFileName = Get-ShareImageFileName -Slug $Slug -ImageFileName $Row.image_filename -ResolvedSiteRoot $ResolvedSiteRoot

    # This is the live-site object shape consumed by tickets.js and index.html.
    return [pscustomobject][ordered]@{
        artist = $Row.artist
        artistSlug = ConvertTo-Slug $Row.artist
        exactDate = $Row.exact_date
        year = $year
        venue = $Row.venue
        city = $Row.city
        state = $Row.state
        country = $country
        copy = $copyText
        extendedNotes = $extendedNotes
        companions = @()
        photos = @()
        youtubeUrl = $Row.youtube_url
        price = $Row.price
        tags = ConvertTo-TagArray $Row.tags
        shareTitle = "$($Row.artist) at $($Row.venue), $year | Shows I Saw"
        shareDescription = $copyText
        shareImage = "$SiteBaseUrl/$shareImageFileName"
        slug = $Slug
        img = $Row.image_filename
        rotation = "0deg"
        notes = ""
        youtube = ""
        opener = $Row.'opener(s)'
        tour = $Row.tour
        associatedAlbum = $Row.associated_album
        setlistFmUrl = $Row.setlist_fm_url
        confidence = $Row.confidence
        needsReview = ConvertTo-BoolValue $Row.needs_review
        spreadsheetRow = $Row._rowNumber
    }
}

function ConvertFrom-JsEscapedString {
    param([string]$Value)

    if ($null -eq $Value) {
        return ""
    }

    return $Value.Replace('\"', '"').Replace('\\', '\')
}

function Get-StringFieldFromJsBlock {
    param(
        [string]$Block,
        [string]$FieldName
    )

    $match = [regex]::Match($Block, '(?ms)^\s*' + [regex]::Escape($FieldName) + ':\s*"((?:\\.|[^"\\])*)"')
    if (-not $match.Success) {
        return ""
    }

    return ConvertFrom-JsEscapedString $match.Groups[1].Value
}

function Get-StringArrayFieldFromJsBlock {
    param(
        [string]$Block,
        [string]$FieldName
    )

    $match = [regex]::Match($Block, '(?ms)^\s*' + [regex]::Escape($FieldName) + ':\s*\[(.*?)\]')
    if (-not $match.Success) {
        return @()
    }

    $items = New-Object System.Collections.Generic.List[string]
    $stringMatches = [regex]::Matches($match.Groups[1].Value, '"((?:\\.|[^"\\])*)"')
    foreach ($stringMatch in $stringMatches) {
        $items.Add((ConvertFrom-JsEscapedString $stringMatch.Groups[1].Value))
    }

    return @($items.ToArray())
}

function Get-ExistingTicketsBySlug {
    param([string]$TicketsFilePath)

    $ticketsBySlug = @{}
    $content = Get-Content -Path $TicketsFilePath -Raw
    $blockMatches = [regex]::Matches($content, '(?ms)^\s*\{.*?^\s*\}')

    foreach ($blockMatch in $blockMatches) {
        $block = $blockMatch.Value
        $slug = Get-StringFieldFromJsBlock -Block $block -FieldName "slug"
        if (-not $slug) {
            continue
        }

        $ticketsBySlug[$slug] = [pscustomobject]@{
            copy = Get-StringFieldFromJsBlock -Block $block -FieldName "copy"
            extendedNotes = Get-StringFieldFromJsBlock -Block $block -FieldName "extendedNotes"
            notes = Get-StringFieldFromJsBlock -Block $block -FieldName "notes"
            youtubeUrl = Get-StringFieldFromJsBlock -Block $block -FieldName "youtubeUrl"
            youtube = Get-StringFieldFromJsBlock -Block $block -FieldName "youtube"
            companions = Get-StringArrayFieldFromJsBlock -Block $block -FieldName "companions"
            photos = Get-StringArrayFieldFromJsBlock -Block $block -FieldName "photos"
            tags = Get-StringArrayFieldFromJsBlock -Block $block -FieldName "tags"
            rotation = Get-StringFieldFromJsBlock -Block $block -FieldName "rotation"
        }
    }

    return $ticketsBySlug
}

function Test-IsPlaceholderRichText {
    param([string]$Value)

    $trimmed = Get-TrimmedString $Value
    if (-not $trimmed) {
        return $true
    }

    $placeholderPatterns = @(
        '^This draft keeps the tone concise',
        '^Update this with personal context',
        '^Image filename suggests supplementary material',
        'A preserved ticket from the archive\.$',
        ' in \d{4}\.?$',
        'Exact date not cleanly legible from OCR\.?$'
    )

    foreach ($pattern in $placeholderPatterns) {
        if ($trimmed -match $pattern) {
            return $true
        }
    }

    return $false
}

function Get-MergedRichString {
    param(
        [string]$IncomingValue,
        [string]$ExistingValue
    )

    $incomingTrimmed = Get-TrimmedString $IncomingValue
    $existingTrimmed = Get-TrimmedString $ExistingValue

    if ((Test-IsPlaceholderRichText $incomingTrimmed) -and $existingTrimmed) {
        return $ExistingValue
    }

    return $IncomingValue
}

function Get-MergedRichArray {
    param(
        [object[]]$IncomingValue,
        [object[]]$ExistingValue
    )

    if (@($IncomingValue).Count -eq 0 -and @($ExistingValue).Count -gt 0) {
        return @($ExistingValue)
    }

    return @($IncomingValue)
}

function Merge-TicketWithExistingRichContent {
    param(
        [pscustomobject]$IncomingTicket,
        [pscustomobject]$ExistingTicket
    )

    if ($null -eq $ExistingTicket) {
        return $IncomingTicket
    }

    $IncomingTicket.copy = Get-MergedRichString -IncomingValue $IncomingTicket.copy -ExistingValue $ExistingTicket.copy
    $IncomingTicket.extendedNotes = Get-MergedRichString -IncomingValue $IncomingTicket.extendedNotes -ExistingValue $ExistingTicket.extendedNotes
    $IncomingTicket.notes = Get-MergedRichString -IncomingValue $IncomingTicket.notes -ExistingValue $ExistingTicket.notes
    $IncomingTicket.youtubeUrl = Get-MergedRichString -IncomingValue $IncomingTicket.youtubeUrl -ExistingValue $ExistingTicket.youtubeUrl
    $IncomingTicket.youtube = Get-MergedRichString -IncomingValue $IncomingTicket.youtube -ExistingValue $ExistingTicket.youtube
    $IncomingTicket.companions = Get-MergedRichArray -IncomingValue $IncomingTicket.companions -ExistingValue $ExistingTicket.companions
    $IncomingTicket.photos = Get-MergedRichArray -IncomingValue $IncomingTicket.photos -ExistingValue $ExistingTicket.photos
    $IncomingTicket.tags = Get-MergedRichArray -IncomingValue $IncomingTicket.tags -ExistingValue $ExistingTicket.tags
    if ((Get-TrimmedString $IncomingTicket.rotation) -in @("", "0deg") -and (Get-TrimmedString $ExistingTicket.rotation) -and $ExistingTicket.rotation -ne "0deg") {
        $IncomingTicket.rotation = $ExistingTicket.rotation
    }

    if ((Test-IsPlaceholderRichText $IncomingTicket.copy) -and (Get-TrimmedString $ExistingTicket.copy)) {
        $IncomingTicket.shareDescription = $ExistingTicket.copy
    }

    return $IncomingTicket
}

function Convert-TicketToJsBlock {
    param([pscustomobject]$Ticket)

    return @"
    {
        artist: "$(ConvertTo-JsString $Ticket.artist)",
        artistSlug: "$(ConvertTo-JsString $Ticket.artistSlug)",
        exactDate: "$(ConvertTo-JsString $Ticket.exactDate)",
        year: "$(ConvertTo-JsString $Ticket.year)",
        venue: "$(ConvertTo-JsString $Ticket.venue)",
        city: "$(ConvertTo-JsString $Ticket.city)",
        state: "$(ConvertTo-JsString $Ticket.state)",
        country: "$(ConvertTo-JsString $Ticket.country)",
        copy: "$(ConvertTo-JsString $Ticket.copy)",
        extendedNotes: "$(ConvertTo-JsString $Ticket.extendedNotes)",
        companions: $(Format-JsArray $Ticket.companions),
        photos: $(Format-JsArray $Ticket.photos),
        youtubeUrl: "$(ConvertTo-JsString $Ticket.youtubeUrl)",
        price: "$(ConvertTo-JsString $Ticket.price)",
        tags: $(Format-JsArray $Ticket.tags),
        shareTitle: "$(ConvertTo-JsString $Ticket.shareTitle)",
        shareDescription: "$(ConvertTo-JsString $Ticket.shareDescription)",
        shareImage: "$(ConvertTo-JsString $Ticket.shareImage)",
        slug: "$(ConvertTo-JsString $Ticket.slug)",

        img: "$(ConvertTo-JsString $Ticket.img)",
        rotation: "$(ConvertTo-JsString $Ticket.rotation)",

        // temporary backward compatibility
        notes: "$(ConvertTo-JsString $Ticket.notes)",
        youtube: "$(ConvertTo-JsString $Ticket.youtube)"
    }
"@
}

function Convert-TicketsToJsFile {
    param([object[]]$Tickets)

    $blocks = $Tickets | ForEach-Object { Convert-TicketToJsBlock $_ }
    $joinedBlocks = $blocks -join ",`r`n"
    return "const tickets = [`r`n$joinedBlocks`r`n];`r`n"
}

function Convert-ToHtmlAttribute {
    param([string]$Value)

    return [System.Net.WebUtility]::HtmlEncode((Get-TrimmedString $Value))
}

function Convert-TicketToSharePageHtml {
    param([pscustomobject]$Ticket)

    $shareUrl = "$SiteBaseUrl/share/$($Ticket.slug)/"
    $deepLinkUrl = "$SiteBaseUrl/#$($Ticket.slug)"
    $title = Convert-ToHtmlAttribute $Ticket.shareTitle
    $description = Convert-ToHtmlAttribute $Ticket.shareDescription
    $shareImage = Convert-ToHtmlAttribute $Ticket.shareImage
    $artist = Convert-ToHtmlAttribute $Ticket.artist
    $venue = Convert-ToHtmlAttribute $Ticket.venue
    $year = Convert-ToHtmlAttribute $Ticket.year
    $shareUrlAttr = Convert-ToHtmlAttribute $shareUrl
    $deepLinkAttr = Convert-ToHtmlAttribute $deepLinkUrl

    return @"
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>$title</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">

  <meta name="description" content="$description">

  <meta property="og:type" content="website">
  <meta property="og:title" content="$title">
  <meta property="og:description" content="$description">
  <meta property="og:url" content="$shareUrlAttr">
  <meta property="og:site_name" content="Anthony C. Dorsey">

  <meta property="og:image" content="$shareImage">
  <meta property="og:image:secure_url" content="$shareImage">
  <meta property="og:image:type" content="image/jpeg">
  <meta property="og:image:width" content="1200">
  <meta property="og:image:height" content="630">
  <meta property="og:image:alt" content="$artist concert ticket from $venue in $year">

  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="$title">
  <meta name="twitter:description" content="$description">
  <meta name="twitter:image" content="$shareImage">

  <link rel="canonical" href="$shareUrlAttr">

  <script>
    window.location.replace("$deepLinkAttr");
  </script>
</head>
<body>
  <p>Opening ticket...</p>
</body>
</html>
"@
}

function Write-SharePages {
    param(
        [object[]]$Tickets,
        [string]$ResolvedShareRoot
    )

    foreach ($ticket in $Tickets) {
        # Share pages stay tiny redirect documents so the current social/share
        # behavior remains unchanged.
        $shareDirectory = Join-Path $ResolvedShareRoot $ticket.slug
        $shareFilePath = Join-Path $shareDirectory "index.html"
        $shareHtml = Convert-TicketToSharePageHtml $ticket

        New-Item -ItemType Directory -Path $shareDirectory -Force | Out-Null
        Set-Content -Path $shareFilePath -Value $shareHtml -Encoding UTF8
    }
}

try {
    if (-not (Test-Path $WorkbookPath)) {
        throw "Workbook not found: $WorkbookPath"
    }

    if (-not (Test-Path $SiteRoot)) {
        throw "Site root not found: $SiteRoot"
    }

    if (-not (Test-Path $ImageRoot)) {
        throw "Image root not found: $ImageRoot"
    }

    if (-not (Test-Path $TicketsJsPath)) {
        throw "tickets.js not found: $TicketsJsPath"
    }

    $resolvedSiteRoot = (Resolve-Path $SiteRoot).Path
    $resolvedImageRoot = (Resolve-Path $ImageRoot).Path
    $resolvedWorkbookPath = (Resolve-Path $WorkbookPath).Path
    $rows = @(Get-WorksheetRows -WorkbookFile $resolvedWorkbookPath -RequestedWorksheetName $WorksheetName)
    $existingTicketsBySlug = Get-ExistingTicketsBySlug -TicketsFilePath $TicketsJsPath
    $usedSlugs = @{}
    $resolvedShareRoot = Join-Path $resolvedSiteRoot "share"
    $tickets = New-Object System.Collections.Generic.List[object]
    $skippedNeedsReview = New-Object System.Collections.Generic.List[string]
    $validationFailures = New-Object System.Collections.Generic.List[string]
    $missingImageRows = New-Object System.Collections.Generic.List[string]

    foreach ($row in $rows) {
        $needsReview = ConvertTo-BoolValue $row.needs_review
        if ($needsReview -and -not $ForceNeedsReview.IsPresent -and -not $IncludeNeedsReview.IsPresent) {
            $skippedNeedsReview.Add("row $($row._rowNumber): $($row.artist) / $($row.venue)")
            continue
        }

        $imagePath = Join-Path $resolvedImageRoot $row.image_filename
        $yearValue = Get-YearValue -YearText $row.year -ExactDateText $row.exact_date
        $errors = @(Get-ValidationErrors -Row $row -YearValue $yearValue -ImagePath $imagePath)
        if (@($errors).Count -gt 0) {
            $message = "row $($row._rowNumber): " + ($errors -join "; ")
            $validationFailures.Add($message)
            if ($errors -match "^image file is missing") {
                $missingImageRows.Add($message)
            }
            continue
        }

        $slug = New-TicketSlug -Row $row -UsedSlugs $usedSlugs
        $ticket = Convert-RowToTicketObject -Row $row -Slug $slug -ResolvedSiteRoot $resolvedSiteRoot
        if ($existingTicketsBySlug.ContainsKey($slug)) {
            $ticket = Merge-TicketWithExistingRichContent -IncomingTicket $ticket -ExistingTicket $existingTicketsBySlug[$slug]
        }
        $tickets.Add($ticket)
    }

    $ticketsJsContent = Convert-TicketsToJsFile $tickets
    $backupPath = Join-Path $resolvedSiteRoot ("tickets.js.backup.{0}.js" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
    $rowsReadCount = @($rows).Count
    $rowsPublishedCount = $tickets.Count
    $rowsSkippedNeedsReviewCount = $skippedNeedsReview.Count
    $rowsValidationFailuresCount = $validationFailures.Count
    $missingImageRowsCount = $missingImageRows.Count
    $includeNeedsReviewEnabled = $IncludeNeedsReview.IsPresent -or $ForceNeedsReview.IsPresent

    Write-Host ""
    Write-Host "Publish summary"
    Write-Host "--------------"
    Write-Host ("Workbook: {0}" -f $resolvedWorkbookPath)
    Write-Host ("Image root used: {0}" -f $resolvedImageRoot)
    Write-Host ("IncludeNeedsReview enabled: {0}" -f $includeNeedsReviewEnabled)
    Write-Host ("Rows read: {0}" -f $rowsReadCount)
    Write-Host ("Rows published: {0}" -f $rowsPublishedCount)
    Write-Host ("Rows skipped for needs_review: {0}" -f $rowsSkippedNeedsReviewCount)
    Write-Host ("Rows with validation errors: {0}" -f $rowsValidationFailuresCount)
    Write-Host ("Missing image rows: {0}" -f $missingImageRowsCount)
    Write-Host ("tickets.js backup: {0}" -f $backupPath)
    Write-Host ""

    if (@($skippedNeedsReview).Count -gt 0) {
        Write-Host "Skipped because needs_review=true"
        $skippedNeedsReview | ForEach-Object { Write-Host "  $_" }
        Write-Host ""
    }

    if (@($validationFailures).Count -gt 0) {
        Write-Host "Validation failures"
        $validationFailures | ForEach-Object { Write-Host "  $_" }
        Write-Host ""
    }

    if ($DryRun.IsPresent) {
        Write-Host "Dry run only. No files were written."
        exit 0
    }

    Copy-Item -Path $TicketsJsPath -Destination $backupPath -Force
    Set-Content -Path $TicketsJsPath -Value $ticketsJsContent -Encoding UTF8
    Write-SharePages -Tickets $tickets -ResolvedShareRoot $resolvedShareRoot

    Write-Host "Published tickets.js and share pages."
}
catch {
    $errorRecord = $_
    Write-Host ""
    Write-Host "Publisher failed"
    Write-Host "----------------"
    Write-Host "Message: $($errorRecord.Exception.Message)"
    if ($errorRecord.InvocationInfo) {
        Write-Host "Line: $($errorRecord.InvocationInfo.ScriptLineNumber)"
        Write-Host "Statement: $($errorRecord.InvocationInfo.Line.Trim())"
        Write-Host "Position:"
        Write-Host $errorRecord.InvocationInfo.PositionMessage
    }

    if (Get-Variable -Name rows -ErrorAction SilentlyContinue) {
        Write-Host "Type rows: $($rows.GetType().FullName)"
    }
    if (Get-Variable -Name errors -ErrorAction SilentlyContinue) {
        Write-Host "Type errors: $($errors.GetType().FullName)"
        Write-Host "errors wrapped count: $(@($errors).Count)"
    }
    if (Get-Variable -Name tickets -ErrorAction SilentlyContinue) {
        Write-Host "Type tickets: $($tickets.GetType().FullName)"
    }
    if (Get-Variable -Name validationFailures -ErrorAction SilentlyContinue) {
        Write-Host "Type validationFailures: $($validationFailures.GetType().FullName)"
    }

    throw
}
