param (
    [Parameter(Mandatory = $true)]
    [string]$CsvPath
)

# CSV einlesen (Semikolon!)
$rows = Import-Csv $CsvPath -Delimiter ';'

if ($rows.Count -eq 0) {
    exit 0
}

foreach ($r in $rows) {
    "$($r.receipt_id)|$($r.name)|$($r.quantity)|$($r.unit_price)|$($r.total_price)|$($r.merchant)|$($r.receipt_date)|$($r.category)"
}
