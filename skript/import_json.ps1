param (
    [string]$JsonPath,
    [string]$OutPath
)

$json = Get-Content $JsonPath -Raw | ConvertFrom-Json

$id       = $json.id
$merchant = $json.receipt.merchant
$date     = $json.receipt.date

$lines = foreach ($item in $json.items) {
    "$id|$($item.name)|$($item.quantity)|$($item.unit_price)|$($item.total_price)|$merchant|$date"
}

$lines | Set-Content -Path $OutPath -Encoding UTF8
