# Run EOD price backfill from 2010-01-01 using markines311 Python
$Python = "$env:LOCALAPPDATA\anaconda3\envs\markines311\python.exe"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
& $Python scripts\update_eod_prices.py --from-date 2010-01-01 @args
