param()
$ErrorActionPreference = 'Stop'
Set-Location 'C:\Users\Lenovo\PROJECTS\get_imdb_data\Scripts'
$py = Join-Path .venv 'Scripts\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }

# 1) Harvest
$reg = Join-Path 'data\processed\registry' ('channel_registry_' + (Get-Date -Format yyyyMMdd_HHmm) + '.csv')
& $py 'src\harvest_channel_ids.py' --out $reg

# 2) Filter
& $py 'src\filter_registry.py' --in $reg

# 3) Build playlist (timestamp inside script if name is curated.m3u)
$outm3u = 'outputs\playlists\curated.m3u'
$cur = Get-ChildItem 'data\processed\registry\curated\curated_registry_*.csv' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $cur) { Write-Error 'No curated registry produced.' }
& $py 'src\build_playlist.py' --in $cur.FullName --out $outm3u
