# apply_batch4.ps1
$ErrorActionPreference = "Stop"
function Write-Text { param([string]$Path,[string]$Content)
  $dir = Split-Path -Parent $Path
  if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
  Set-Content -Path $Path -Value $Content -Encoding UTF8
  Write-Host "Wrote: $Path"
}
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Host "Repo root: $root"

# --- src\ChannelStudio.pyw ---
$gui = @"
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import sys, subprocess, threading, time, glob
from pathlib import Path
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

class Runner(QtCore.QObject):
    line = QtCore.Signal(str); done = QtCore.Signal(int)
    def run(self, args: list[str], cwd: Path|None=None):
        def work():
            try:
                p = subprocess.Popen(args, cwd=str(cwd) if cwd else None,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8")
                for ln in p.stdout: self.line.emit(ln.rstrip("\n"))
                p.wait(); self.done.emit(p.returncode or 0)
            except Exception as e:
                self.line.emit(f"[!] Error: {e}"); self.done.emit(1)
        threading.Thread(target=work, daemon=True).start()

class Studio(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IPTV ChannelStudio")
        self.setMinimumSize(980, 700)
        self.runner = Runner(); self.runner.line.connect(self.on_line); self.runner.done.connect(self.on_done)

        self.btnHarvest = QtWidgets.QPushButton("1) Harvest ➜ Registry")
        self.btnFilter  = QtWidgets.QPushButton("2) Filter/Dedupe ➜ Curated")
        self.btnBuild   = QtWidgets.QPushButton("3) Build Playlist")
        self.btnCompare = QtWidgets.QPushButton("4) EPG Compare")
        self.btnPipe    = QtWidgets.QPushButton("Run Full Pipeline")
        for b in (self.btnHarvest,self.btnFilter,self.btnBuild,self.btnCompare,self.btnPipe): b.setMinimumHeight(36)

        self.outPath = QtWidgets.QLineEdit(str(REPO_ROOT / "outputs" / "playlists" / "curated.m3u"))
        self.pickOut = QtWidgets.QPushButton("…"); self.pickOut.setFixedWidth(36)

        self.table = QtWidgets.QTableWidget(0,6)
        self.table.setHorizontalHeaderLabels(["channel_name","tvg_id","epg_channel_id","group_title","country","language"])
        self.table.horizontalHeader().setStretchLastSection(True)

        self.log = QtWidgets.QTextEdit(); self.log.setReadOnly(True); self.log.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)

        top = QtWidgets.QHBoxLayout()
        for b in (self.btnHarvest,self.btnFilter,self.btnBuild,self.btnCompare): top.addWidget(b)
        top.addStretch(1); top.addWidget(self.btnPipe)
        outrow = QtWidgets.QHBoxLayout()
        outrow.addWidget(QtWidgets.QLabel("Playlist Out:")); outrow.addWidget(self.outPath,1); outrow.addWidget(self.pickOut)

        v = QtWidgets.QVBoxLayout(self)
        v.addLayout(top); v.addLayout(outrow)
        v.addWidget(QtWidgets.QLabel("Curated Preview (first rows):"))
        v.addWidget(self.table,2)
        v.addWidget(QtWidgets.QLabel("Log:"))
        v.addWidget(self.log,3)

        self.btnHarvest.clicked.connect(self.do_harvest)
        self.btnFilter.clicked.connect(self.do_filter)
        self.btnBuild.clicked.connect(self.do_build)
        self.btnCompare.clicked.connect(self.do_compare)
        self.btnPipe.clicked.connect(self.do_pipeline)
        self.pickOut.clicked.connect(self.pick_out)

        QtCore.QTimer.singleShot(600, self.refresh_preview)

    def append_log(self, txt: str):
        self.log.append(txt)
        self.log.moveCursor(QTextCursor.MoveOperation.End)

    def on_line(self, s: str): self.append_log(s)
    def on_done(self, code: int):
        self.append_log(f"[exit code {code}]")
        self.refresh_preview()

    def run(self, args: list[str]): self.log.clear(); self.append_log(" ".join(args)); self.runner.run(args, REPO_ROOT)

    def pick_out(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Playlist Output", self.outPath.text(), "M3U (*.m3u)")
        if fn: self.outPath.setText(fn)

    def latest_curated(self) -> Path|None:
        cur = sorted((REPO_ROOT/"data/processed/registry/curated").glob("curated_registry_*.csv"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
        return cur[0] if cur else None

    def refresh_preview(self):
        try:
            import pandas as pd
        except Exception:
            return
        src = self.latest_curated()
        if not src: return
        try:
            df = pd.read_csv(src)
            want = [c for c in ["channel_name","tvg_id","epg_channel_id","group_title","country","language"] if c in df.columns]
            df = df[want].head(200)
            self.table.setRowCount(len(df)); self.table.setColumnCount(len(want)); self.table.setHorizontalHeaderLabels(want)
            for r,(idx,row) in enumerate(df.iterrows()):
                for c,col in enumerate(want):
                    self.table.setItem(r,c,QtWidgets.QTableWidgetItem(str(row.get(col,""))))
        except Exception as e:
            self.append_log(f"[!] Preview error: {e}")

    def do_harvest(self):
        out = REPO_ROOT/"data/processed/registry"/("channel_registry_"+time.strftime("%Y%m%d_%H%M")+".csv")
        self.run([PY, str(REPO_ROOT/"src/harvest_channel_ids.py"), "--out", str(out)])

    def do_filter(self):
        reg = sorted((REPO_ROOT/"data/processed/registry").glob("channel_registry_*.csv"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
        if not reg: self.append_log("[!] No harvested registry found."); return
        self.run([PY, str(REPO_ROOT/"src/filter_registry.py"), "--in", str(reg[0])])

    def do_build(self):
        cur = self.latest_curated()
        if not cur: self.append_log("[!] No curated registry found."); return
        out = Path(self.outPath.text().strip() or str(REPO_ROOT/"outputs/playlists/curated.m3u"))
        self.run([PY, str(REPO_ROOT/"src/build_playlist.py"), "--in", str(cur), "--out", str(out)])

    def do_compare(self):
        m3u = Path(self.outPath.text().strip() or str(REPO_ROOT/"outputs/playlists/curated.m3u"))
        if not m3u.exists(): self.append_log(f"[!] Playlist not found: {m3u}"); return
        ps1 = REPO_ROOT/"compare.ps1"
        if ps1.exists():
            self.run(["powershell","-ExecutionPolicy","Bypass","-File",str(ps1),"-M3U",str(m3u)])
        else:
            self.run([PY, str(REPO_ROOT/"src/epg_compare.py"), "--m3u", str(m3u)])

    def do_pipeline(self):
        ps1 = REPO_ROOT/"pipeline.ps1"
        if ps1.exists():
            self.run(["powershell","-ExecutionPolicy","Bypass","-File",str(ps1)])
        else:
            self.append_log("[!] pipeline.ps1 not found.")

def main():
    app = QtWidgets.QApplication(sys.argv)
    ui = Studio(); ui.show()
    sys.exit(app.exec())

if __name__ == "__main__": main()
"@
Write-Text "$root\src\ChannelStudio.pyw" $gui

# --- pipeline.ps1 ---
$pipe = @"
param()
`$ErrorActionPreference = 'Stop'
Set-Location '$root'
`$py = Join-Path .venv 'Scripts\python.exe'
if (-not (Test-Path `$py)) { `$py = 'python' }

# 1) Harvest
`$reg = Join-Path 'data\processed\registry' ('channel_registry_' + (Get-Date -Format yyyyMMdd_HHmm) + '.csv')
& `$py 'src\harvest_channel_ids.py' --out `$reg

# 2) Filter
& `$py 'src\filter_registry.py' --in `$reg

# 3) Build playlist (timestamp inside script if name is curated.m3u)
`$outm3u = 'outputs\playlists\curated.m3u'
`$cur = Get-ChildItem 'data\processed\registry\curated\curated_registry_*.csv' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (`$null -eq `$cur) { Write-Error 'No curated registry produced.' }
& `$py 'src\build_playlist.py' --in `$cur.FullName --out `$outm3u
"@
Write-Text "$root\pipeline.ps1" $pipe

# --- wrappers (idempotent creates) ---
if (-not (Test-Path "$root\harvest.ps1")) {
  Write-Text "$root\harvest.ps1" "param([string]\$Out=''); \$py = Join-Path .venv 'Scripts\python.exe'; if(-not(Test-Path \$py)){ \$py='python' }; & \$py 'src\harvest_channel_ids.py' --out (\$Out -ne '' ? \$Out : ('data\processed\registry\channel_registry_'+(Get-Date -Format yyyyMMdd_HHmm)+'.csv'))"
}
if (-not (Test-Path "$root\playlist.ps1")) {
  Write-Text "$root\playlist.ps1" "param([string]\$In, [string]\$Out='outputs\playlists\curated.m3u'); if(-not \$In){ Write-Error 'Provide -In CSV'; exit 1 }; \$py=Join-Path .venv 'Scripts\python.exe'; if(-not(Test-Path \$py)){ \$py='python' }; & \$py 'src\build_playlist.py' --in \$In --out \$Out"
}
# compare.ps1 may already exist from batch2; keep if present
if (-not (Test-Path "$root\compare.ps1")) {
  Write-Text "$root\compare.ps1" "param([string]\$M3U, [string[]]\$EpgUrls=@(), [switch]\$FilteredXmltv); if(-not \$M3U){ Write-Error 'Provide -M3U'; exit 1 }; \$py=Join-Path .venv 'Scripts\python.exe'; if(-not(Test-Path \$py)){ \$py='python' }; \$args = @('--m3u', \$M3U); foreach(\$u in \$EpgUrls){ \$args += @('--epg', \$u) }; if(\$FilteredXmltv){ \$args += '--filtered-xmltv' }; & \$py 'src\epg_compare.py' @args"
}

Write-Host "apply_batch4.ps1 completed successfully."
