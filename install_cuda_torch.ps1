# install_cuda_torch.ps1
# Downloads and installs PyTorch 2.6.0 + CUDA 12.4 for Python 3.13 in C:\pakvenv
# Run this from any PowerShell window (no Admin required)

$ErrorActionPreference = "Stop"

$python = "C:\pakvenv\Scripts\python.exe"
$pip    = "C:\pakvenv\Scripts\python.exe -m pip"

$wheelDir = "$env:TEMP\pytorch_cu124_wheels"
New-Item -ItemType Directory -Force -Path $wheelDir | Out-Null

$torchUrl        = "https://download-r2.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp313-cp313-win_amd64.whl"
$torchvisionUrl  = "https://download-r2.pytorch.org/whl/cu124/torchvision-0.21.0%2Bcu124-cp313-cp313-win_amd64.whl"

$torchWheel       = "$wheelDir\torch-2.6.0+cu124-cp313-cp313-win_amd64.whl"
$torchvisionWheel = "$wheelDir\torchvision-0.21.0+cu124-cp313-cp313-win_amd64.whl"

function Download-File($url, $outFile) {
    if (Test-Path $outFile) {
        $existingSize = (Get-Item $outFile).Length
        Write-Host "Partial file found ($existingSize bytes). Resuming download..." -ForegroundColor Cyan
        try {
            # BITS supports resume
            Start-BitsTransfer -Source $url -Destination $outFile -TransferType Download -ErrorAction Stop
        } catch {
            Write-Host "BITS resume failed, falling back to full re-download with WebClient..." -ForegroundColor Yellow
            Remove-Item $outFile -Force -ErrorAction SilentlyContinue
            $wc = New-Object System.Net.WebClient
            $wc.DownloadFile($url, $outFile)
        }
    } else {
        Write-Host "Downloading $([System.IO.Path]::GetFileName($outFile)) (this will take a few minutes)..." -ForegroundColor Cyan
        try {
            Start-BitsTransfer -Source $url -Destination $outFile -TransferType Download -ErrorAction Stop
        } catch {
            Write-Host "BITS failed, falling back to WebClient..." -ForegroundColor Yellow
            $wc = New-Object System.Net.WebClient
            $wc.DownloadFile($url, $outFile)
        }
    }
}

Write-Host "`n=== Step 1/4: Uninstalling existing CPU torch/torchvision ===" -ForegroundColor Green
Invoke-Expression "$pip uninstall -y torch torchvision torchaudio" | Out-Null
Write-Host "Done." -ForegroundColor Green

Write-Host "`n=== Step 2/4: Downloading wheels ===" -ForegroundColor Green
Download-File $torchUrl $torchWheel
Download-File $torchvisionUrl $torchvisionWheel
Write-Host "Downloads complete." -ForegroundColor Green

Write-Host "`n=== Step 3/4: Installing from local wheels ===" -ForegroundColor Green
Invoke-Expression "$pip install `"$torchWheel`" `"$torchvisionWheel`""
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

Write-Host "`n=== Step 4/4: Verifying installation ===" -ForegroundColor Green
Invoke-Expression "$python -c `"import torch; print('PyTorch version :', torch.__version__); print('CUDA available  :', torch.cuda.is_available()); print('CUDA version    :', torch.version.cuda if torch.cuda.is_available() else 'N/A')`""

Write-Host "`nAll done! You can now resume training on GPU." -ForegroundColor Green
