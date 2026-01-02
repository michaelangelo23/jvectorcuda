# JVectorCUDA - JCuda Setup Script (Windows PowerShell)
# Downloads JCuda 12.0.0 JARs for CUDA 11.8 compatibility

param(
    [string]$LibsDir = "libs"
)

$ErrorActionPreference = "Stop"

# JCuda version configuration
$JCUDA_VERSION = "12.0.0"
$BASE_URL = "https://repo1.maven.org/maven2/org/jcuda"

# Required JCuda modules
$MODULES = @(
    "jcuda",
    "jcuda-natives",
    "jcublas",
    "jcublas-natives",
    "jcurand",
    "jcurand-natives",
    "jcufft",
    "jcufft-natives",
    "jcusparse",
    "jcusparse-natives",
    "jcusolver",
    "jcusolver-natives"
)

# Detect OS for native classifier
$OS_CLASSIFIER = "windows-x86_64"
if ($IsLinux) { $OS_CLASSIFIER = "linux-x86_64" }
if ($IsMacOS) { $OS_CLASSIFIER = "apple-x86_64" }

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " JVectorCUDA - JCuda Setup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "JCuda Version: $JCUDA_VERSION"
Write-Host "Platform: $OS_CLASSIFIER"
Write-Host "Target Directory: $LibsDir"
Write-Host ""

# Create libs directory
if (-not (Test-Path $LibsDir)) {
    New-Item -ItemType Directory -Path $LibsDir | Out-Null
    Write-Host "Created directory: $LibsDir" -ForegroundColor Green
}

# Download function
function Download-JCudaJar {
    param($Module, $Classifier)
    
    if ($Classifier) {
        $jarName = "$Module-$JCUDA_VERSION-$Classifier.jar"
        $url = "$BASE_URL/$Module/$JCUDA_VERSION/$jarName"
    } else {
        $jarName = "$Module-$JCUDA_VERSION.jar"
        $url = "$BASE_URL/$Module/$JCUDA_VERSION/$jarName"
    }
    
    $targetPath = Join-Path $LibsDir $jarName
    
    if (Test-Path $targetPath) {
        Write-Host "  [SKIP] $jarName (already exists)" -ForegroundColor Yellow
        return
    }
    
    Write-Host "  [DOWNLOAD] $jarName..." -ForegroundColor White
    try {
        Invoke-WebRequest -Uri $url -OutFile $targetPath -UseBasicParsing
        Write-Host "  [OK] $jarName" -ForegroundColor Green
    } catch {
        Write-Host "  [FAIL] $jarName - $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host "Downloading JCuda JARs..." -ForegroundColor Cyan
Write-Host ""

foreach ($module in $MODULES) {
    if ($module -like "*-natives") {
        # Native JARs need platform classifier
        Download-JCudaJar -Module $module -Classifier $OS_CLASSIFIER
    } else {
        # Pure Java JARs
        Download-JCudaJar -Module $module -Classifier $null
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Ensure CUDA Toolkit 11.8+ is installed"
Write-Host "  2. Run: ./gradlew build"
Write-Host "  3. Run: ./gradlew test"
Write-Host ""
Write-Host "Troubleshooting:"
Write-Host "  - CUDA_ERROR_NO_DEVICE: No NVIDIA GPU detected"
Write-Host "  - CUDA_ERROR_INVALID_PTX: CUDA version mismatch"
Write-Host "  - UnsatisfiedLinkError: Missing CUDA runtime DLLs"
Write-Host ""
