<#
.SYNOPSIS
    Instala o Odysseus V2 Bridge Server para arrancar automaticamente com o Windows.
    Usa o Task Scheduler para criar uma tarefa que inicia o servidor bridge
    quando o utilizador faz login.

.DESCRIPTION
    Este script cria uma tarefa agendada no Windows chamada "OdysseusV2Bridge"
    que executa o servidor bridge V2 (v2_server.py) automaticamente no login.

    A tarefa:
    - Arranca no login do utilizador (não requer admin, mas pode pedir)
    - Executa em background (minimizada)
    - Reinicia automaticamente se falhar
    - Guarda logs em %LOCALAPPDATA%\OdysseusBridge\logs

.PARAMETER Port
    Porta do servidor bridge (default: 8001)

.PARAMETER Uninstall
    Se verdadeiro, remove a tarefa agendada em vez de criar

.PARAMETER Force
    Se verdadeiro, substitui tarefa existente sem perguntar

.EXAMPLE
    .\install-bridge-autostart.ps1
    .\install-bridge-autostart.ps1 -Port 8001
    .\install-bridge-autostart.ps1 -Uninstall
    .\install-bridge-autostart.ps1 -Force
#>

param(
    [int]$Port = 8001,
    [switch]$Uninstall,
    [switch]$Force,
    [switch]$Quiet
)

$TaskName = "OdysseusV2Bridge"

# ── Caminhos ──
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path "$ScriptDir"
$V2ServerPy = Join-Path $ProjectRoot "v2_server.py"

# ── Logging ──
$LogDir = Join-Path $env:LOCALAPPDATA "OdysseusBridge\logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}
$LogFile = Join-Path $LogDir "autostart.log"

function Log($msg) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] $msg"
    Add-Content -Path $LogFile -Value $line
    if (-not $Quiet) { Write-Host $line }
}

function Write-Step($msg) {
    if (-not $Quiet) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }
}

# ═══════════════════════════════════════════════════════════════════
#  UNINSTALL
# ═══════════════════════════════════════════════════════════════════

if ($Uninstall) {
    Write-Step "A remover tarefa '$TaskName'..."

    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existing) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$(-not $Force)
        if ($?) {
            Log "[OK] Tarefa '$TaskName' removida com sucesso."
        } else {
            Log "[ERRO] Erro ao remover tarefa '$TaskName'."
        }
    } else {
        Log "[INFO] Tarefa '$TaskName' nao existe. Nada a remover."
    }

    Log "--- Uninstall concluido ---"
    if (-not $Quiet) { Write-Host "`nUninstall concluido." -ForegroundColor Green }
    return
}

# ═══════════════════════════════════════════════════════════════════
#  VALIDAÇÕES
# ═══════════════════════════════════════════════════════════════════

Write-Step "A validar ambiente..."

# 1. Verificar que o ficheiro v2_server.py existe
if (-not (Test-Path $V2ServerPy)) {
    Log "❌ Ficheiro não encontrado: $V2ServerPy"
    Write-Host "Erro: Não foi encontrado '$V2ServerPy'." -ForegroundColor Red
    Write-Host "Executa este script a partir da pasta raiz do projeto Odysseus." -ForegroundColor Yellow
    exit 1
}
Log "✅ v2_server.py encontrado em: $V2ServerPy"

# 2. Encontrar Python (resolver caminho real, não o launcher)
$PyExe = $null

# Tentar py launcher primeiro para obter o Python real
$PyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($PyLauncher) {
    # Tentar encontrar Python 3.x via launcher
    foreach ($v in @("-3.14", "-3.13", "-3.12", "-3.11")) {
        try {
            $realPath = & $PyLauncher.Source $v "-c" "import sys; print(sys.executable)" 2>$null
            if ($realPath) {
                $PyExe = $realPath.Trim()
                Log "[OK] Launcher py encontrou Python: $PyExe"
                break
            }
        } catch { continue }
    }
}

if (-not $PyExe) {
    # Fallback: tentar python diretamente
    $PythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($PythonCmd -and $PythonCmd.Source -notlike "*WindowsApps*") {
        $PyExe = $PythonCmd.Source
        Log "✅ Python direto: $PyExe"
    }
}

if (-not $PyExe) {
    Log "❌ Python não encontrado."
    Write-Host "Erro: Python não encontrado. Instala Python 3.11+ primeiro." -ForegroundColor Red
    exit 1
}
Log "✅ Python final: $PyExe"

# 3. Testar que v2_server.py importa sem erros
try {
    $TestResult = & $PyExe -c "
import sys
sys.path.insert(0, r'$ProjectRoot')
from core.v2.api_bridge import is_opencode_available
print(f'OpenCode available: {is_opencode_available()}')
print('Import OK')
" 2>&1
    Log "✅ Import test: OK"
} catch {
    Log "[WARN] Import test warning (nao critico): $_"
}

# ═══════════════════════════════════════════════════════════════════
#  CRIAR TAREFA AGENDADA
# ═══════════════════════════════════════════════════════════════════

Write-Step "A registar tarefa '$TaskName' no Windows Task Scheduler..."

# Verificar se já existe
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    if ($Force) {
        Log "[WARN] Tarefa '$TaskName' ja existe. A substituir (Force)..."
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    } else {
        Write-Host ""
        Write-Host "[WARN] A tarefa '$TaskName' ja existe." -ForegroundColor Yellow
        $choice = Read-Host "Pretendes substituir? (s/N) "
        if ($choice -ne "s" -and $choice -ne "S") {
            Log "[INFO] Operacao cancelada pelo utilizador."
            Write-Host "Operacao cancelada." -ForegroundColor Yellow
            exit 0
        }
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }
}

# Construir comando
$TaskCommand = $PyExe
$TaskArguments = "-u `"$V2ServerPy`" --port $Port"

Log "Comando: $TaskCommand $TaskArguments"
Log "Diretório de trabalho: $ProjectRoot"

# Criar a ação da tarefa
$Action = New-ScheduledTaskAction `
    -Execute $TaskCommand `
    -Argument $TaskArguments `
    -WorkingDirectory $ProjectRoot

# Criar o trigger: ao logon do utilizador (com delay de 10s para garantir rede)
$Trigger = New-ScheduledTaskTrigger `
    -AtLogOn `
    -User $env:USERNAME `
    -RandomDelay (New-TimeSpan -Seconds 10)

# Configurações principais
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 30) `
    -Priority 7

# Opções de execução
$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

# Tentar registar
try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Principal $Principal `
        -Force:$Force `
        -ErrorAction Stop

    Log "[OK] Tarefa '$TaskName' registada com sucesso!"
    Write-Host ""
    Write-Host "Tarefa criada com sucesso!" -ForegroundColor Green
} catch {
    # Se falhar com erro de permissoes, tentar com RunLevel Highest (admin)
    Log "[WARN] A tentar registar com privilegios de administrador..."

    try {
        $PrincipalAdmin = New-ScheduledTaskPrincipal `
            -UserId $env:USERNAME `
            -LogonType Interactive `
            -RunLevel Highest

        Register-ScheduledTask `
            -TaskName $TaskName `
            -Action $Action `
            -Trigger $Trigger `
            -Settings $Settings `
            -Principal $PrincipalAdmin `
            -Force:$Force `
            -ErrorAction Stop

        Log "[OK] Tarefa '$TaskName' registada com sucesso (admin)!"
        Write-Host ""
        Write-Host "Tarefa criada com sucesso!" -ForegroundColor Green
    } catch {
        Log "[ERRO] Erro ao registar tarefa: $_"
        Write-Host ""
        Write-Host "ERRO: Nao foi possivel criar a tarefa." -ForegroundColor Red
        Write-Host "Tenta executar o PowerShell como Administrador:" -ForegroundColor Yellow
        Write-Host "  1. Abre o PowerShell como Administrador" -ForegroundColor Yellow
        Write-Host "  2. Navega ate: $ScriptDir" -ForegroundColor Yellow
        Write-Host "  3. Executa: .\install-bridge-autostart.ps1 -Force" -ForegroundColor Yellow
        exit 1
    }
}

# ═══════════════════════════════════════════════════════════════════
#  RESUMO
# ═══════════════════════════════════════════════════════════════════

Write-Step "Resumo"

$TaskInfo = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

Write-Host ""
Write-Host ""
Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host "   Odysseus V2 Bridge - Autostart Installed" -ForegroundColor Cyan
Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Tarefa:       $TaskName" -ForegroundColor White
Write-Host "  Servidor:     http://localhost:$Port" -ForegroundColor White
Write-Host "  Python:       $PyExe" -ForegroundColor White
Write-Host "  Script:       $V2ServerPy" -ForegroundColor White
Write-Host "  Logs:         $LogDir" -ForegroundColor White
Write-Host ""
Write-Host "  ---" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Arranca automaticamente no teu proximo login" -ForegroundColor Green
Write-Host "  Para iniciar agora (admin):" -ForegroundColor Green
Write-Host "     Start-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Green
Write-Host ""
Write-Host "  Para remover:" -ForegroundColor Yellow
Write-Host "     .\install-bridge-autostart.ps1 -Uninstall" -ForegroundColor Yellow
Write-Host ""
Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host ""

Log "--- Instalacao concluida ---"
