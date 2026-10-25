param([switch]$Train, [switch]$Generation)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$referencePython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $referencePython)) {
    throw '先按照 README.md 创建 reference_run/.venv 并安装 requirements-lock.txt。'
}
if ($Train) {
    & $referencePython train.py
    if ($LASTEXITCODE -ne 0) { throw '训练失败，请检查错误输出。' }
    & $referencePython holdout.py
    if ($LASTEXITCODE -ne 0) { throw '未见方法实验失败，请检查错误输出。' }
    & $referencePython finetune.py
    if ($LASTEXITCODE -ne 0) { throw '微调失败，请检查错误输出。' }
}
if ($Generation) {
    foreach ($generationScript in @('download_lfw.py', 'download_ffhq_subset.py', 'prepare_generation.py', 'train_generation.py', 'calibrate_generation.py')) {
        & $referencePython $generationScript
        if ($LASTEXITCODE -ne 0) { throw "整脸生成流程失败：$generationScript" }
    }
    & $referencePython train_generation.py --include-sd35
    if ($LASTEXITCODE -ne 0) { throw '联合整脸生成模型训练失败。' }
    & $referencePython calibrate_generation.py --experiment joint
    if ($LASTEXITCODE -ne 0) { throw '联合模型阈值校准失败。' }
}
& $referencePython verify.py
if ($LASTEXITCODE -ne 0) { throw '验证失败，请检查错误输出。' }
& $referencePython verify.py --variant finetuned
if ($LASTEXITCODE -ne 0) { throw '微调模型验证失败，请检查错误输出。' }
& $referencePython gui.py --smoke-test
if ($LASTEXITCODE -ne 0) { throw '桌面界面检查失败。' }
& $referencePython make_report.py
if ($LASTEXITCODE -ne 0) { throw '结果报告生成失败。' }
if (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'artifacts/generation/metrics.json')) {
    & $referencePython verify_generation.py
    if ($LASTEXITCODE -ne 0) { throw '整脸生成模型核验失败。' }
    & $referencePython report_generation.py
    if ($LASTEXITCODE -ne 0) { throw '单生成器报告生成失败。' }
    if (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'artifacts/generation/joint/metrics.json')) {
        & $referencePython verify_generation.py --experiment joint
        if ($LASTEXITCODE -ne 0) { throw '联合整脸生成模型核验失败。' }
        & $referencePython activate_generation.py
        if ($LASTEXITCODE -ne 0) { throw '联合模型未通过启用检查。' }
        & $referencePython report_generation_joint.py
        if ($LASTEXITCODE -ne 0) { throw '联合整脸生成报告生成失败。' }
    }
    & $referencePython gui.py --self-test --self-test-task generation
    if ($LASTEXITCODE -ne 0) { throw '整脸生成界面流程核验失败。' }
    & $referencePython gui.py --self-test --self-test-task manipulation
    if ($LASTEXITCODE -ne 0) { throw '换脸/篡改界面流程核验失败。' }
}
