# FaceGuard 队友环境安装指南

本指南用于在自己的 Windows 电脑上运行和修改 FaceGuard（慧眼辨伪）源码，以及准备后续模型训练环境。项目包含 AI 整脸生成与换脸 / 面部篡改检测，仓库名称统一为 `faceguard`。

如果只需要体验照片、视频检测，直接下载 [Windows 免安装预览版](https://github.com/pocket996/faceguard/releases/tag/v0.1.0)，完整解压后运行 `FaceGuard.exe`，不需要完成下面的开发环境安装。

## 1. 先确定安装哪一种环境

| 你的电脑与任务 | 选择 |
| --- | --- |
| Windows 10/11 x64，有兼容的 NVIDIA 显卡，参与模型训练 | GPU 环境 |
| Windows 10/11 x64，没有 NVIDIA 显卡，主要修改代码、运行检测或调试 | CPU 环境 |
| macOS、Windows ARM 或 Linux / AutoDL | 不照搬本指南；参考 [环境总说明](ENVIRONMENT.md)，这些平台尚未完成实机验收 |

CPU 和 GPU 使用相同的项目代码与通用依赖，主要区别是 PyTorch 安装包与计算设备。CPU 也可做小规模训练调试，但完整训练速度需要单独评估。

**每个人都在自己的电脑上创建根目录 `.venv`。不要复制其他人的 `.venv` 文件夹，也不要把依赖安装到系统 Python。**

## 2. 安装三个基础工具

### Git：获取和更新代码

从 [Git 官方网站](https://git-scm.com/downloads/win) 安装 Windows 版。安装完成后，重新打开 PowerShell，执行：

```powershell
git --version
```

显示版本号即可。

### Python 3.11：运行项目程序

安装 **Python 3.11、64 位**，保留安装选项中的 **pip、Tcl/Tk 和 Python Launcher**。安装来源及 Windows 配置见 [Python 官方说明](https://docs.python.org/3.11/using/windows.html)。已有其他版本的 Python 可以保留，本项目明确使用 3.11。

重新打开 PowerShell，执行：

```powershell
py -3.11 --version
```

应显示 `Python 3.11.x`。小版本号不必与开发者电脑完全相同，不要因为默认 `python` 指向其他版本就直接使用它。

### FFmpeg：视频处理与环境验收

从 [FFmpeg 官方下载入口](https://ffmpeg.org/download.html) 选择 Windows 构建。解压后找到包含 `ffmpeg.exe` 的 `bin` 文件夹，将该文件夹加入 Windows 的**用户 Path** 环境变量。

例如实际位置是 `D:\tools\ffmpeg\bin\ffmpeg.exe`，应加入 Path 的是 `D:\tools\ffmpeg\bin`。不要将 ZIP 压缩包或 EXE 文件本身加入 Path。

重新打开 PowerShell，执行：

```powershell
ffmpeg -version
```

显示版本信息即可。若使用其他 FFmpeg 发行包，版本号可能与开发者不同，最终以项目的视频读写检查为准。

## 3. 获取源码并创建主环境

下面使用 `D:\faceguard` 举例，可以选择其他磁盘。当前安装指南及配套环境改动发布在 `codex/team-environment-setup` 分支，合并到 `main` 前请明确克隆该分支。先进入准备存放项目的父目录，再执行：

```powershell
git clone --branch codex/team-environment-setup https://github.com/pocket996/faceguard.git
cd faceguard
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe --version
```

后续所有命令都在项目根目录执行。根目录应能看到 `README.md`、`requirements.txt`、`scripts` 和刚创建的 `.venv`。

已有旧名称的仓库时，不必重新下载：先关闭占用项目的程序，根据需要将本地文件夹改名，再在仓库中更新远程地址：

```powershell
git remote set-url origin https://github.com/pocket996/faceguard.git
git remote -v
```

旧目录下的虚拟环境不应直接作为改名后的可迁移环境使用，应在新位置重新创建。环境目录可以重建，源码、数据与模型请保留。

## 4. 安装 PyTorch：CPU / GPU 二选一

### 有 NVIDIA 显卡：GPU 环境

先执行 `nvidia-smi`，确认系统能识别显卡。再运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

项目目前对齐 PyTorch 2.9.0、torchvision 0.24.0 和 CUDA 12.8 构建，需要相容的显卡和驱动。不要只因为 NVIDIA 显卡存在就认定该组合一定支持；安装后必须通过下文的 GPU 计算检查。

若官方索引下载发生 TLS / SSL 错误，可使用针对 Windows x64 / Python 3.11 的官方直链备用文件：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu-windows.txt
```

### 没有 NVIDIA 显卡：CPU 环境

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
```

**CPU 与 GPU 的安装步骤选一个即可。** 不要依次安装两种 PyTorch。

## 5. 安装通用依赖并验收

两类环境都需要执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

`pip check` 应显示 `No broken requirements found.`。随后根据所选设备执行一条验收命令：

```powershell
# GPU 环境：
.\.venv\Scripts\python.exe scripts/check_environment.py --device cuda

# CPU 环境：
.\.venv\Scripts\python.exe scripts/check_environment.py --device cpu
```

检查包含实际前向 / 反向计算、图片和视频读写、Parquet 数据读写、FFmpeg 与依赖一致性。结果写入 `outputs/environment.json`，最终应为 `"status": "passed"`，且 `checks` 中各项均为 `passed`。

再运行已有代码测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试应以 `OK` 结束。基础验收与已有测试不需要下载完整训练数据。新电脑尚未通过这些检查前，不应认为开发环境已经安装完成。

## 6. 环境好了，还能不能直接检测和训练？

**运行环境、模型权重和训练数据是三样不同的东西。Git 克隆只带来源码和依赖说明。**

| 目标 | 下一步 |
| --- | --- |
| 修改代码、运行已有测试 | 环境验收通过即可开始 |
| 用源码检测自己的照片 / 视频 | 按 [README 的模型准备步骤](../README.md) 放好两个模型、YuNet 与就绪标记 |
| 训练换脸 / 篡改模型 | 按 [数据说明](data-setup.md) 和 [FF++ 预处理说明](ffpp-preparation.md) 准备合法来源的数据与清单 |
| 训练整脸生成模型 | 按 [生成检测复现说明](../reference_run/GENERATION_README.md) 准备数据 |

源码推理需要以下四个资源文件。可从 v0.1.0 解压后的 `FaceGuard` 文件夹中，按相同相对路径复制到项目的 `reference_run` 目录：

```text
reference_run/weights/face_detection_yunet_2023mar.onnx
reference_run/artifacts/model_finetuned.pt
reference_run/artifacts/generation/joint/model.pt
reference_run/artifacts/generation/joint/READY.json
```

模型准备完成后，在根目录运行：

```powershell
.\.venv\Scripts\python.exe reference_run/gui.py
```

也可以双击 `reference_run/run_demo.cmd`。当前启动器使用根目录 `.venv`；`reference_run/.venv` 是历史临时环境，新队友不需要创建它。

已准备好 FF++ 处理后图片与清单时，可进一步检查一次小批量训练及模型读写：

```powershell
# CPU 环境将 cuda 换成 cpu：
.\.venv\Scripts\python.exe scripts/check_environment.py --device cuda --training-smoke
```

这只执行两次优化更新，使用临时模型，不覆盖稳定权重，也不代表正式训练成绩。

## 7. 在 VS Code 选择正确解释器

打开根目录的 `faceguard.code-workspace`，或通过“Python: Select Interpreter”选择项目根目录 `.venv\Scripts\python.exe`。只阅读原型源码时也可打开 `reference-source.code-workspace`，它同样使用主环境。

确认解释器路径属于你自己电脑上的 `faceguard\.venv`。终端命令统一使用本文中的完整解释器相对路径，不需要激活脚本或修改 PowerShell 执行策略。

## 8. 常见错误

| 现象 | 处理方法 |
| --- | --- |
| 找不到 `py` 或提示没有 Python 3.11 | 检查 Python 3.11 与 Launcher 是否安装，重新打开终端 |
| 找不到 `requirements.txt` | 当前不在仓库根目录；先 `cd` 到包含 README 的文件夹 |
| `No module named ...` | 确认使用根目录 `.venv` 的解释器，并用它的 `-m pip` 安装对应依赖 |
| CUDA 不可用 | 检查 `nvidia-smi`、显卡/驱动支持，以及是否误装 CPU 版 PyTorch；无 NVIDIA 显卡应走 CPU 路径 |
| 找不到 FFmpeg | 检查加入用户 Path 的是否为包含 `ffmpeg.exe` 的 bin 目录，并重新打开终端 |
| 模型不存在、就绪标记不存在 | 环境安装不会自动带来模型；按 README 准备资源 |
| 训练验收找不到图片或清单 | 尚未准备本机训练数据；先完成基础验收和代码测试 |
| 下载中断或超时 | 网络恢复后重跑同一条安装命令；GPU 可尝试上述官方直链备用文件 |

反馈问题时，提供出错命令、完整错误文字、CPU/GPU 安装路线和 `outputs/environment.json`。不需要附个人照片或整个 `.venv` 文件夹。

更完整的本地训练说明及 AutoDL 使用时机，见 [环境总说明](ENVIRONMENT.md)。
