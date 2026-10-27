# 主开发环境：本地训练、源码运行与团队重建

新加入项目的队友请先阅读 [一步步安装指南](TEAM_ENV_SETUP.md)，本文保留主环境结构、验收细节与云端部署说明。

根目录 `.venv` 是当前主开发环境，用于数据处理、模型训练、评估和源码界面。`reference_run` 暂时保留原型代码、数据与模型；其中的 `.venv` 是历史临时环境。代码目录与环境目录是两回事：用根目录解释器执行原型代码即可，后续移动代码时另行调整资源路径。

## 当前本机基线

- Windows x64、Python 3.11.15。
- PyTorch 2.9.0 + CUDA 12.8、torchvision 0.24.0。
- RTX 5070 Laptop，约 8 GB 显存；系统内存约 16 GB。
- 通用依赖包含 NumPy、OpenCV、Pillow、pandas、scikit-learn、tqdm 与 pyarrow。
- 系统 FFmpeg 可在命令行调用；本地桌面界面需要 Python 的 Tk 支持。

根目录 `requirements-lock.txt` 记录本机已安装版本，含 Windows 专用 PyTorch 地址。原型目录的锁定文件作为历史记录保留。

## 队友在 Windows 重建

以下命令均在仓库根目录执行。先安装 Python 3.11（桌面使用需带 Tcl/Tk）及系统 FFmpeg。

```powershell
py -3.11 -m venv .venv

# 有兼容的 NVIDIA 显卡及驱动：
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt

# 再安装通用数据处理与训练依赖：
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

没有 NVIDIA 显卡时，上面的 GPU 安装步骤替换为：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
```

GPU/CPU 安装文件二选一，再安装通用依赖。已有可用环境无需重复创建，更不需要为了与队友一致而覆盖已验证的 GPU 包。

若本机访问 PyTorch 索引出现 TLS 问题，可用 `requirements-gpu-windows.txt` 的官方 wheel 直链备用方案；它仅支持 Windows x64 / Python 3.11。CPU 机器可以运行模型和调试流程，完整训练耗时需单独评估。

## 验收环境，而不只是看安装成功

```powershell
# NVIDIA GPU 机器；明确要求 CUDA 可用，失败时不会静默改用 CPU：
.\.venv\Scripts\python.exe scripts/check_environment.py --device cuda

# CPU 使用者：
.\.venv\Scripts\python.exe scripts/check_environment.py --device cpu

# 已准备 FF++ 人脸清单和图片后，再做小批量训练验收：
.\.venv\Scripts\python.exe scripts/check_environment.py --device cuda --training-smoke --output outputs/environment-main-gpu.json

# 数据加载、推理、训练协议与视频审计的已有测试：
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

基础检查包含依赖一致性、所选设备的真实前向/反向计算、图片/视频读写、Parquet 读写和 FFmpeg。报告默认写入 `outputs/environment.json`，失败会返回非零退出码。`--device auto` 为默认值，有 CUDA 则用 CUDA，否则用 CPU；需要验证 GPU 时应显式使用 `--device cuda`。

`--training-smoke` 从 FF++ 清单的 **train** 划分各取两张真实/伪造人脸，以批量 4 执行两次 ResNet18 优化更新；GPU 模式使用混合精度，随后临时保存并重新加载模型，比较输出。使用随机初始化，不下载预训练权重，不改写稳定模型或数据，临时产物检查后清理。它证明小规模训练与模型读写能工作，不能当成模型质量或全量训练速度的结论。自定义清单可传 `--manifest`，格式参见 `src/faceguard/data.py`。

没有本地数据集的队友可以先运行基础检查与已有测试；缺少数据时训练验收会明确失败，需要先按数据文档准备文件。

## 运行当前源码

```powershell
.\.venv\Scripts\python.exe reference_run/gui.py
.\.venv\Scripts\python.exe reference_run/predict.py "C:\media\photo.jpg" --task generation --device cpu
.\.venv\Scripts\python.exe reference_run/train_generation.py --help
```

`reference_run/run_demo.cmd`、`reproduce.ps1` 及两个 VS Code 工作区均指向根目录主环境。`reproduce.ps1` 包含正式实验核验与报告生成，传 `-Train` 或 `-Generation` 会启动相应训练流程，不是环境检查命令。

源码推理仍需要模型和 YuNet 资源；模型准备方式见根目录 README。训练还需要合法来源的数据和清单。更换解释器不会自动迁移数据、模型，也不会把临时原型代码变成新的正式代码架构。

## 什么时候再使用 AutoDL？

当前本机已完成现有 ResNet18 级别实验，历史主要训练耗时约 9–13 分钟，日常编写代码、界面开发、推理和小规模实验可继续在本地进行。具体历史范围见 `reference_run/RESULTS.md` 与 `GENERATION_CURRENT.md`。

出现以下实际需求时再考虑租服务器：

- 目标模型、输入尺寸或视频序列在合理减小批量后仍超出本机显存。
- 一次完整实验耗时已明显影响进度，需要跑多组参数、随机种子或消融对照。
- 需要长时间连续训练，或本机内存、存储和使用时间成为限制。
- 团队需要共同使用一个训练实例，并明确管理数据、作业与结果。

先在本地跑通小批次，记录显存和单轮耗时，再估算完整实验预算、选择云端资源。服务器主要承担正式训练；代码和界面仍可在本地开发，训练所得权重与模型定义、预处理及配置一同带回验证。

## 为 Linux / AutoDL 保留的安装路径

这是待实机验证的部署说明，当前没有租用服务器，也没有完成 Linux 验收。尽量对齐 Python 3.11、PyTorch 2.9.0、torchvision 0.24.0 与通用依赖；根据 GPU 和驱动选择适用构建，不直接安装本机 Windows 锁定文件。

在提供 Python 3.11、兼容 NVIDIA 驱动的 Linux 环境中，可以从以下流程开始：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-gpu.txt
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/check_environment.py --device cuda
```

服务器还需提供 FFmpeg 及 OpenCV 所需系统库，数据和输出路径要按实例目录重新配置。原型的 Windows 启动器不在服务器上使用，训练使用 Python 命令行入口。完整部署仍需训练与保存/加载验收，当前源码并未完成跨平台承诺。

不要复制 `.venv` 文件夹给队友或服务器；共享代码、依赖文件、数据准备步骤与验收命令，在目标机器上重建。参考 [Python 虚拟环境说明](https://docs.python.org/3.11/library/venv.html)、[PyTorch 对应版本安装说明](https://pytorch.org/get-started/previous-versions/)、[AutoDL 快速开始](https://www.autodl.com/docs/quick_start/)。
