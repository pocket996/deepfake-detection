# 人脸伪造图片与视频检测

当前阶段：准备环境和数据。尚未实现检测模型。

## 本机环境

- Windows，Python 3.11 独立虚拟环境 `.venv`。
- 已验证配置：PyTorch 2.9.0 / torchvision 0.24.0，官方 CUDA 12.8 构建。
- 不修改系统 Python；基础环境不代表已兼容任何完整研究框架。

## 重建环境

GPU安装文件使用Windows x64 / Python 3.11官方直链，不能用于其他系统或Python版本。先安装 Python 3.11，在项目目录执行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/check_environment.py
```

环境检查必须通过真实GPU卷积前向/反向、图片读写、视频读写和FFmpeg检查。结果写入 `outputs/environment.json`。无NVIDIA显卡的队友需使用CPU版PyTorch，不要照搬GPU安装文件。项目使用虚拟环境中的完整Python路径，不要求修改PowerShell脚本执行策略。

数据步骤见 [docs/data-setup.md](docs/data-setup.md)。

队友没有NVIDIA显卡时，安装步骤中的 `requirements-gpu.txt` 替换为 `requirements-cpu.txt`，其余基础依赖相同。GPU检查脚本在CPU机器上会报告CUDA不可用，这是预期结果。CPU环境尚未在另一台电脑上验证。

完整安装版本记录在 `requirements-lock.txt`，该文件适用于本机Windows/Python 3.11/GPU环境。常规重建使用上述两个安装文件。

VS Code中选择解释器 `D:\deepfake-detection\.venv\Scripts\python.exe`。运行程序时始终使用该虚拟环境，不要将依赖安装到系统Python 3.13。

## 当前准备状态

- 硬件：RTX 5070 Laptop GPU，8151MiB显存；系统内存约16GB；初次检查D盘剩余约219GiB。
- Python 3.11.15虚拟环境已创建；图片、视频与评估基础依赖已安装。
- GPU依赖安装完成；真实GPU卷积前向/反向计算、图片读写、视频读写及FFmpeg检查全部通过，依赖一致性检查通过。详见 `outputs/environment.json`。
- FFmpeg已有安装。
- 数据尚未申请获批、尚未下载；项目成员将提交官方申请。
