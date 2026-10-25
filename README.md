# AI 人脸生成与篡改检测系统

面向包含人脸的照片与视频，检测 AI 整脸生成、换脸及面部篡改。现已完成独立参考实现，包括数据整理、模型训练与评估、命令行推理和中文桌面界面。

## 最新参考实现

- [使用教程与源码阅读](reference_run/README.md)：独立的 `reference_run` 目录和 Python 环境。
- [整脸生成模型与实测结果](reference_run/GENERATION_CURRENT.md)：FLUX + SD3.5 联合训练，照片默认使用此分支。
- [换脸/面部篡改实验](reference_run/RESULTS.md)：FF++ 官方划分、视频级指标与微调对照，视频默认使用此分支。
- [生成检测复现步骤](reference_run/GENERATION_README.md)：数据获取、分组、训练、阈值校准与核验。
- `reference-source.code-workspace`：在 VS Code 中单独查看参考实现源码。

本机已配置好环境和模型时，双击 `reference_run/run_demo.cmd` 打开界面。照片若要查换脸，在顶部切换检测项目。整脸生成检测仍是实验版，不能把库内高分或单张 GPT 个例的结果视为对所有生成器有效。

**GitHub 仅保存源码、依赖版本和文档。** 数据集、虚拟环境、训练权重、特征缓存、日志和个人图片均未上传；新克隆的仓库需要按教程准备数据、环境和模型后才能运行演示。下文保留基础环境与原始数据准备说明。

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

FF++ 划分、抽帧裁脸及统一加载的运行方法见 [docs/ffpp-preparation.md](docs/ffpp-preparation.md)。5000 个视频的官方划分清单已建立；首批 30 个视频已生成 480 张人脸图并通过加载验证，全量预处理已启动。较新图片数据候选见 [docs/image-dataset-options.md](docs/image-dataset-options.md)。

队友没有NVIDIA显卡时，安装步骤中的 `requirements-gpu.txt` 替换为 `requirements-cpu.txt`，其余基础依赖相同。GPU检查脚本在CPU机器上会报告CUDA不可用，这是预期结果。CPU环境尚未在另一台电脑上验证。

完整安装版本记录在 `requirements-lock.txt`，该文件适用于本机Windows/Python 3.11/GPU环境。常规重建使用上述两个安装文件。

VS Code中选择解释器 `D:\deepfake-detection\.venv\Scripts\python.exe`。运行程序时始终使用该虚拟环境，不要将依赖安装到系统Python 3.13。

## 当前准备状态

- 硬件：RTX 5070 Laptop GPU，8151MiB显存；系统内存约16GB；初次检查D盘剩余约219GiB。
- Python 3.11.15虚拟环境已创建；图片、视频与评估基础依赖已安装。
- GPU依赖安装完成；真实GPU卷积前向/反向计算、图片读写、视频读写及FFmpeg检查全部通过，依赖一致性检查通过。详见 `outputs/environment.json`。
- FFmpeg已有安装。
- 数据范围：AI-Face 的 Real/DMs/GANs/deepfakes 图片和 FF++ 标准 c23 视频分别保存。FF++ 共 5000 个视频已下载完成（零失败，尚未全量解码），状态见 `outputs/ffpp-download-status.json`，官方划分已取得。AI-Face 浏览器可访问，但本地请求返回 403，尚未下载；整段 AI 生成视频的数据源尚待补充。详见 `docs/data-setup.md`。
