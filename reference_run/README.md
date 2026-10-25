# FaceGuard：从现有数据到可运行模型的独立参考实现

**新增 AI 整脸生成检测分支**：使用方式与当前联合模型实测结果分别见 [GENERATION_README.md](GENERATION_README.md)、[GENERATION_CURRENT.md](GENERATION_CURRENT.md)。第一版跨生成器实验另存 GENERATION_RESULTS.md。现在默认照片检测整脸生成，视频检测换脸/面部篡改；照片要查换脸时，在界面上方切换「检测项目」。下文记录原 FF++ 第一阶段，数据状态以新增分支报告为准。

**队友直接使用**：推荐下载 [Windows CPU 免安装预览版](https://github.com/pocket996/deepfake-detection/releases/tag/v0.1.0)，解压后双击 FaceGuard.exe。[操作说明](../docs/TEAM_QUICKSTART.md)。

**Git 源码快照说明**：源码树不包含本机 `.venv`、数据集、模型权重、特征、运行日志及个人测试图片；Release 附件另包含便携运行时与必要推理模型。下文“已经准备好”描述的是实测机器；新机器若需要源码训练，需先按本文及 GENERATION_README.md 重建环境和准备数据。YuNet 权重的获取方法见父目录 docs/ffpp-preparation.md。

这是供学习和复现的**人脸换脸/篡改检测基线**。包含数据审计、真实训练、参数选择、视频级测试、未见方法实验、图片/视频推理和本地桌面演示。准确结果和实际耗时见 [RESULTS.md](RESULTS.md)。

采用 ImageNet 预训练 ResNet18，完成两种真实训练：先冻结骨干训练逻辑回归分类层作为对照，再全网微调。桌面演示默认使用微调模型；两个模型和全部结果分别保留。**没有训练时序模型**，视频使用逐帧检测后取平均；不能把它当作覆盖所有 AI 图片和视频的成品。

## 1. 隔离方式

- 独立代码：`D:\deepfake-detection\reference_run`。
- 独立虚拟环境：本目录 `.venv`。依赖是原环境的物理副本，已经运行 pip check；不是 `--system-site-packages`，没有链接回原环境。
- 独立权重、特征、模型、结果：本目录 `weights`、`artifacts`。
- 原始数据及已处理图片只读取，不重新复制约 8.4 GiB 视频，也不改动原项目代码、数据、环境、输出。
- Python 3.11 基础解释器和 NVIDIA 驱动由两个虚拟环境共享，这是普通 venv 的工作方式。包安装互相隔离，不是虚拟机。
- 代码仅依赖本目录的 Python 模块；训练默认读取父项目已有的 FF++ 数据清单。移动目录时需要一起保留数据目录结构或相应调整 `core.py` 的 `PROJECT`。

打开 PowerShell，执行：

```powershell
cd D:\deepfake-detection\reference_run
.\.venv\Scripts\python.exe -m pip check
```

不需要激活环境，也不需要改 PowerShell 执行策略。不要用裸 `python` 或 `pip`，以免进入系统 Python 3.13。

## 2. 先体验已经训练好的模型

双击 `run_demo.cmd`，或：

```powershell
.\.venv\Scripts\python.exe gui.py
```

界面按下面三步操作：

1. **选文件**：左侧点「选择照片」或「选择视频」。第一次可以直接点「载入真实照片示例」或「载入 AI 生成示例」，中间会显示画面预览。在上方选择「换脸 / 面部篡改」后，示例按钮会切换为视频示例。
2. **开始检测**：确认预览后，点左侧蓝色「开始检测」。右侧会显示模型加载、视频读取、人脸分析等进度。首次加载可能需要几十秒。
3. **看结果**：右侧显示「疑似人脸篡改」「未检出明显篡改」或「未找到可分析的人脸」。可点击「保存检测报告」导出中文文本；模型阈值和逐帧分数在「查看技术详情」中。

换文件时重新从第一步开始。示例按钮只负责载入文件，还需要点击「开始检测」。程序只在本机运行。照片检测最大人脸；视频均匀采样最多 16 帧，每帧检测最大人脸。没有检测到人脸会明确返回“无法判断”。长视频局部篡改可能落在采样间隔内；多脸画面也可能遗漏目标。

命令行：

```powershell
.\.venv\Scripts\python.exe predict.py 'D:\你的文件夹\照片.jpg' --task manipulation
.\.venv\Scripts\python.exe predict.py 'D:\你的视频.mp4' --output artifacts/my_prediction.json
```

已经裁好的人脸图片可加 `--face-crop`；普通照片不要加。分数越高代表模型倾向于篡改，但分数没有经过概率校准，`0.9` 不等于“90% 概率为假”。图片和视频使用各自在验证集选出的阈值；图片阈值来自视频帧，不是独立照片验证集。

## 3. 从头复现模型训练

前置数据已经完成：5,000 段 FF++ c23 视频 → 统一 YuNet 裁脸 → 79,994 张 PNG。原项目 `scripts/prepare_ffpp_manifest.py`、`scripts/prepare_ffpp_faces.py` 以及 `docs/ffpp-preparation.md` 记录下载、官方划分和裁脸过程。这次读取其真实产物，未重新下载或重做裁脸。

```powershell
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe holdout.py
.\.venv\Scripts\python.exe finetune.py
.\.venv\Scripts\python.exe verify.py
.\.venv\Scripts\python.exe verify.py --variant finetuned
.\.venv\Scripts\python.exe make_report.py
```

`train.py` 首次下载约 45 MB 官方 ImageNet 权重到本目录；随后遍历所有图片，核对路径、来源分组、重复编号，并计算解码后像素的 SHA256，遇到跨划分的精确重复会停止。它不能检测近重复、同人不同照或跨库身份泄漏。特征为冻结网络的 FP32 输出，保存在 `artifacts/features.npy`，约 156 MiB。

流程：

1. **train** 拟合特征标准化器，并训练分类层；每个视频在其类别内权重相同，真假两类总权重相同。
2. **val** 比较四个正则化参数 `C = 0.001, 0.01, 0.1, 1`，按视频 AUC 选择；最大化验证集 balanced accuracy 选择阈值。
3. 锁定模型、参数和阈值。
4. **test** 只做最终测试：先取一个视频内各帧分数均值，再计算视频指标。另报告帧指标和每种篡改方法对比真实视频的指标。
5. 将标准化器精确折叠进 PyTorch 线性层，检查 sklearn 与导出模型数值一致。

然后 `finetune.py` 用同一个 ImageNet 骨干重新初始化分类层，训练全部网络参数，使用 AdamW、水平翻转和 CUDA 混合精度。固定运行 4 轮，每轮每段训练视频取 4 张交错帧，四轮合计覆盖所有现有训练帧；这不等于每轮使用全部 16 帧。每轮都在完整验证集评估，按视频 AUC 保存最佳轮次，再确定阈值，最后评估测试集。骨干学习率 1e-4、分类层 3e-4，余弦调度；真假损失按类别数量平衡。

微调模型为 `artifacts/model_finetuned.pt`，其结果文件以 `finetuned_` 开头；演示和命令行默认优先用它。若要比较冻结骨干对照，给 `predict.py` 加 `--checkpoint artifacts/model.pt`。微调也没有使用 FairFaceGen、来源不明数据或外部测试库。

官方划分：train 3,600 段、val 700 段、test 700 段，按 500 对来源编号分组。类别是真实 1,000 段、四类篡改各 1,000 段。因此“全部预测为假”也能拿到 80% accuracy；阅读结果时应同时看 AUC、balanced accuracy、混淆矩阵和真实类误报。

再次运行会复用清单哈希匹配的特征缓存，重新训练分类层。若修改了骨干权重、图像变换或特征提取实现，应先将 `artifacts/features.npy` 和 `features.json` 改名保存，再重新提取；不要复用旧缓存。当前代码检查清单哈希，不会自动检查每个源文件是否在上次运行后被篡改。

训练后生成 `model.pt`（骨干 + 分类层 + 两种阈值 + 预处理信息），推理时不需要 sklearn 或特征缓存。`verify.py` 在 CPU/GPU 比较分数，并直接读取测试原始视频，验证重新裁脸后的推理与测试清单输出一致。批量 GPU 卷积与单图推理存在浮点差异，图片/视频分数比较容差为 0.001，实际偏差记录在 verification 文件里；接近阈值的输入判定可能随数值扰动变化。相关机制见 [PyTorch 数值精度说明](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)。

## 4. 未见方法测试

`holdout.py` 排除 train 和 val 中所有 NeuralTextures 视频，只在其余三种伪造和真实视频上训练/调参。最后测试集仅取 NeuralTextures 和真实视频，各 140 段。

结果保存在 `holdout_metrics.json`，另存 `model_holdout.pt`，不替换主模型。这是同一个 FF++ 数据集内的方法留出，**不是跨数据集测试**，也不能说明能识别 FLUX、SD3.5 或整段 AI 生成视频。

## 5. 需要准备的内容

| 目标 | 当前数据是否够用 | 尚缺什么 |
| --- | --- | --- |
| 学习换脸/面部篡改完整流程 | 够用 | 当前参考实现已完成模型侧闭环 |
| 比较更强的换脸模型 | 够做库内实验 | 微调、多个随机种子、压缩与缩放扰动测试 |
| 对陌生来源视频证明泛化 | 不够 | 独立数据集，例如申请 Celeb-DF / DFDC，并固定外部测试协议 |
| 检测 AI 整脸生成照片 | 实验版配套已补齐 | 已补真实照片并完成独立生成检测分支；仍缺多版本 GPT 等新工具的大规模外部测试，详见 GENERATION_CURRENT.md |
| 检测整段 AI 生成视频 | 不够 | 来源明确的生成视频、匹配真实视频和独立测试；FF++ 不能替代 |
| 公开发布或商用 | 当前没有验证这个条件 | 确认各数据和权重的使用、再分发条款 |

第三方来源不明的图片/视频未用于本实验。AI-Face 目录仍为空。FairFaceGen 的下载完成不代表已获得明确的数据使用条款；原项目文档也记录了这一待确认事项。

建议先把“换脸/人脸篡改”作为第一阶段交付范围，再单独做整脸生成，最后考虑联合模型。不要把“人脸帧和生成头像来源不同”导致的高分当成检测能力。

## 6. 算力和时间安排

本机 RTX 5070 Laptop 8GB、约 16GB 内存，足以完成这版冻结骨干基线，也能在小批量和混合精度下尝试 ResNet18 微调。**目前无需购买云服务器**。需要更多显存、大模型、大量消融实验或长时间连续训练时，再考虑按小时租 GPU；先用本机测一次吞吐再预算，不必先买服务器。

实际本次耗时和显存见 RESULTS.md；以下是计划估算，不是已测结果：

| 工作 | 预计时间 | 条件 |
| --- | --- | --- |
| 运行现成演示 | 几分钟 | 环境与模型已经准备好 |
| 重建 GPU 环境 | 10–60 分钟 | 需要下载数 GB 依赖，网络慢会更久 |
| 复现冻结骨干模型 | 5–30 分钟 | 已有全部处理图；首次需要提取特征 |
| 从视频重新抽帧裁脸 | 约 0.5–2 小时 | 当前已有产物，无需再做；历史日志约半小时完成 |
| 从零下载 FF++ | 数十分钟到数小时，可能更久 | 约 9 GB；申请与跨境网络等待另计 |
| ResNet18 全网微调一轮实验 | 本次约 9.4 分钟；扩大训练预算粗估 0.5–2 小时 | 本次 4 轮、每轮 4 帧/视频；扩大到 5–15 轮且每轮 16 帧时需重新测速 |
| 你读懂并独立复现第一阶段 | 约 1–2 周业余时间 | 每天 1–3 小时，熟悉 Python 会更快 |
| 补齐生成照片、外部测试、对比实验与报告 | 约 4–8 周 | 数据申请/获取等待可能成为主要耗时 |

D 盘创建环境后仍剩约 179 GiB；无需全量再下载一个超大数据集。新环境有数 GB，数据缓存和权重另占几百 MB；本次磁盘实测见 `artifacts/storage.json`。

## 7. 推荐阅读顺序

先运行演示 → 看 RESULTS.md → 看 `core.py` 的预处理与推理 → 看 `train.py` 的 train/val/test 分工 → 看 `finetune.py` 如何更新骨干 → 看 `holdout.py` → 运行 `verify.py`。每次只改一件事，保留同一划分，并用验证集决定是否接受。留出方法实验目前只做了冻结骨干版，不能把微调主模型的成绩当成微调模型的未见方法成绩。

重建环境时先用 Python 3.11 创建本目录 `.venv`，再使用该环境的 `python.exe -m pip install -r requirements-lock.txt`。这个锁定文件适用于 Windows x64 / Python 3.11 / CUDA 12.8，CPU 或 Linux 不要直接照搬其中的 Windows GPU wheel 地址。当前参考目录内的 `.venv` 已经建好，无需重复安装。

官方资料：[FF++ 数据与划分](https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md)、[Torchvision ResNet18](https://docs.pytorch.org/vision/0.24/models/generated/torchvision.models.resnet18.html)、[YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)、[FairFaceGen 发布页](https://zenodo.org/records/17062365)。
