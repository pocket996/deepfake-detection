# 整脸生成检测分支：复现与阅读

[Current development environment setup](../docs/TEAM_ENV_SETUP.md). The commands below use the root .venv.

当前默认联合版见 [GENERATION_CURRENT.md](GENERATION_CURRENT.md)。第一版「只用 FLUX 训练、SD3.5 完全留出」的原始实验保留在 [GENERATION_RESULTS.md](GENERATION_RESULTS.md)，不能把其成绩与联合训练版混为一谈。这是独立于 FF++ 换脸/篡改模型的实验分支。

## 使用

关闭旧窗口，重新双击 `run_demo.cmd`。默认模式下：

- **照片 → AI 整脸生成检测**。左侧有真实照片和 AI 生成照片示例。
- **视频 → 换脸/面部篡改检测**。本次没有训练整段 AI 生成视频检测。
- 照片如果需要检测换脸/局部篡改，在界面上方「检测项目」选择「换脸 / 面部篡改」。此时示例按钮也会切换为原有视频示例。

结果中的“生成倾向分”与“篡改倾向分”来自两个不同模型，不可直接比较，也不是真假概率。没有人脸时返回“无法判断”。

```powershell
cd D:\faceguard\reference_run
..\.venv\Scripts\python.exe predict.py 'D:\你的照片.jpg'
..\.venv\Scripts\python.exe predict.py 'D:\你的照片.jpg' --task generation
..\.venv\Scripts\python.exe predict.py 'D:\你的照片.jpg' --task manipulation
..\.venv\Scripts\python.exe predict.py 'D:\你的视频.mp4'
```

`--task auto` 是默认值。显式传 `--checkpoint` 时，旧命令仍按原换脸模型解释；自定义整脸生成权重应同时传 `--task generation`。

个人回归图片不随仓库公开。如需额外检查自己的样本，使用 `verify_generation.py --experiment joint --user-image 'D:\你的照片.jpg'`。不传 `--user-image` 时跳过个人样本检查，仍可完成其余核验与报告。

## 数据、环境与隔离

沿用本目录独立 `.venv`，只新增 `pyarrow==23.0.1` 读取 Parquet；完整依赖已写入 requirements-lock.txt。所有新原始照片、处理结果和清单在 `reference_run/data`；模型与评估在 `artifacts/generation`。父目录的 FairFaceGen ZIP 只读取，原 FF++ 模型没有覆盖。

来源分别为 scikit-learn 固定校验值的 LFW、固定提交和 SHA256 的 FFHQ 公开镜像两个分片、已经下载并校验的官方 FairFaceGen 种子包。下载脚本不需要账户密钥。镜像没有完整身份与逐图许可元数据，未建立商用或图片再分发资格；数据不提交到 Git。

## 从数据到模型

```powershell
..\.venv\Scripts\python.exe download_lfw.py
..\.venv\Scripts\python.exe download_ffhq_subset.py
..\.venv\Scripts\python.exe prepare_generation.py
..\.venv\Scripts\python.exe train_generation.py
..\.venv\Scripts\python.exe calibrate_generation.py
..\.venv\Scripts\python.exe verify_generation.py
..\.venv\Scripts\python.exe report_generation.py
# 覆盖两种已知生成器的联合版；本版本 SD3.5 不再是未见生成器
..\.venv\Scripts\python.exe train_generation.py --include-sd35
..\.venv\Scripts\python.exe calibrate_generation.py --experiment joint
..\.venv\Scripts\python.exe verify_generation.py --experiment joint
..\.venv\Scripts\python.exe activate_generation.py
..\.venv\Scripts\python.exe gui.py --self-test --self-test-task generation
..\.venv\Scripts\python.exe gui.py --self-test --self-test-task manipulation
..\.venv\Scripts\python.exe report_generation_joint.py
```

父项目 `data/raw/fairfacegen` 下需有两个种子包；若缺少，先运行父项目 `scripts/download_fairfacegen.py`。当前已经具备。重跑 `prepare_generation.py` 会在本参考目录重建派生图片与清单，原始数据保留。

整脸生成模型训练 5 轮，每轮遍历全部生成任务的训练图片。ResNet18 从 ImageNet 权重初始化并全网微调，AdamW、骨干学习率 1e-4、分类层 3e-4，余弦调度，批量 48，CUDA 混合精度。损失依次平衡真假类别、真实照片来源和已知分组，避免 LFW 中照片多的人物主导训练。

**第一版单生成器实验**：FLUX 按种子分到 train/val/test；真实照片中 LFW 按人物名称、FFHQ 镜像按图片分组。该版本中 SD3.5 **不参与训练和调参**。只看验证集 AUC 选择轮次，只用验证集 balanced accuracy 选阈值。`calibrate_generation.py` 将阈值放在同一个最优验证决策区间的中点；验证分类保持不变，避免贴在最小生成分数上，整个过程不使用测试集或用户图片选择阈值。保存最终阈值后，再统一计算最终测试、SD3.5 和 JPEG 指标。初始阈值的原始结果保存在 pre_calibration_metrics.json，不隐藏该阶段。SD3.5 测试复用同一真实测试集，不能把这两个指标当成完全独立的实验。

用户提供的 GPT 图片在模型选定后才检查，未用于训练、调参，也没有针对其文件名或标签加规则。它是单张个例，不能用来宣称“支持所有 GPT 生成图”。

第一版发现未见 SD3.5 召回率低，因此追加独立联合版：对 SD3.5 使用同一固定分组哈希建立 train/val/test，与 FLUX 一起训练。原始全部留出实验保留，不把联合版的 SD3.5 测试包装成未见测试。两个版本都不使用用户图片训练或调阈值，但该 GPT 个例在第一版已被观察过，联合版再次检查属于重复回归。

联合版数据清单、权重、指标、阈值及回归结果在 `artifacts/generation/joint`。`activate_generation.py` 只在模型哈希与核验一致、且验证集 balanced accuracy 至少 0.90 时写入 READY.json，随后照片默认路由到联合版。该启用条件不根据测试集或 GPT 个例的结果决定。

## 关键文件

| 文件 | 作用 |
| --- | --- |
| generation_common.py | 训练与推理共用的 128px / JPEG90 标准化 |
| prepare_generation.py | 统一 YuNet 裁脸，分组划分，精确重复隔离，记录未检出人脸 |
| train_generation.py | 独立训练、验证集选模型、正式与未见生成器评估 |
| face_analysis.py | 整脸生成推理，以及照片/视频任务路由 |
| verify_generation.py | 原图与清单分数、CPU/GPU、无脸、跨来源真实帧、用户 GPT 个例检查 |
| artifacts/generation/metrics.json | 各划分指标、真实类误报率、JPEG 重压缩成绩与耗时 |
| artifacts/generation/user_gpt_regression.json | 用户图片的实际输出与原篡改分支对照 |
| data/generation_manifest.csv | 每张图片的原始来源、ZIP 成员、分组、划分、处理图路径和像素 SHA256 |

仍存在数据来源、人物和生成器覆盖限制：FFHQ 镜像无身份标签；未做生物身份或近重复审计。单生成器版本仅用 FLUX，当前联合版本用 FLUX 与 SD3.5，但均没有规模化 GPT 专项训练/测试。统一格式不能完全消除来源差异。高 AUC 和单张 GPT 个例结果都不能替代更广泛的独立测试。
