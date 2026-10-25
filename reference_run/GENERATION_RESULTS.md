# AI 整脸生成检测：数据、实验与接入结果

本次新增独立的整脸生成检测分支，保留原有 FF++ 换脸/面部篡改模型。默认照片进入生成检测，默认视频进入篡改检测；需要检查照片换脸时，可在界面上方切换「检测项目」。两个模型的分数不混成一个“真假概率”。

## 数据缺口如何处理

原来已有 FairFaceGen 的 FLUX 与 SD3.5 生成脸，缺少配套真实照片和整脸生成训练/评估。此次在 `reference_run/data` 独立保存并处理：

- LFW 13,233 张真实照片，通过 scikit-learn 固定的 SHA256 校验，按人物名称分组划分。
- FFHQ-256 公开镜像的两个分片，共 8,750 张照片，固定镜像提交版本与 LFS SHA256。它是发布者标注的 FFHQ 镜像，不是 NVIDIA 官方分发；没有附完整身份、作者和逐图许可元数据。
- FairFaceGen：FLUX 11,200 张用于分组划分；SD3.5 原始 11,200 张全部预留作未见生成器测试。
- 共 44,383 张候选图，成功裁脸并纳入 43,521 张；862 张 SD3.5 未检出人脸，返回“无法判断”，未视为真实，也未纳入有脸样本 AUC。SD3.5 人脸检测覆盖率为 92.30%。

| 划分 | 方法 | 来源 | 图片数 |
| --- | --- | --- | ---: |
| heldout_generator | SD35 | FairFaceGen | 10,338 |
| test | FLUX | FairFaceGen | 1,670 |
| test | real | FFHQ-256-mirror | 1,305 |
| test | real | LFW | 1,760 |
| train | FLUX | FairFaceGen | 7,905 |
| train | real | FFHQ-256-mirror | 6,102 |
| train | real | LFW | 9,696 |
| val | FLUX | FairFaceGen | 1,625 |
| val | real | FFHQ-256-mirror | 1,343 |
| val | real | LFW | 1,777 |

所有来源统一使用 YuNet 最大人脸、1.3 倍裁剪范围、128×128 与 JPEG90 编码；训练时再做两类相同的翻转、颜色、缩放、轻度模糊和 JPEG 扰动，降低格式捷径。统一编码不能消除来源差异，因此高分仍可能包含数据来源线索。

LFW 按已知人物分组，生成脸按种子编号分组；FFHQ 镜像按图片分组。已检查精确像素重复和分组跨划分；**未完成 FFHQ 的生物身份或近重复审计，不能声称全部身份互斥**。FairFaceGen 及镜像未附完整适用条款；当前是本地研究实验，没有确认商用或图片再分发资格。来源不明的第三方压缩包没有用于训练。

## 实际成绩

固定训练 5 轮，只按验证集 AUC 选择模型，并只在验证集确定阈值；选中第 3 轮，阈值 0.662368。阈值取最优验证决策区间中点，验证分类保持不变；此规则不使用测试集或用户图片，详见 threshold_selection.json。初始观察点阈值的结果另存 pre_calibration_metrics.json，随后按锁定的中点阈值统一报告最终测试成绩。训练、验证均不包含 SD3.5，也不包含用户 GPT 图片。生成模型从 ImageNet 初始化，未修改 FF++ 模型权重。

| 评估 | 图片数 | AUC | balanced accuracy | 生成图片召回率 |
| --- | ---: | ---: | ---: | ---: |
| 验证：FLUX + 真实 | 4,745 | 1.0000 | 100.00% | 100.00% |
| 测试：FLUX + 独立真实 | 4,735 | 1.0000 | 100.00% | 100.00% |
| 未见生成器：SD3.5 + 同一真实测试集 | 13,403 | 0.9508 | 59.02% | 18.03% |
| 测试集额外 JPEG60 重压缩 | 4,735 | 1.0000 | 99.97% | 99.94% |

未见生成器测试复用了同一真实测试集，不能与 FLUX 测试行相加当作独立真实样本。AUC 只统计通过人脸检测的样本，不覆盖所有输入图片。

| 评估 | 真实照片来源 | 数量 | 误报为生成的比例 |
| --- | --- | ---: | ---: |
| test | LFW | 1760 | 0.00% |
| test | FFHQ-256-mirror | 1305 | 0.00% |
| heldout_generator | LFW | 1760 | 0.00% |
| heldout_generator | FFHQ-256-mirror | 1305 | 0.00% |

额外在完全未参与生成模型训练的 FF++ 真实测试视频中，每段取一张现有人脸裁剪图，共 140 张，误报率 **0.00%**。这是只有真实类的跨来源压力检查，不是完整跨数据集 AUC，也不验证整段生成视频。

模型训练与正式评估耗时 12.19 分钟（不含导入、下载及数据准备），PyTorch 峰值 allocated 显存 0.74 GiB。完整数据整理耗时 10.86 分钟。

## 你提供的 GPT 图片

样本：用户提供的 GPT 肖像图片（个人路径和原图未公开）。生成来源由用户提供，未独立确定具体 GPT 版本。

**新整脸生成模型识别为「疑似 AI 整脸生成」。**

- 生成分数：0.846742，验证集锁定的阈值：0.662368。
- 该图片没有参与训练、验证、阈值选择或模型选择；也已核对其标准化像素不在数据清单里。
- 原模型是换脸/篡改分支，这个个例的问题属于另一个任务；新增分支没有把输入文件名或用户标签作为判断规则。
- 单张 GPT 图片的成功或失败，均不能代表对所有 GPT 图片的准确率。当前也没有 GPT 专门训练集或成规模的 GPT 独立测试集。

完整原始输出见 `artifacts/generation/user_gpt_regression.json`。没有以压低阈值或将该图片塞进训练集的方式让它通过。

## 如何使用和查看代码

重新打开 `run_demo.cmd`。默认照片检测 AI 整脸生成，视频检测换脸/面部篡改；照片要查换脸时，在上方「检测项目」选择「换脸 / 面部篡改」。左侧可直接载入真实照片与 AI 生成照片示例。

代码阅读顺序：`generation_common.py`（统一预处理）→ `prepare_generation.py`（来源及划分）→ `train_generation.py`（独立训练）→ `face_analysis.py`（模型和路由）→ `verify_generation.py`（含 GPT 个例的训练外核验）→ `gui.py`（界面）。

复现命令见 `GENERATION_README.md`。原换脸模型及其结果仍在 `RESULTS.md`，不能将两个任务的成绩互相替代。当前模型是实验版；要更可靠地检测 GPT 等新工具，仍需补充多个生成器、更多真实来源、身份/近重复审计以及成规模的独立 GPT 测试。

来源：[LFW / scikit-learn](https://scikit-learn.org/stable/datasets/real_world.html#the-labeled-faces-in-the-wild-face-recognition-dataset)、[FFHQ 官方说明](https://github.com/NVlabs/ffhq-dataset)、[使用的 FFHQ 镜像](https://huggingface.co/datasets/bitmind/ffhq-256)、[FairFaceGen](https://zenodo.org/records/17062365)。
