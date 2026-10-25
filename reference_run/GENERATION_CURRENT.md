# 当前整脸生成检测版本：FLUX + SD3.5 联合训练

已新增并接入独立的 AI 整脸生成检测。照片默认使用本联合模型，视频仍使用原 FF++ 换脸/面部篡改模型；照片要检查换脸时，在界面上方「检测项目」切换。两个任务不会混成一个分数。

## 数据是否补齐

**用于实验版训练的配套数据已补齐；GPT 专项训练和成规模独立测试仍缺少。**

本次处理了 LFW 13,233 张真实照片、FFHQ-256 镜像两个分片的 8,750 张真实照片，以及 FairFaceGen 的 FLUX / SD3.5 各 11,200 张生成照片。共 44,383 张候选，43,521 张通过同一 YuNet 裁脸与格式标准化；862 张 SD3.5 未检出人脸，算作无法判断，不标真实。新数据位于 reference_run/data，原数据和 FF++ 权重没有覆盖。

LFW 校验 scikit-learn 发布的 SHA256；FFHQ 镜像固定提交和 LFS SHA256；FairFaceGen 校验官方 MD5。FFHQ 镜像并非 NVIDIA 官方发布，没有完整身份和逐图原始许可元数据；未做生物身份、近重复去重或商用/再分发清理。因此这仍是可审计来源的研究原型，不是已经完成所有数据审查的生产基准。

## 为什么训练第二版

先完成了「FLUX 训练，SD3.5 完全留出」的实验。其未见 SD3.5 AUC 为 0.9508，但锁定阈值后的生成召回率只有 **18.03%**，说明排序能力和可用的判定阈值不是一回事，FLUX 的高分不能代表跨工具泛化。

第一版模型、预测、阈值和报告全部保留在 artifacts/generation 与 GENERATION_RESULTS.md。随后将 SD3.5 按预先固定的分组哈希划成 train/val/test，训练覆盖两种生成器的新版本；SD3.5 **不再称为本版本的未见生成器**。选择第二版是为了补上已发现的 SD3.5 能力缺口，不是把同一测试集重新命名为独立测试。

| 联合版划分 | 方法 | 图片数 |
| --- | --- | ---: |
| test | FLUX | 1,670 |
| test | SD35 | 1,513 |
| test | real | 3,065 |
| train | FLUX | 7,905 |
| train | SD35 | 7,311 |
| train | real | 15,798 |
| val | FLUX | 1,625 |
| val | SD35 | 1,514 |
| val | real | 3,120 |

已按 LFW 人物、生成种子分组；FFHQ 只能按图片分组，没有证明所有人物互斥。所有来源统一裁脸、128px / JPEG90；两类同样应用颜色、缩放、模糊、翻转与 JPEG 扰动，但来源捷径仍可能存在。

## 联合模型实际结果

固定 5 轮全网微调，真假、来源/生成器和已知身份分组均衡损失。仅用验证集选择第 5 轮；仅用验证集 balanced accuracy 最优区间中点确定阈值 **0.707737**，未用测试或用户图片调阈值。原始观察点阈值结果也保留在 pre_calibration_metrics.json。

| 已见生成器测试 | 图片数（含真实） | AUC | balanced accuracy | 生成召回率 |
| --- | ---: | ---: | ---: | ---: |
| FLUX 对真实 | 4,735 | 1.0000 | 99.95% | 100.00% |
| SD35 对真实 | 4,578 | 1.0000 | 99.92% | 99.93% |

两行包含同一批真实测试照片，不能相加当成独立真实样本。整体测试 6,248 张，AUC 1.0000，balanced accuracy 99.94%。额外 JPEG60 重压缩后 balanced accuracy 99.92%。

额外对未参与生成训练的 FF++ 真实测试视频各取一张人脸帧，共 140 张，误报率 0.00%；这是仅真实类的跨来源压力检查，不是完整跨库 AUC。评估不包含前述未检出脸的 862 张图片。

训练与初次正式评估 13.16 分钟，PyTorch 峰值 allocated 显存 0.74 GiB；校准和核验时间另计。非常高的库内成绩可能仍来自来源差异，不能推断对任意 AI 图片有效。

## 你提供的 GPT 图片

用户提供的 GPT 肖像图片（个人路径和原图未公开）。

本版本结果：**疑似 AI 整脸生成**。生成倾向分 **99.8/100**，阈值 0.707737。分数不是概率。

图片从未进入训练、验证、模型轮次或阈值选择，也没有文件名/标签判断规则。第一版已经观察过这个个例，所以这里属于重复回归检查，**不是新的盲测，也不能代表所有 GPT 图片**。GPT 的具体生成模型版本由你口述来源，未独立归因。

## 使用与代码

关闭旧窗口，重新双击 run_demo.cmd。选择照片 → 开始检测 → 看「疑似 AI 整脸生成」或「未检出明显生成特征」。默认视频走原换脸/篡改模型。要检查照片换脸，在顶部切换「检测项目」。

复现见 GENERATION_README.md。源码重点：generation_common.py、prepare_generation.py、train_generation.py、calibrate_generation.py、face_analysis.py、verify_generation.py、gui.py。当前模型为 artifacts/generation/joint/model.pt；READY.json 表示该版本经过核验并被默认启用。

仍需补充更多生成器（尤其多个 GPT 版本）、更多真实来源、跨来源身份/近重复审计以及真正独立的外部测试。当前完成了能运行、能分任务、保留失败实验与真实成绩的实验版功能，尚未证明通用可靠性。

数据来源：[LFW / scikit-learn](https://scikit-learn.org/stable/datasets/real_world.html#the-labeled-faces-in-the-wild-face-recognition-dataset)、[FFHQ 官方说明](https://github.com/NVlabs/ffhq-dataset)、[FFHQ 镜像](https://huggingface.co/datasets/bitmind/ffhq-256)、[FairFaceGen](https://zenodo.org/records/17062365)。
