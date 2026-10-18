# 较新的图片数据集候选（核查日期：2026-10-18）

AI-Face 不是唯一选择。新论文年份不能保证覆盖所有最新生成器；应核对实际生成方法，并按未见生成器或独立来源测试。

## 先考虑：FairFaceGen（2025）

Idiap 于 2025-09-05 发布，原用途为合成人脸识别与偏差研究，包含 FLUX.1-dev 和 Stable Diffusion 3.5 的种子人脸及后续身份扩增。原始种子包各含 11,200 个生成身份：flux-seed.zip 约 299.4 MB，sd35-seed.zip 约 432.4 MB。可从官方 Zenodo 分包直接下载，无需全量 202.2 GB 扩增包。

- 官方页面：https://zenodo.org/records/17062365
- FLUX 原始种子包：https://zenodo.org/records/17062365/files/flux-seed.zip?download=1
- SD3.5 原始种子包：https://zenodo.org/records/17062365/files/sd35-seed.zip?download=1
- 说明：https://zenodo.org/records/17062365/files/README.md?download=1

适合补充较新的整脸生成样本，但不是现成的真假检测基准：需搭配可核实的真实人脸照片，匹配裁剪、分辨率和压缩流程。不能只把生成头像与 FF++ 压缩视频帧直接二分后报告泛化。若使用身份扩增包，同一身份与其派生图必须分到同一组。下载后确认许可、包校验值和标签，不用肉眼真实感取代实验。

## 人脸扩散检测：DiFF（ACM MM 2024）

覆盖 13 种方法、四类条件、超过 50 万张面向人脸的扩散生成/篡改图片。2026 年迁移仓库不代表生成方法更新为 2026 年。官方当前合成图片入口为 Google Drive；原始图片与提示词需要申请。仍可能遇到跨境下载瓶颈。

官方：https://github.com/iLearn-Lab/MM24-DiFF

适合补充人脸扩散检测和独立测试，需按整脸生成、换脸、编辑等任务分别记录。

## 通用生成检测：Community Forensics（CVPR 2025）

覆盖 4,803 个生成模型、约 270 万张生成图片，偏通用内容，不仅是人脸。官方提供 Hugging Face 数据。全量约 1.1 TB，Small 约 278 GB，不能按名称理解为适合本机全量下载的小包；当前 D 盘空间不足以容纳整个 Small 集，需选择有限分片/流式样本并另行核对存储和真实数据。

- 项目：https://jespark.net/projects/2024/community_forensics/
- Small：https://huggingface.co/datasets/OwensLab/CommunityForensics-Small

适合将来扩大到通用 AI 图片检测，或筛选含人脸的子集。不能直接替代专门的人脸数据或绕过真实/生成来源差异问题。

## 当前建议

继续用 FF++ 建立换脸基线；整脸生成部分优先核查并获取 FairFaceGen 两个原始种子包，配套独立真实照片，再以 DiFF/其他未见生成器补充泛化测试。不必为了等待 AI-Face 而停下数据处理和加载流程，也不应把增加数据集数量等同于解决真实环境泛化。

## 本机下载与训练安排（2026-10-18）

已启动两个种子包和官方 README 的下载，目录为 `data/raw/fairfacegen`。是否完成以 `outputs/fairfacegen-download-status.json` 的 `state=complete` 为准。下载器核对官方 MD5，并检查 ZIP 全部文件的 CRC，完成前保留 `.part`；未解压、未自动纳入训练。官方页面和 README 未发现明确的数据许可文本，研究使用前仍需确认适用条款。

仅在没有同类下载进程运行时恢复：

```powershell
.\.venv\Scripts\python.exe scripts/download_fairfacegen.py
```

训练安排是先用 FF++ 的官方划分建立换脸/篡改基线，再给 FairFaceGen 配套独立真实照片并建立整脸生成实验。两类实验验证后，可以合并各自的训练清单联合训练，通过采样控制来源、任务和真假比例；验证、测试样本不进入训练。同一身份、源视频及其派生图片保持在同一划分，并检查跨库重复。统一裁脸尺寸和处理流程，避免模型靠图片/视频帧之间的压缩或来源差异判断真假。

评估分别报告换脸/篡改和整脸生成效果，额外保留未见生成方法测试；例如用 FLUX 训练、SD3.5 测试时，SD3.5 不参与训练和调参，真实照片也需独立划分。这是与两种生成器都参与训练不同的实验配置。当前尚未实现训练模型、联合采样或 FairFaceGen 配套真实照片清单。
