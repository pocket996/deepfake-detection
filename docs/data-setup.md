# 数据准备：AI 人脸生成与篡改检测系统

## 项目范围

检测包含人脸的照片与视频，覆盖整脸生成、换脸及面部篡改。模型尚未实现，当前准备数据。AI-Face 图片与 FF++ 视频分别保存，不把全部内容混成一个真假目录。

## 当前状态（2026-10-17）

- AI-Face：用户已能在浏览器看到 Real、DMs、GANs、deepfakes 四个文件夹；本地直接请求官方 SharePoint 链接返回 403，尚未取得图片与标注。需从可访问的浏览器启动下载，然后核对包大小、完整性与文件数量。
- FF++：重新获取已获批的官方 v4 脚本，EU2 试下载五类各一个视频成功。标准 c23 全量下载已完成，共 5000 个视频、9,041,543,739 字节，零失败；状态见 `outputs/ffpp-download-status.json`。已做传输长度、MP4 文件头和 SHA256 收据检查，尚未对全部视频完整解码。
- FF++ 官方划分已保存：`data/manifests/ffpp-splits/train.json`、`val.json`、`test.json`，分别为 360、70、70 对原始视频编号（对应 720、140、140 个原始序列）。
- 整段 AI 生成的视频尚未选定独立数据集。AI-Face 的生成图片和 FF++ 的换脸视频不能替代这类测试数据。

## FF++ 下载范围与恢复

范围为 original、Deepfakes、Face2Face、FaceSwap、NeuralTextures 的 c23 视频各 1000 个；不包含 raw/c40、生成器权重、掩码或独立的 Google DeepFakeDetection 数据集。

```powershell
.\.venv\Scripts\python.exe scripts/download_ffpp.py --accept-terms --workers 4
```

不要与已运行的下载进程同时执行上述命令。后台 PID 记录于 `outputs/ffpp-download.pid`，日志位于 `outputs/ffpp-download.log` 和 `outputs/ffpp-download-errors.log`。

脚本读取获批的 `tools/download-FaceForensics.py` 中的官方目录与 EU2 地址；按官方清单下载，校验响应长度及 MP4 文件头、记录 SHA256 后原子写入正式文件。恢复时校验已有下载收据，未完成的 `.part` 文件会重新下载。网络请求超时后最多重试三次。磁盘剩余不足 20 GiB 时停止写入并记录失败。传输检查不替代视频完整解码检查。

官方说明：https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md

## AI-Face 下载入口

官方仓库：https://github.com/Purdue-M2/AI-Face-FairnessBench

- 图片：https://purdue0-my.sharepoint.com/:f:/g/personal/lin1785_purdue_edu/EoFNIgrp3A5JiIbP5fv9BqABUnbp_BnHKbvpR1xGUTaM7g?e=u5cmIy
- 标注：https://purdue0-my.sharepoint.com/:f:/g/personal/lin1785_purdue_edu/IgA_6O457sRAR5tmhBnIRTwRARCrVHJ7TC4paxwrBszapfU?e=ZJ82Gx

当前计划下载四类图片及官方标注，在同一数据集下保留 Real、DMs、GANs、deepfakes 的原始目录和生成方法。下载前确认总大小与解压空间，本机剩余空间有限，不能默认全量归档和解压结果均可容纳。若浏览器无法下载，官方联系邮箱为 lin1785@purdue.edu、hu968@purdue.edu。

AI-Face 的 GAN/扩散模型子集也可能包含编辑任务，需按方法记录任务类别，不能一概标成整脸生成。官方仓库声明数据许可为 CC BY-NC-ND 4.0；遵循数据及来源条款，并在研究报告引用对应论文。不要把数据、访问凭据或 FF++ 获批脚本提交到公开仓库。

## 数据整理与实验

已实现官方划分清单生成、均匀抽帧、YuNet 人脸裁剪和统一图片加载器。首批 30 个视频、480 张图的流程验证已通过，全量预处理已启动；实际状态以 `outputs/ffpp-preprocess-full-93e2b370610e.json` 为准，运行方法见 [ffpp-preparation.md](ffpp-preparation.md)。不能在全量任务完成前将首批验证称为全数据处理完成。较新图片数据候选与下载入口见 [image-dataset-options.md](image-dataset-options.md)。

保留官方划分，记录数据集、任务类别、生成方法、真实/伪造标签、来源关系和原路径。检查跨数据集重复与身份/源视频重叠，尤其避免 AI-Face 中来自 FF++ 的帧与 FF++ 测试视频相互泄漏。相关视频、抽帧及派生版本须位于同一划分。额外留出未参与训练的生成方法，单独报告图片生成、图片篡改、视频篡改等任务的效果。

## 目录

- data/raw/ai-face：AI-Face 官方归档与图片，目前尚未下载。
- data/raw/ffpp：FF++ c23 原始与伪造视频，5000 个均已下载。
- data/manifests：官方划分、下载收据及后续样本清单。
- data/processed/ai-face、data/processed/ffpp：按需生成处理结果。
- outputs：下载状态、日志、检查报告和实验结果。

环境检查产生的合成文件仅用于验证程序，不能作为检测效果评估数据。
