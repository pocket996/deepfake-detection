# 数据准备

当前状态：尚未下载真实研究数据。环境检查中的图片和视频为合成测试文件，不能用于检测效果评估。

## 第一阶段范围

使用 FaceForensics++ 的 original 和 Deepfakes，先选 c23 视频。图片由同一批视频抽帧产生，不另外寻找来源不同的真假图片合集。开发样例先准备约10个真实视频和10个伪造视频，之后再按官方划分扩大。

官方说明：https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md

官方申请入口：https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform

需要项目成员填写身份、用途并接受条款，审批后获得下载脚本。不要将脚本中的私人访问链接或数据上传到公开仓库。暂不下载 raw 或全量图片；官方估计 c23 全部视频约10GB，raw 视频约500GB，图片约2TB，实际以下载清单为准。

## 收到官方脚本之后

把脚本保存为 tools/download-FaceForensics.py。在项目目录的 PowerShell 中执行以下命令。若收到的脚本参数有所变化，以其 --help 为准。

```powershell
.\.venv\Scripts\python.exe .\tools\download-FaceForensics.py --help
.\.venv\Scripts\python.exe .\tools\download-FaceForensics.py .\data\raw\ffpp -d original -c c23 -t videos --num_videos 10
.\.venv\Scripts\python.exe .\tools\download-FaceForensics.py .\data\raw\ffpp -d Deepfakes -c c23 -t videos --num_videos 10
```

两次下载不保证原始/伪造文件逐个对应，需要根据文件编号核对关联。这20个文件只作为开发样例，不能直接作为训练、验证、测试的正式划分。

## 正式实验之前

1. 获取并遵循官方 train/val/test 划分，记录来源与版本。
2. 原始文件保留编号；伪造文件的 target_source 两个编号都记录。
3. 建立样本清单：sample_id,path,label,dataset,method,compression,target_id,source_id,split。
4. 先按源视频关系分组，再抽帧；相关帧、压缩版本、真实及伪造派生视频不能跨划分。
5. 检查所用检测权重的训练数据，避免把见过的数据称为独立测试。
6. 确认数据条款允许的演示方式，不将受限原始视频直接附在公开成果中。

## 目录

- data/raw/ffpp：官方视频，保留下载器目录结构。
- data/manifests：样本与划分清单。
- data/processed：后续抽帧与裁剪，按需生成。
- weights：检测权重。
- outputs：运行报告与实验结果。

目前没有运行人脸裁剪，没有加载伪造检测模型，也没有形成正式评估集。
