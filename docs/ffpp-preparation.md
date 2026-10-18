# FF++ 数据整理与统一加载

## 已完成的首批验证

2026-10-18 已对全部 5,000 个视频建立官方划分清单：train 3,600、val 700、test 700，相关视频按 500 组原始编号对隔离。每种方法各 1,000 个视频。清单检查包含文件存在性、下载收据大小和来源编号不跨划分，不代表生物身份去重。

从每个划分确定性选择一组编号对，共 30 个真实/伪造视频，已完成顺序帧遍历、均匀抽帧与裁脸，得到 480 张 256×256 PNG：每个划分 160 张，每种方法 96 张。该小批次用于流程调试，不能作为正式检测效果结论。

首批清单：`data/manifests/ffpp-frames-preview-1-93e2b370610e.csv`。
报告：`outputs/ffpp-preprocess-preview-1-93e2b370610e.json`。

全量处理已启动（每段最多 16 帧、2 个工作线程）。进度见 `outputs/ffpp-preprocess-full-93e2b370610e.json`；日志见 `outputs/ffpp-preprocess-full.log` 和 `outputs/ffpp-preprocess-full-errors.log`。全量完成前不要把首批结果当成已处理全部数据。

## 运行方式

始终在 `D:\deepfake-detection` 的 PowerShell 使用项目虚拟环境。

1. 重建视频清单：

```powershell
.\.venv\Scripts\python.exe scripts/prepare_ffpp_manifest.py
```

2. 重现首批处理：

```powershell
.\.venv\Scripts\python.exe scripts/prepare_ffpp_faces.py --pairs-per-split 1 --frames 16 --size 256 --workers 2
```

3. 验证首批真实/伪造图片均能组成模型输入批次：

```powershell
.\.venv\Scripts\python.exe scripts/check_data_loader.py data/manifests/ffpp-frames-preview-1-93e2b370610e.csv
```

4. 全量处理或中断后的恢复命令（仅在没有同类处理进程运行时执行）：

```powershell
.\.venv\Scripts\python.exe scripts/prepare_ffpp_faces.py --all --frames 16 --size 256 --workers 2
```

处理成功的组有缓存，恢复时跳过；改变抽帧数、裁剪尺寸、边界比例或检测权重会产生独立配置目录，不覆盖原始视频。新配置的清单名会包含新的配置编号。磁盘剩余不足 10 GiB 时停止新的目标组处理并记录错误。

完成后，全量清单为 `data/manifests/ffpp-frames-full-93e2b370610e.csv`。运行同一个 check_data_loader.py 验证它。若进度报告 state=incomplete，先检查 errors，不能把部分输出当成全量成功。

## 裁脸规则与已知限制

- 真实与各类伪造视频均使用相同 YuNet 参数：检测前将长边限制到 640，置信度阈值 0.8，选择最大人脸，扩大到 1.3 倍方框并限制在画面内，输出 256×256 PNG。
- 原始与伪造视频长度可能不同，不假定逐帧对齐，不把原始视频某一帧的框直接套到伪造视频。每个视频独立均匀抽样、检测。
- 没有人脸的采样帧记录在报告中并跳过，不以整张画面冒充人脸样本；多脸视频中最大人脸规则仍可能选错，需抽样查看。
- 顺序遍历所有帧，解码取出的采样帧，并核对帧遍历数与元数据。未转换保存所有原始帧，不将这一检查描述为逐像素完整质量审查。
- 图像帧保留 parent_video_id、source_video、frame_index、timestamp、crop_box、face_score 与来源分组。视频指标应先按 parent_video_id 汇总，再计算，不能把相邻帧当独立视频。
- 官方文件名编号字段保留为 target_id/source_id；这里只用它们隔离相关来源，不据此假定所有伪造方法的逐帧映射相同。

## 统一数据加载

`src/faceguard/data.py` 的 FaceImageDataset 接受相同 CSV 格式的图片清单；来自视频的帧也以图片形式加载。默认返回 RGB、float32、[0,1] 范围、CHW 形式的张量以及标签和来源信息。预训练模型需要的归一化由后续训练 transform 明确配置。

```python
from faceguard.data import FaceImageDataset
from torch.utils.data import DataLoader

dataset = FaceImageDataset(
    'data/manifests/ffpp-frames-preview-1-93e2b370610e.csv',
    split='train',
)
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
batch = next(iter(loader))
images, labels = batch['image'], batch['label']
```

在自建脚本中先将项目 `src` 目录加入 Python 模块路径，方式可参考 scripts/check_data_loader.py。Windows 后续使用多进程 DataLoader 时需将入口放在 `if __name__ == '__main__':` 内；当前验证使用 num_workers=0。

新图片集需提供相同字段：sample_id、path、label（真实0/伪造1）、dataset、task、method、compression、target_id、source_id、group_id、split、media_type=image、provenance_status=verified。没有语义的压缩、视频编号字段可留空。保留可审计的来源后才能标为 verified；这一字段不是自动证明许可或标签准确性的工具。

加载器在过滤划分前检查全清单中的重复路径、编号和来源分组泄漏。派生数据应保留相同的全局 group_id；接入 AI-Face 时需要额外识别它与 FF++ 的重叠，不能只换 dataset 名称绕过分组规则。来源不明的第三方数据仍隔离，不自动进入正式加载器。

验证命令：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 人脸检测权重

使用 OpenCV Zoo 官方 YuNet 2023mar：
https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

模型下载到 weights/face_detection_yunet_2023mar.onnx；MIT 许可保存于 weights/YuNet-LICENSE.txt。模型 SHA256 为 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`。

```powershell
Invoke-WebRequest -Uri 'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx' -OutFile 'weights/face_detection_yunet_2023mar.onnx'
Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/LICENSE' -OutFile 'weights/YuNet-LICENSE.txt'
```
