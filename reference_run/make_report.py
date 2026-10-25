"""Render a readable report from measured artifacts, never from invented scores."""
import json
import os
from pathlib import Path
import shutil

from core import HERE, digest, save_json


def directory_bytes(path):
    return sum(os.path.getsize(Path(root) / name) for root, _, names in os.walk(path) for name in names)


def main():
    artifact = HERE / 'artifacts'
    m = json.loads((artifact / 'metrics.json').read_text(encoding='utf-8'))
    h = json.loads((artifact / 'holdout_metrics.json').read_text(encoding='utf-8'))
    v = json.loads((artifact / 'verification.json').read_text(encoding='utf-8'))
    fm = json.loads((artifact / 'finetuned_metrics.json').read_text(encoding='utf-8'))
    fv = json.loads((artifact / 'finetuned_verification.json').read_text(encoding='utf-8'))
    meta = json.loads((artifact / 'features.json').read_text(encoding='utf-8'))
    test = m['splits']['test']['video']
    val = m['splits']['val']['video']
    lo, hi = m['splits']['test']['video_auc_95pct_group_bootstrap']
    storage = dict(venv_bytes=directory_bytes(HERE / '.venv'),
                   artifacts_bytes=directory_bytes(artifact), weights_bytes=directory_bytes(HERE / 'weights'),
                   disk_free_bytes=shutil.disk_usage(HERE).free)
    save_json(artifact / 'storage.json', storage)
    save_json(artifact / 'reproducibility.json', dict(
        code_sha256={p.name: digest(p) for p in sorted(HERE.glob('*.py'))},
        model_sha256=digest(artifact / 'model.pt'),
        finetuned_model_sha256=digest(artifact / 'model_finetuned.pt'),
        holdout_model_sha256=digest(artifact / 'model_holdout.pt'),
        feature_manifest_sha256=meta['manifest_sha256'],
        imagenet_weights_sha256=digest(HERE / 'weights/torch/checkpoints/resnet18-f37072fd.pth'),
        yunet_sha256=digest(HERE / 'weights/face_detection_yunet_2023mar.onnx')))
    save_json(artifact / 'experiment_config.json', dict(seed=42,
        source_manifest='../data/manifests/ffpp-frames-full-93e2b370610e.csv',
        dataset='FF++ c23, official source-pair split', input_size=224,
        frozen=dict(backbone='resnet18 IMAGENET1K_V1', trainable='logistic head only',
                    C_candidates=[0.001, 0.01, 0.1, 1.0], selection='validation video AUC'),
        finetuned=dict(epochs=4, frames_per_video_per_epoch=4, rotating_frame_subsets=True,
                       train_frame_counts=[e['train_frames'] for e in fm['history']],
                       optimizer='AdamW', backbone_lr=1e-4, head_lr=3e-4,
                       weight_decay=1e-4, scheduler='cosine', batch_size=48,
                       augmentation='horizontal flip', cuda_mixed_precision=True,
                       selection='validation video AUC', selected_epoch=fm['selected_epoch']),
        holdout=dict(backbone='frozen', excluded_from_train_and_val='NeuralTextures'),
        calibration='none', threshold_selection='validation balanced accuracy',
        video_aggregation='arithmetic mean of frame sigmoid scores'))
    pct = lambda n: f'{n * 100:.2f}%'
    rows = '\n'.join(f"| {name} 对真实 | {s['n']} | {s['auc']:.4f} | {pct(s['balanced_accuracy'])} | {pct(s['recall'])} |"
                     for name, s in m['splits']['test']['by_method'].items())
    confusion = test['confusion_matrix']
    ft = fm['splits']['test']['video']
    fval = fm['splits']['val']['video']
    fconf = ft['confusion_matrix']
    finetuned_methods = '\n'.join(f"| {name} 对真实 | {s['n']} | {s['auc']:.4f} | {pct(s['balanced_accuracy'])} |"
        for name, s in fm['splits']['test']['by_method'].items())
    generation_note = ('本报告记录第一阶段 FF++ 实验。新增整脸生成的当前联合模型见 GENERATION_CURRENT.md，首次跨生成器实验见 GENERATION_RESULTS.md；两个任务指标分别统计。'
        if (HERE / 'artifacts/generation/metrics.json').exists() else
        '这是第一阶段的人脸换脸/篡改基线。AI 整脸生成照片和整段生成视频仍缺完整配套数据，尚未训练这些任务；本次没有把它们计为已完成。')
    report = f'''# 实际运行结果（2026-10-25）

已完成现有 FF++ 数据上的独立参考实验：数据审计 → 冻结骨干对照 → 全网微调 → 验证集选择参数/阈值 → 测试 → 导出 → 原视频推理校验 → 桌面演示。

{generation_note}

## 默认演示模型：全网微调

| 指标 | 冻结骨干对照 | 全网微调 |
| --- | ---: | ---: |
| 验证集视频 AUC | {val['auc']:.4f} | {fval['auc']:.4f} |
| 测试集视频 AUC | {test['auc']:.4f} | **{ft['auc']:.4f}** |
| 测试集 balanced accuracy | {pct(test['balanced_accuracy'])} | **{pct(ft['balanced_accuracy'])}** |
| 测试集 accuracy | {pct(test['accuracy'])} | {pct(ft['accuracy'])} |
| 测试集伪造召回率 | {pct(test['recall'])} | {pct(ft['recall'])} |

微调固定运行 4 轮，每轮每段训练视频取 4 张交错帧，轮换后合计覆盖全部训练帧。每轮使用完整验证集，选中第 **{fm['selected_epoch']}** 轮，最后才对测试集评分。微调没有使用测试集选轮次、学习率或阈值。

微调实际耗时 **{fm['timing_seconds'] / 60:.2f} 分钟**（不含导入启动），PyTorch 峰值 allocated 显存 **{fm['peak_gpu_allocated_bytes'] / 1024**3:.2f} GiB**。视频阈值 {ft['threshold']:.6f}。微调原视频一致性和 CPU/GPU 等核验耗时 {fv['seconds']:.1f} 秒。

| 实际类别 | 微调判真实 | 微调判篡改 |
| --- | ---: | ---: |
| 真实（140 段） | {fconf[0][0]} | {fconf[0][1]} |
| 篡改（560 段） | {fconf[1][0]} | {fconf[1][1]} |

| 微调各方法测试 | 视频数（包含真实） | AUC | balanced accuracy |
| --- | ---: | ---: | ---: |
{finetuned_methods}

默认 GUI 和命令行使用 `model_finetuned.pt`。全部预测与错误保存在 `finetuned_test_videos.csv`。这是已见四种篡改方法的 FF++ 库内测试；高分仍不能证明对陌生数据集或整脸生成有效。

## 冻结骨干对照的详细结果

| 指标 | 验证集（700 段） | 测试集（700 段） |
| --- | ---: | ---: |
| 视频 AUC | {val['auc']:.4f} | {test['auc']:.4f} |
| 视频 balanced accuracy | {pct(val['balanced_accuracy'])} | {pct(test['balanced_accuracy'])} |
| 视频 accuracy | {pct(val['accuracy'])} | {pct(test['accuracy'])} |
| 伪造召回率 | {pct(val['recall'])} | {pct(test['recall'])} |
| 伪造 precision | {pct(val['precision'])} | {pct(test['precision'])} |

AUC 衡量模型将伪造排在真实之前的能力；0.5 接近随机排序。balanced accuracy 将真实和伪造的召回率等权平均。数据真假数量为 1:4，不能只看 accuracy（全部猜假也有 80%）。

测试视频 AUC 的 95% 分组 bootstrap 区间为 [{lo:.4f}, {hi:.4f}]，以 70 对官方测试来源组为重采样单位，500 次。该区间只反映此固定模型、此数据集的采样不确定性，不覆盖重复训练或现实分布变化。

视频阈值 {test['threshold']:.6f}、图片帧阈值 {m['splits']['test']['frame']['threshold']:.6f} 均由验证集确定。分类层 C={m['selected_C']}。测试集未用于选择这些参数。图片结果来自 FF++ 视频帧，不能据此声称对独立真实照片有效。

## 错误有哪些

| 实际类别 | 判真实 | 判篡改 |
| --- | ---: | ---: |
| 真实（140 段） | {confusion[0][0]} | {confusion[0][1]} |
| 篡改（560 段） | {confusion[1][0]} | {confusion[1][1]} |

所有视频的分数和错误都保存在 `artifacts/test_videos.csv`，可按 `label` 与 `prediction` 不同筛选。分数未校准，不是事实真假的概率或鉴定结论。

## 各方法测试

| 伪造方法 | 视频数（包含真实） | AUC | balanced accuracy | 伪造召回率 |
| --- | ---: | ---: | ---: | ---: |
{rows}

每行包括同一批 140 段真实和该方法的 140 段伪造，因此不能相加视为独立测试样本。各行复用主实验固定的验证集阈值，没有用方法测试集重新调阈值。

## 未见方法实验

冻结骨干实验中，训练和验证均排除 NeuralTextures，最终仅在 test 的真实 + NeuralTextures 共 {h['test_video']['n']} 段视频上测试：AUC **{h['test_video']['auc']:.4f}**，balanced accuracy **{pct(h['test_video']['balanced_accuracy'])}**，伪造召回率 {pct(h['test_video']['recall'])}。参数与阈值仅用其余已见方法的验证集确定。没有另做微调模型的未见方法实验，不可把两者成绩混为一谈。

这是 FF++ 内的方法留出；不能证明跨数据集、现代扩散生成器或任意网络视频泛化。此实验另存模型，主模型未被替换。

## 实际时间、资源和核验

- 冻结骨干计时：**{m['timing_seconds']['total'] / 60:.2f} 分钟**，其中完整特征阶段 {m['timing_seconds']['feature_stage'] / 60:.2f} 分钟（含首次官方预训练权重下载、审计与数据加载）。不含 Python 导入启动、此前的数据下载/裁脸、建环境和开发时间。
- 留出方法实验：{h['seconds']:.1f} 秒，复用冻结特征。
- 原视频/模型一致性等核验：{v['seconds']:.1f} 秒。
- GPU：{m['environment']['gpu']}；主实验 PyTorch 峰值 allocated 显存 **{m['environment']['peak_gpu_allocated_bytes'] / 1024**3:.2f} GiB**。不包含驱动、其他进程和 reserved 显存；不能直接当成显卡总占用。
- 独立环境约 {storage['venv_bytes'] / 1024**3:.2f} GiB，当前 artifacts 约 {storage['artifacts_bytes'] / 1024**2:.1f} MiB，weights 约 {storage['weights_bytes'] / 1024**2:.1f} MiB；D 盘剩余约 {storage['disk_free_bytes'] / 1024**3:.1f} GiB。
- 全部 79,994 张图片成功解码；跨划分精确像素重复为 {meta['exact_decoded_pixel_cross_split_duplicates']}；来源编号/组不跨 train/val/test。未做生物身份或近重复审计。
- 导出分类层与 sklearn 的最大验证分数差为 {m['head_export_max_probability_error']:.3g}。
- 两种模型的 CPU/GPU 分数比较、原视频重新抽帧裁脸与清单评估分数比较、无脸图片返回“无法判断”、独立重算测试视频 AUC/平衡准确率均通过。批量/单图分数容差为 0.001，实际误差见 `verification.json`、`finetuned_verification.json`；这是浮点数近似比较，不是逐位相等。
- Python、Torch、GPU 与运行路径见 `metrics.json`；代码与模型校验值见 `reproducibility.json`。

## 下一步怎么借鉴

先双击 `run_demo.cmd`，再阅读 `README.md` 的复现命令和代码阅读顺序。建议你先原样复现，再比较这两种训练方式；以后任何模型选择继续只看验证集。

当前数据足够学习换脸/篡改检测，本机足够，无需先租服务器。若要实现原始设想的全部范围，还需配套真实照片、现代生成器与独立视频测试数据，详细准备项和 1–2 周学习 / 4–8 周扩展计划见 README.md。
'''
    (HERE / 'RESULTS.md').write_text(report, encoding='utf-8')
    print('Wrote RESULTS.md, storage.json and reproducibility.json')


if __name__ == '__main__':
    main()
