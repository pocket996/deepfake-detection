# FaceGuard v0.1.0 模型说明

本发布版包含两个独立模型：

| 分支 | 训练与用途 | 没有证明的能力 |
| --- | --- | --- |
| AI 整脸生成 | LFW、FFHQ-256 镜像真实照片与 FairFaceGen 的 FLUX.1-dev、SD3.5；照片中的最大人脸 | 所有 GPT 版本、所有生成器、普通无脸 AI 图片 |
| 换脸/面部篡改 | FF++ c23 四种已知篡改方法；视频均匀采样最多16帧，每帧最大人脸 | 所有网络视频、短暂局部篡改、整段 AI 生成视频 |

两者都是 ImageNet ResNet18 骨干微调而来。生成分支统一裁脸到128像素并做JPEG90标准化，再送入224像素网络。它没有通过时序模型连续观看视频，也不会对所有人脸逐一追踪。

阈值仅从各自验证集确定。分数是未经概率校准的模型输出，不能按真实性概率解释。没有检测到脸时返回无法判断。

当前生成模型的已见来源测试分数很高，但只反映本次划分：FFHQ 镜像缺少完整身份标签，未完成跨来源人物与近重复审计，来源差异仍可能被模型当成捷径。第一版仅训练 FLUX 时，未见 SD3.5 的召回率只有18.03%；该失败实验保留在源仓库。后来的联合训练将两种生成器都纳入，因此 SD3.5 不再算当前版本的未见生成器。

一个用户提供的 GPT 肖像曾被联合模型识别为生成，分数约99.8/100；它没有用于训练或阈值选择，但也不能替代多版本、成规模的 GPT 测试。个人原图不包含在发布包中。

模型、就绪标记与 YuNet 文件校验值见 runtime_manifest.json。源代码、版本与构建依赖见 BUILD_INFO.json。构建包只包含推理需要的文件，不包含训练数据或个人回归图片。

自有代码按仓库 MIT 许可证提供；第三方程序与 YuNet 的版权和许可证随包保留在 THIRD_PARTY_LICENSES 与 weights 中。模型训练来源及其素材条款不因代码采用 MIT 而被替代；本项目尚未完成训练素材的完整许可与商用适用性核查。

来源：

- FF++：https://github.com/ondyari/FaceForensics
- FairFaceGen：https://zenodo.org/records/17062365
- FFHQ 官方：https://github.com/NVlabs/ffhq-dataset
- 使用的 FFHQ 镜像：https://huggingface.co/datasets/bitmind/ffhq-256
- LFW 下载校验配置：https://github.com/scikit-learn/scikit-learn/blob/1.6.1/sklearn/datasets/_lfw.py
- YuNet：https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- Torchvision：https://github.com/pytorch/vision
