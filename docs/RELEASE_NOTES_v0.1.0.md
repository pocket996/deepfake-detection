# FaceGuard v0.1.0 · Windows 免安装 CPU 预览版

## 队友如何使用

1. 下载附件 **FaceGuard-v0.1.0-windows-x64-cpu.zip**，不要下载 Source code。
2. 完整解压，打开 FaceGuard 文件夹，双击 **FaceGuard.exe**。
3. 选择自己的照片或视频，点击蓝色「开始检测」。

不需要 Python、CUDA、独立显卡或训练数据。适用 Windows 10/11 x64 Intel/AMD 电脑；目前没有 macOS、Linux 或 ARM 原生包。首次加载模型需等待，视频处理速度取决于长度和电脑性能。

## 包含功能

- 照片默认检测 AI 整脸生成；视频默认检测换脸/面部篡改。
- 照片要查换脸时，可切换顶部「检测项目」。
- 中文步骤引导、照片/视频首帧预览、真实读取进度、结果解释和报告保存。
- 运行时、两份检测模型与 YuNet 随包提供，资源缺失或损坏会给出说明。
- 文件在本机处理，不上传网络。个人照片、训练数据与数据集示例未包含在发布包中。
- ZIP 校验值见 SHA256SUMS.txt；资源与构建信息见包内 runtime_manifest.json / BUILD_INFO.json。

## 已做检查

新目录解压并从 PATH 移除 Python 后，检查启动、生成照片、真实照片、篡改视频、无脸输入、资源损坏拦截。尚未在每位队友的实际电脑上验收。

## 当前能力边界

这是实验预览版。生成模型主要训练于 FLUX 和 SD3.5，陌生生成工具仍可能判断错误；分数不是真实性概率。视频最多采样16帧，每帧看最大人脸，未提供整段 AI 生成视频检测。代码的 MIT 许可证不替代模型训练素材或第三方组件的适用条款。

详细的大白话解释：[项目不足与改进](https://github.com/pocket996/deepfake-detection/blob/v0.1.0/docs/PROJECT_LIMITATIONS_PLAIN.md)。
