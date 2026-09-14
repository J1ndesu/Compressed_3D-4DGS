# Compressed 3D–4D Gaussian Splatting

基于 [Hybrid 3D–4D Gaussian Splatting](https://github.com/ohsngjun/3D-4DGS) 的动态场景压缩实验仓库，研究混合 3D/4D 高斯表示中的**高斯数量剪枝**与**球谐（SH）系数剪枝**。

本页对应 **my_method** 分支。上游论文和作者信息见文末；本仓库的压缩扩展与上游原论文贡献分别说明。

## 分支导航

| 分支 | 用途 |
| --- | --- |
| [main](https://github.com/J1ndesu/Compressed_3D-4DGS/tree/main) | 基础混合 3D/4DGS 代码 |
| [my_method](https://github.com/J1ndesu/Compressed_3D-4DGS/tree/my_method) | 时间统计加权的高斯剪枝、独立 SH 分带剪枝与评估 |
| [RDOGS-based](https://github.com/J1ndesu/Compressed_3D-4DGS/tree/RDOGS-based) | 压缩对照实现；使用该分支自己的参数与检查点 |
| [page](https://github.com/J1ndesu/Compressed_3D-4DGS/tree/page) | 页面资源 |

## 已实现的功能

- **混合表示**：动态池使用 4D 高斯，静态池使用 3D 高斯；训练时可按时间尺度将动态点转入静态池。
- **高斯剪枝**：分别学习两类高斯的掩码 logits。训练期间通过直通估计器（STE）门控不透明度；验证期间按阈值删除点及对应参数行。
- **动态稀疏正则**：使用累积计数比、条件均值偏移量和时间尺度构造权重，对低于静态转换阈值的动态点施加正则。
- **SH 剪枝**：每个高斯独立学习空间 SH 的 1、2、3 阶掩码，保留 DC 分量与额外通道。高阶保留不要求低阶保留。
- **评估**：比较硬剪枝前后的 PSNR、SSIM、LPIPS、点数与参数载荷估计，导出预测图和参考图。

VQ 相关开关和权重是预留参数，当前没有完整的矢量量化、熵编码或压缩码流导出流程。SH 硬剪枝只将系数置零，不缩小张量。

## 代码结构

| 路径 | 作用 |
| --- | --- |
| [main.py](main.py) | 训练入口、掩码调度、正则项、剪枝前后验证 |
| [arguments/__init__.py](arguments/__init__.py) | 命令行参数与默认值 |
| [configs/](configs/) | N3V、DNeRF 实验配置 |
| [scene/gaussian_model.py](scene/gaussian_model.py) | 高斯参数、掩码、增密、剪枝、检查点和大小统计 |
| [gaussian_renderer/](gaussian_renderer/) | 混合渲染、掩码应用与 CUDA 自动微分包装 |
| [utils/compression_utils.py](utils/compression_utils.py) | STE、SH 掩码和分带应用 |
| [scripts/n3v2blender.py](scripts/n3v2blender.py) | 单个 N3V 场景预处理 |
| [train.sh](train.sh)、[val.sh](val.sh) | 多场景训练与验证脚本 |
| [curve.py](curve.py) | 手工实验数据的率失真曲线 |
| [diff-gaussian-rasterization/](diff-gaussian-rasterization/)、[simple-knn/](simple-knn/)、[pointops2/](pointops2/) | CUDA 渲染与点云算子 |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | 注释约定与文档维护方法 |

## 环境安装

代码需要 NVIDIA GPU、兼容的驱动、CUDA 编译工具链（含 nvcc）及 C++ 编译器。[environment.yml](environment.yml) 固定了 Python 3.7.13、PyTorch 1.12.1 和 CUDA runtime 11.6；这些是仓库记录的依赖版本，不代表已验证所有系统组合。

~~~bash
git clone --branch my_method https://github.com/J1ndesu/Compressed_3D-4DGS.git
cd Compressed_3D-4DGS
conda env create --file environment.yml
conda activate 3d4dgs
~~~

环境创建期间会编译本地 simple-knn 和 pointops2。渲染器在首次导入时通过 PyTorch JIT 编译 diff-gaussian-rasterization；CUDA runtime 不能替代 nvcc。

预处理还需要 FFmpeg、COLMAP 和 OpenCV；JIT 编译需要 Ninja。Matplotlib 用于渲染辅助工具和曲线，TensorBoard 为可选日志依赖。请在已激活环境中补齐兼容版本，并检查：

~~~bash
nvcc --version
colmap -h
ffmpeg -version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
~~~

LPIPS 首次运行可能下载预训练权重。本次文档整理未执行 CUDA 编译、GPU 训练或渲染。

## 数据准备

### Neural 3D Video（N3V）

从 [Neural 3D Video 项目](https://github.com/facebookresearch/Neural_3D_Video) 获取数据，对每个场景分别处理：

~~~bash
python scripts/n3v2blender.py /path/to/N3V/coffee_martini
~~~

输入目录需包含各相机的 MP4 视频和 poses_bounds.npy。脚本提取前 10 秒视频，将 PNG 宽高各缩小一半，生成 images/、transforms_train.json、transforms_test.json，并调用 COLMAP 生成 points3d.ply。脚本会写入输入目录并覆盖处理后的图像，已有半分辨率图像无需再次手动缩小。

### DNeRF / Blender 格式

source_path 指向含相机 JSON 与图像的单个场景目录。读取器需要 transforms_train.json 和 transforms_test.json；路径以 lego 结尾时，测试文件改用 transforms_val.json。JSON 中的 file_path、相机变换和可选 time 字段必须与图像对应。

缺少 points3d.ply 时会生成随机点云。两个默认配置的 num_pts 均为 300_0000，即 **3,000,000**，请结合显存调整。DNeRF 配置仍使用 [0.0, 10.0] 时间区间与黑背景，需核对实际数据的时间范围和背景。

## 训练

从仓库根目录运行，替换示例路径：

~~~bash
# N3V
python main.py \
  --config configs/n3v/default.yaml \
  --source_path /path/to/N3V/coffee_martini \
  --model_path output/N3V/coffee_martini

# DNeRF
python main.py \
  --config configs/dnerf/default.yaml \
  --source_path /path/to/dnerf/standup \
  --model_path output/DNeRF/standup
~~~

**配置优先级：YAML > 命令行 > 参数类默认值。** main.py 先解析命令行，再将 YAML 分组中的字段覆盖到参数对象。修改总迭代数、损失权重或开关时应编辑所用 YAML，同名 CLI 选项不能覆盖 YAML。布尔 CLI 选项使用 store_true，不支持通过 “--use_pruning False” 关闭开关。

批量训练前，编辑 train.sh 的 GPU、数据根目录、场景数组、数据集开关及底部调用，再执行：

~~~bash
bash train.sh
~~~

当前 N3V 场景数组为空且调用被注释；DNeRF 仅选择 standup。仅将 TRAIN_N3V 改为 true 不会启用被注释的调用。

## 压缩参数

以下值来自默认 YAML，参数类回退值可能不同。

| 参数 | N3V | DNeRF | 含义 |
| --- | --- | --- | --- |
| use_pruning / use_sh_adaptive | True / True | True / True | 启用高斯 / SH 掩码训练与对应硬剪枝步骤 |
| lambda_static_mask | 0.0002 | 0.0001 | 静态高斯稀疏正则权重 |
| lambda_dynamic_mask | 0.0005 | 0.0002 | 动态高斯加权稀疏正则权重 |
| lambda_sh | 0.0005 | 0.0002 | SH 分带稀疏正则权重 |
| phi_threshold | 0.1 | 0.05 | 训练渲染时的高斯门控阈值 |
| phi_prune_dynamic | 0.25 | 0.05 | 验证时动态高斯硬剪枝阈值 |
| phi_prune_static | 0.15 | 0.05 | 验证时静态高斯硬剪枝阈值 |
| phi_prune_sh | 0.1 | 0.05 | SH 渲染门控与硬剪枝阈值 |
| gs_mask_start_iter | 4000 | 4000 | 高斯门控起始迭代 |
| gs_mask_warmup_iters_static / gs_mask_warmup_iters_dynamic | 1000 / 1000 | 1000 / 1000 | 两类高斯正则权重的线性预热长度 |
| sh_mask_start_iter / sh_mask_warmup_iters | 4500 / 1000 | 4500 / 1000 | SH 门控起始迭代 / 正则预热长度 |
| static_mask_lr / dynamic_mask_lr / sh_mask_lr | 均 0.005 | 均 0.005 | 掩码 logits 学习率 |

阈值作用于 sigmoid(logit)，仅当其**严格大于**阈值时保留。门控在起始迭代立即启用，正则权重从零逐步增加，预热长度应为正数。SH 的三个正则权重比例为 3/15、5/15、7/15，对应各阶系数数量。默认使用 sh_degree: 3 和 force_sh_3d: True；apply_sh_masks 要求至少 16 个 SH 通道。

进行消融实验时复制 YAML，按下表配置，并可将未启用功能的对应正则权重设为零，使日志与实验配置更清晰：

| 实验 | use_pruning | use_sh_adaptive |
| --- | --- | --- |
| 高斯剪枝 | True | False |
| SH 剪枝 | False | True |
| 联合剪枝 | True | True |
| 不训练掩码 | False | False |

use_vq 保持 False；lambda_vqr、lambda_vqs、lambda_vqc 和 codebook_lr 保持零值。通用日志路径尚无对应的 VQ 损失计算。

## 验证与输出

默认训练、保存与测试配置包含 6000 次迭代。使用实际生成的检查点：

~~~bash
python main.py \
  --config configs/n3v/default.yaml \
  --source_path /path/to/N3V/coffee_martini \
  --model_path output/N3V/coffee_martini \
  --start_checkpoint output/N3V/coffee_martini/chkpnt6000.pth \
  --val
~~~

也可编辑 val.sh 后执行 bash val.sh。当前批量验证仅调用 DNeRF 的 mutant 和 standup，默认检查点名为 chkpnt6000.pth；缺少数据或检查点的场景会跳过。

| 输出 | 内容 |
| --- | --- |
| cfg_args | 模型/数据参数，不是完整的最终 YAML 配置快照 |
| chkpnt&lt;iteration&gt;.pth | 模型参数、掩码及优化器状态 |
| chkpnt_best.pth | 定期测试中按 PSNR 保存的最佳检查点 |
| point_cloud/iteration_&lt;iteration&gt;/point_cloud.ply | 动态池的空间属性，不含静态池、时间参数、掩码或优化器 |
| test_images/iter_6000/ | 验证导出的预测图与可用参考图；目录编号当前硬编码为 6000 |
| test/ours_&lt;checkpoint_iteration&gt;/stats/validation.json | 渲染提取器生成的汇总指标与两类点数 |
| figures/ | 手动运行 python curve.py 后生成的曲线 |

验证先评估载入的模型，再在内存中删除高斯并将 SH 分带置零，最后重新评估。它**不保存剪枝后的检查点**。测试集必须非空，保留 ModelParams.eval: True；否则 Blender 读取器会将测试视图并入训练集。

curve.py 使用手工录入的数据，不会自动读取训练日志。论文或报告中的曲线需另外记录场景、配置、检查点与统计口径。

### 大小统计口径

- **dense** 按选中参数张量的形状和元素字节数统计载荷。SH 置零不改变张量形状，不会单独减少这一数值。
- **effective** 按高斯与 SH 掩码估计保留参数的载荷，包含未剪枝的额外 SH 通道；它不是实际编码文件大小。
- 日志与返回值沿用 MB / *_mb 命名，但除数为 1024²，实际单位为 **MiB**。
- 默认不计掩码、训练统计量、环境图和优化器状态；序列化、索引等开销也未计入。检查点文件大小和显存占用应单独测量。

### 当前实现限制

1. 检查点使用位置元组格式，分支间字段不同，不应假定可以互换。训练恢复会调用 training_setup 重建掩码和统计量，不能视作严格等价的无缝续训。
2. 验证时 restore(..., None) 从检查点推断 SH 开关并启用门控，模型对象的 phi_prune_sh 保留构造默认值 0.1；后续硬剪枝使用 YAML 阈值。因此配置值不为 0.1 时，两阶段可能使用不同阈值。关闭验证选项也不等于关闭已载入的 SH 门控。
3. vis 使用两个累积计数器之比，二者通常对相同可见行递增，不能直接视作整段视频的可见概率。
4. 修改总迭代数时同步核对 test_iterations、save_iterations、门控起点和预热长度。保存列表在 YAML 合并前追加 CLI 的迭代值，不会自动同步 YAML 中新的结束迭代。
5. YAML 的 densify_grad_t_threshold: 0.0002 / 40 是表达式文本；当前增密选择未使用该参数。启用新的时间梯度逻辑前需核对类型与数值。
6. 控制台剪枝前后的 LPIPS 使用 AlexNet；最终 JSON 来自另一套渲染提取器评估路径，不应混作同一口径的前后对比。

## 上游项目与引用

本仓库建立在 Seungjun Oh、Younggeun Lee、Hyejin Jeon、Eunbyung Park 的工作之上：
[论文](https://arxiv.org/abs/2505.13215) · [项目主页](https://ohsngjun.github.io/3D-4DGS/) · [上游代码](https://github.com/ohsngjun/3D-4DGS)。

下图来自上游混合表示工作，不是本分支的压缩结果。

![上游混合 3D/4D Gaussian Splatting 方法概览](assets/main.jpg)

~~~bibtex
@article{oh2025hybrid,
  title={Hybrid 3D-4D Gaussian Splatting for Fast Dynamic Scene Representation},
  author={Oh, Seungjun and Lee, Younggeun and Jeon, Hyejin and Park, Eunbyung},
  journal={arXiv preprint arXiv:2505.13215},
  year={2025}
}
~~~

感谢 [Real-time 4D Gaussian Splatting](https://github.com/fudan-zvg/4d-gaussian-splatting)、[Ex4DGS](https://github.com/juno181/Ex4DGS)、[4D-Rotor Gaussians](https://github.com/weify627/4D-Rotor-Gaussians) 及 [@sorceressyidi](https://github.com/sorceressyidi) 的相关代码与工作。

## 许可证

根目录包含 [MIT LICENSE](LICENSE)。部分继承源码头部另有上游使用声明，第三方目录也保留自己的许可文件；原始版权与许可声明均予以保留。
