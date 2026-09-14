# Compressed 3D–4D Gaussian Splatting

基于 [Hybrid 3D–4D Gaussian Splatting](https://github.com/ohsngjun/3D-4DGS) 的动态场景表示与压缩实验仓库。

**当前 `main` 分支提供基础混合 3D/4DGS 实现。高斯数量剪枝和 SH 系数剪枝位于 `my_method` 分支。** 请先选择对应版本，再使用该分支的配置、文档与检查点。

## 分支导航

| 分支 | 内容 |
| --- | --- |
| [`main`](https://github.com/J1ndesu/Compressed_3D-4DGS/tree/main) | 基础混合表示、训练与验证 |
| [`my_method`](https://github.com/J1ndesu/Compressed_3D-4DGS/tree/my_method) | 时间统计加权的高斯剪枝、独立 SH 分带剪枝与大小评估 |
| [`RDOGS-based`](https://github.com/J1ndesu/Compressed_3D-4DGS/tree/RDOGS-based) | 压缩对照方案 |
| [`page`](https://github.com/J1ndesu/Compressed_3D-4DGS/tree/page) | 页面资源 |

要运行本仓库的压缩方法：

```bash
git clone --branch my_method https://github.com/J1ndesu/Compressed_3D-4DGS.git
cd Compressed_3D-4DGS
```

随后阅读 [my_method 使用说明](https://github.com/J1ndesu/Compressed_3D-4DGS/blob/my_method/README.md)。分支间检查点采用不同字段布局，不应直接互换。

## 基础版安装

以下命令针对 `main`。需要 NVIDIA GPU、兼容的 CUDA 编译工具链（含 `nvcc`）、C++ 编译器和 Conda。

```bash
git clone --branch main https://github.com/J1ndesu/Compressed_3D-4DGS.git
cd Compressed_3D-4DGS
conda env create --file environment.yml
conda activate 3d4dgs
```

[`environment.yml`](environment.yml) 记录 Python 3.7.13、PyTorch 1.12.1 和 CUDA runtime 11.6。本地 `simple-knn`、`pointops2` 在安装时编译；渲染器在首次导入时通过 PyTorch JIT 编译。CUDA runtime 不能替代 `nvcc`，JIT 构建还需要 Ninja。N3V 预处理需要 FFmpeg、COLMAP 和 OpenCV；渲染辅助工具使用 Matplotlib，TensorBoard 日志为可选功能。请核对相应依赖是否已安装。

## 数据与运行

从 [Neural 3D Video 项目](https://github.com/facebookresearch/Neural_3D_Video) 获取数据，输入单个含 `.mp4` 和 `poses_bounds.npy` 的场景目录：

```bash
python scripts/n3v2blender.py /path/to/N3V/coffee_martini
```

脚本在场景目录生成图像、Blender 格式的训练/测试相机 JSON 和点云。下列训练与验证命令均从仓库根目录运行：

```bash
python main.py \
  --config configs/n3v/default.yaml \
  --source_path /path/to/N3V/coffee_martini \
  --model_path output/coffee_martini

python main.py \
  --config configs/n3v/default.yaml \
  --source_path /path/to/N3V/coffee_martini \
  --model_path output/coffee_martini \
  --start_checkpoint output/coffee_martini/chkpnt6000.pth \
  --val
```

配置由 [`arguments/__init__.py`](arguments/__init__.py) 和 [`configs/n3v/default.yaml`](configs/n3v/default.yaml) 共同决定。**YAML 会覆盖同名命令行参数。** 默认配置训练 6000 次迭代；使用实际保存的检查点验证，并保持测试集非空。修改总迭代数时也需核对保存和测试列表。

批量训练入口为 `bash train.sh`，运行前必须将脚本中的 `<your_dataset_path>` 替换为本机数据根目录，并核对场景和 GPU。该占位符不是可直接运行的路径。

## 主要文件

| 路径 | 作用 |
| --- | --- |
| [`main.py`](main.py) | 训练、定期评估和验证入口 |
| [`scene/`](scene/) | 场景读取、高斯参数与检查点 |
| [`gaussian_renderer/`](gaussian_renderer/) | 混合高斯渲染与 CUDA 包装 |
| [`arguments/`](arguments/)、[`configs/`](configs/) | 参数定义与实验配置 |
| [`scripts/`](scripts/) | 数据预处理 |
| [`utils/`](utils/) | 相机、损失、图像和渲染工具 |
| `diff-gaussian-rasterization/`、`simple-knn/`、`pointops2/` | CUDA 扩展源码 |

`chkpnt<iteration>.pth` 保存模型及优化器状态；`chkpnt_best.pth` 按定期测试的 PSNR 选择。导出的 PLY 只包含动态池的空间属性，不能替代完整的混合 3D/4D 检查点。

## 上游项目与引用

基础表示来自 Seungjun Oh、Younggeun Lee、Hyejin Jeon、Eunbyung Park 的 *Hybrid 3D-4D Gaussian Splatting for Fast Dynamic Scene Representation*：
[论文](https://arxiv.org/abs/2505.13215) · [项目主页](https://ohsngjun.github.io/3D-4DGS/) · [上游代码](https://github.com/ohsngjun/3D-4DGS)。本仓库的压缩分支扩展与上游原论文贡献分别说明。

![上游混合 3D/4D Gaussian Splatting 方法概览](assets/main.jpg)

```bibtex
@article{oh2025hybrid,
  title={Hybrid 3D-4D Gaussian Splatting for Fast Dynamic Scene Representation},
  author={Oh, Seungjun and Lee, Younggeun and Jeon, Hyejin and Park, Eunbyung},
  journal={arXiv preprint arXiv:2505.13215},
  year={2025}
}
```

感谢 [Real-time 4D Gaussian Splatting](https://github.com/fudan-zvg/4d-gaussian-splatting)、[Ex4DGS](https://github.com/juno181/Ex4DGS)、[4D-Rotor Gaussians](https://github.com/weify627/4D-Rotor-Gaussians) 及 [@sorceressyidi](https://github.com/sorceressyidi) 的相关代码与工作。

## 许可证

根目录包含 [MIT LICENSE](LICENSE)。部分继承源码头部另有上游使用声明，第三方目录也保留自己的许可文件；原始版权与许可声明均予以保留。
