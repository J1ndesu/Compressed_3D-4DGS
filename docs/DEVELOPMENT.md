# 注释与文档维护约定

## 语言与格式

- 面向使用者的 README 使用中文，参数名、标识符和标准术语保留英文。
- 项目维护的 Python 注释与 docstring 使用英文。复杂接口按摘要、Args、Returns 和必要限制组织，简单函数只写一句准确摘要。
- 注释解释算法意图、张量形状、边界条件和状态变化，避免逐行复述代码。使用普通井号注释，不使用装饰性分隔线。
- 保留版权与许可头部。第三方代码遵循原有风格，不为统一语言批量改写。
- 清理失效调试语句和大段注释代码；可配置的脚本场景列表保留。

## 术语

| 术语 | 约定 |
| --- | --- |
| mask logit | Sigmoid 之前的可训练值；零 logit 对应概率 0.5 |
| soft mask | sigmoid(logit) 的输出 |
| hard mask / STE gate | 前向使用严格阈值比较，反向使用替代梯度 |
| Gaussian pruning | 删除点及对应参数、掩码和统计量行 |
| SH pruning | 独立置零空间 SH 分带，张量形状不变 |
| dynamic / static pool | 动态 4D 高斯池 / 静态 3D 高斯池 |
| dense / effective size | 分配张量的参数载荷 / 掩码选择后的参数载荷估计 |

完整 SH 张量 (N, C, 3) 包含 DC，1–3 阶切片为 1:4、4:9、9:16；features_rest 不含 DC，因此切片为 0:3、3:8、8:15。额外通道不受这三个掩码控制。

prune_points(mask) 和 prune_static_points(mask) 的 True 表示删除；内部 _prune_optimizer(mask) 的 True 表示保留。

## 文档核对

1. 同时检查当前分支的 arguments、YAML 和 main.py，不把参数类默认值当作最终值；YAML 会覆盖同名 CLI 参数。
2. 示例使用本仓库的正确分支与真实入口。新增功能应能在调用链中定位；预留开关不写成已实现能力。
3. 大小报告区分参数载荷、实际文件大小与显存，标明单位和是否包含掩码、优化器及索引开销。
4. 实验结果记录场景、分支/提交、配置、检查点、阈值及指标实现。手工绘图数据不能替代实验记录。
5. 注释与文档修改应确认可执行内容未改变；具备 Python 环境时，去除 docstring 后比较修改前后的 AST。

可在仓库根目录运行：

~~~bash
git diff --check
bash -n train.sh
bash -n val.sh
python -m compileall -q main.py arguments scene gaussian_renderer utils scripts curve.py
~~~

静态检查不导入 CUDA 扩展，也不能证明训练、断点恢复或渲染正确。未运行的验证步骤应如实记录。
