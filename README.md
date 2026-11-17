# RoboCodify

**Reusable Programs for Robotic Agents**

RoboCodify 探索将机器人 Agent 可复用的计算、观察与恢复流程沉淀为程序，
在有限交互预算下提高任务完成能力。当前是基于 **PyRUA-Lean** 的研究工作区，
不是新的机器人控制器，也不是已经验证能提高成功率的完整方法。

当前实现是 `StaticRecovery`：由主 Agent 判断失败、解释图像与选择补救；
程序维护错误记录、局部动作状态、静态观测、历史帧与位姿引用。
动作、原有检查和任务执行仍复用 PyRUA-Lean。

## 当前边界

- 已实现：失败记录、动作定位、暂定诊断、静态观察、指定历史帧回看、重试计划与结果记录。
- 尚未实现：自动发现关键帧、侧方观察臂路径规划、自动积累跨任务程序、完整动态诊断。
- 本次仓库整理保持恢复模块与附录字节不变；接入脚本去除了固定机器和任务路径。
- 已有结果使用历史 Lean 输入、Codex runtime 和 Isaac Sim 6 兼容环境；
  部分任务有单独物理兼容适配，**不是严格原版复现或当前官方规则下的成绩**。
- 不把 Agent 自报成功当作原生成功，不将不同版本的开发实验合并成方法成功率。

## 文件入口

| 内容 | 位置 |
| --- | --- |
| 当前程序 | [src/static_recovery.py](src/static_recovery.py) |
| 当前提示词与外部接入 | [integrations/pyrua_lean](integrations/pyrua_lean) |
| 具体流程、接口与限制 | [docs/implementation.md](docs/implementation.md) |
| 用户研究想法 | [docs/research_ideas.md](docs/research_ideas.md) |
| 复现、实验与版本约定 | [docs/reproduction.md](docs/reproduction.md) |
| 配置、结果与外部证据索引 | [experiments](experiments) |
| 来源、文件哈希 | [provenance.json](provenance.json)、[NOTICE](NOTICE) |

## 不启动机器人的测试

以下命令从仓库根目录执行。需 Python 3.10+ 和 Git；此次实际验证环境是
Windows / Python 3.13。测试使用固定作者源码，不需要历史日志、GPU、模型或凭据。

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-test.txt
git clone --filter=blob:none --no-checkout https://github.com/DAGroup-PKU/PyRUA-Lean.git .upstream/PyRUA-Lean
git -C .upstream/PyRUA-Lean checkout --detach e754cf43c822b91f4728777a78071fa4e012c371
python -m pytest -q
```

已有 `.upstream/PyRUA-Lean` 时先核对提交和工作区，不重复克隆或覆盖。
`.upstream/`、虚拟环境和运行产物不提交。
集成测试缺少固定上游依赖时会失败，不会静默跳过。

## 接入已有 Lean 环境

先按作者文档安装固定版本的 PyRUA-Lean 及对应机器人环境。
本仓库不会自动安装模拟器、下载资产、修改作者代码或启动服务。

在相同环境中，使用原作者命令运行 baseline；method 将入口替换为：

```sh
python integrations/pyrua_lean/run.py --help
```

将原来的 `python -m pyrualean.play` 替换成上面的脚本路径，保留实验所需的
全部原参数，并使用 `--arm cells`。每轮使用独立的 `--out`。
脚本复制当前模块到 episode workspace，并在原始提示末尾增加一次附录，
保存 `baseline-prompt.txt` 和 `static-recovery-installation.json`。
模拟器、凭据、模型入口与路径继续由作者 CLI 或外部环境配置。
本次只验证了这个可移植入口的本地行为，尚未用它重新运行机器人实验。

## 来源与结果声明

初始基线为 [PyRUA-Lean](https://github.com/DAGroup-PKU/PyRUA-Lean)，固定提交
`e754cf43c822b91f4728777a78071fa4e012c371`。
交互式 cell、原语、观察测量及其操作知识归属于上游；本项目不将这些能力声称为新贡献。
具体引用和衍生关系见 [NOTICE](NOTICE)。

历史开发记录包含失败、错误自评、环境故障和用户中止；不因发布而删除这些结果。
本仓库收录的是带来源哈希的摘要与索引，完整视频、日志和旧源码保留在外部档案中，
当前没有公共下载地址。研究者仅凭本仓库不能重放全部历史运行。

代码采用 Apache-2.0；上游声明见 [LICENSE](LICENSE) 和
[third_party/PyRUA-Lean-NOTICE](third_party/PyRUA-Lean-NOTICE)。
