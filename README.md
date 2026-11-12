# MGDiag

MGDiag 是一个面向微服务系统的多模态故障诊断项目。项目将指标（Metric）、调用链（Trace）和日志（Log）转换为离散异常事件，再把服务、Pod、节点及其依赖关系构造成图，最终同时完成：

- 根因定位（Root Cause Localization，RCL）；
- 故障类型识别（Failure Type Identification，FTI）。

当前仓库已经接入 `data/aiops2025`：共 400 个故障案例，其中训练集 320 个、测试集 80 个，包含 18 种故障类型；327 个案例为单根因，73 个网络类案例包含两个根因服务。

## 1. 项目目录

```text
MGDiag/
├── config/
│   └── exp_config.py              # 模型和训练配置
├── core/
│   ├── MGDiag.py                  # 训练、损失计算和测试评估
│   ├── multimodal_dataset.py      # DGL 图数据集
│   ├── model/                     # 编码器、分类器和投票器
│   └── loss/                      # 对比学习及动态加权损失
├── data/
│   └── aiops2025/
│       ├── train/                 # 320 个训练案例的 Parquet 原始遥测
│       ├── test/                  # 80 个测试案例的 Parquet 原始遥测
│       ├── ground_truth.json      # 根因和故障类型标签
│       ├── deploy.parquet         # Pod、服务、节点部署关系
│       ├── extractor_config.json  # 事件抽取阈值
│       ├── dataset_config.json    # 数据集专属训练参数
│       ├── artifacts/             # 训练集 normal 数据拟合出的抽取器
│       ├── raw/                   # 离散事件、图节点和图边
│       ├── tmp/                   # 每个图节点的多模态 FastText 向量
│       └── label.csv              # 统一后的训练/测试标签
├── extractor/
│   ├── pipeline.py                # 原始遥测预处理入口
│   ├── aiops2025_adapter.py       # AIOps2025 字段、时间和拓扑适配
│   ├── events.py                  # Metric、Trace、Log 事件抽取
│   ├── base.py                    # 数据集适配器接口
│   └── registry.py                # 数据集适配器注册表
├── process/
│   ├── EventProcess.py            # 事件编码和图数据集构建
│   └── events/fasttext_w2v.py     # FastText 事件编码
├── helper/                        # 评估、日志、早停等工具
├── logs/                          # 模型、TensorBoard 和预测结果
├── tmp/                           # FastText 训练语料（不是节点向量缓存）
└── main.py                        # 训练和评估入口
```

## 2. 端到端处理流程

```text
train/test UUID 目录中的 Parquet + metadata.json
                         │
                         ▼
              AIOps2025Adapter 标准化
       ┌─────────────────┴─────────────────┐
       │                                   │
       ▼                                   ▼
训练集 normal 窗口                   每个案例的 normal/abnormal 窗口
       │                                   │
       ▼                                   │
拟合 Metric/Trace/Drain 抽取器              │
       │                                   │
       └──────── artifacts/ ───────────────┘
                         │
                         ▼
             抽取 Metric/Trace/Log 事件
             构建 service↔pod↔node 图
                         │
                         ▼
       label.csv + raw/{events,nodes,edges}.json
                         │
                         ▼
           仅用训练集事件训练三个 FastText
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
      编码全部训练案例          编码全部测试案例
             └───────────┬───────────┘
                         ▼
        tmp/{metric,trace,log}.pkl 节点向量
                         │
                         ▼
                   构建 DGL 图
             ┌───────────┴───────────┐
             ▼                       ▼
     train_data + aug_data        test_data
             │                       │
             ▼                       │
       训练 MGDiag 权重               │
             └───────────┬───────────┘
                         ▼
            测试集推理、指标计算和预测导出
```

整个流程分为两条命令：

1. `extractor.pipeline`：Parquet → 抽取器 → 离散事件和拓扑；
2. `main.py`：离散事件 → FastText 节点向量 → DGL 图 → MGDiag 训练与测试。

## 3. 原始数据标准化

运行 `extractor.pipeline` 后，`AIOps2025Adapter` 会完成以下处理：

1. 扫描 `train/<uuid>` 和 `test/<uuid>`；
2. 从 `ground_truth.json` 读取根因、根因层级和故障类型；
3. 从每个案例的 `metadata.json` 读取 normal 与 abnormal 时间窗口；
4. 将所有时间统一解析为 `Asia/Shanghai` 时区；
5. 将 service/node 指标宽表转换成长表；
6. 将 Pod 指标统一成 `time/entity/entity_type/metric/value`；
7. 将 Trace 的 `status_cod` 统一为 `status_code`；
8. 把每个案例切分成 normal 与 abnormal 两部分。

每个 `CaseData` 最终包含：

```text
normal_metrics    abnormal_metrics
normal_traces     abnormal_traces
normal_logs       abnormal_logs
```

normal 是故障前的案例内上下文，abnormal 是待诊断的故障时间段。

## 4. 抽取器拟合：只有训练集 normal 参与

`extractor.pipeline.fit_extractors()` 只筛选 `split == "train"` 的 320 个案例，并且只使用这些案例的 normal 窗口。测试集不会更新任何全局抽取器参数。

### 4.1 Metric 基线

对每个 `(entity_type, entity, metric)` 汇总训练集 normal 指标。计数器类指标（名称以 `_total` 或 `_count` 结尾）先转换为非负差分，再计算稳健统计量：

```text
median：正常值中位数
scale：1.4826 × MAD；退化时回退到标准差或最小尺度
count：参与统计的有效样本数
```

同时建立 `(entity_type, *, metric)` 回退基线。当测试实体没有专属基线时，可以使用同层级、同指标的共享基线。

结果保存为：

```text
data/aiops2025/artifacts/metric_baseline.pkl
```

### 4.2 Trace 基线

首先通过 `(trace_id, parent_span_id) -> (trace_id, span_id)` 自连接恢复父子调用，形成：

```text
source service → destination service → operation
```

然后以默认 30 秒窗口聚合：

- `latency`：调用时延 P95；
- `error_ratio`：错误调用比例；
- `call_count`：调用次数。

每种特征同样使用 `median/scale/count` 建立稳健基线，并提供 operation 级和全局回退基线。结果保存为：

```text
data/aiops2025/artifacts/trace_baseline.pkl
```

### 4.3 Drain 日志模板

Drain 只接收训练集 normal 日志：

- JSON 日志优先提取内部的 `message` 或 `msg`；
- 统计每个模板的正常出现次数；
- 标记低频模板；
- 标记包含 `error/fail/exception/timeout/fatal/panic` 的模板。

结果保存为：

```text
data/aiops2025/artifacts/drain.pkl
```

## 5. 事件抽取：训练集和测试集都参与 transform

抽取器拟合完成后，程序会遍历全部 400 个案例。训练案例和测试案例都使用相同且已经冻结的全局抽取器。

### 5.1 Metric 事件

abnormal 指标与训练集 normal 基线比较：

```text
score = (value - median) / scale
```

默认情况下，满足以下任一条件即生成事件：

- 至少 2 个采样点的 `|score| >= 3.5`；
- 单点严重异常 `|score| >= 6.0`。

输出格式：

```json
["shippingservice-0", "cpu_usage_rate", "up"]
```

案例自身的 normal 指标只用于计数器在 normal→abnormal 边界上的首个差分，不会更新全局基线。

### 5.2 Trace 事件

对每个案例的 abnormal Trace 计算 30 秒窗口特征，与训练集基线比较，生成：

```json
["checkoutservice", "paymentservice", "/Charge", "latency_up"]
["frontend", "adservice", "/GetAds", "error_up"]
["frontend", "cartservice", "/GetCart", "call_count_down"]
```

案例自身的 normal Trace 还用于识别“normal 中存在、abnormal 中完全消失”的调用关系。这是单案例上下文比较，不是用测试集重新拟合基线。

### 5.3 Log 事件

冻结后的 Drain 对 normal 和 abnormal 日志做模板匹配，并按“每分钟出现次数”比较。如果模板满足以下任一条件，就生成日志事件：

- 无法匹配训练模板，模板 ID 为 `-1`；
- 属于训练集 normal 中的低频模板；
- 属于错误关键词模板；
- abnormal 出现率相对该案例 normal 明显升高。

输出格式：

```json
["shippingservice-2", "17"]
```

这里会读取测试案例自身的 normal 日志作为在线上下文，但不会修改 Drain 模板树或训练统计。

## 6. 拓扑和标签构建

每个训练/测试案例都会独立构建一张图。

### 6.1 图节点和边

节点来自：

- `deploy.parquet` 中与案例时间窗口重叠的 service、pod、node；
- normal/abnormal 指标中出现的实体；
- normal/abnormal 日志中的 Pod 和节点；
- normal/abnormal Trace 中的 service、pod 和 node；
- Trace 恢复出的服务调用关系。

边包括：

```text
service ↔ pod
pod ↔ node
caller service ↔ callee service
```

边在 `raw/edges.json` 中保存为案例局部节点下标；进入 DGL 后会去掉旧自环，并为所有节点统一添加自环。

### 6.2 多根因标签

`label.csv` 的主要字段为：

| 字段 | 含义 |
|---|---|
| `index` | 案例 UUID |
| `data_type` | `train` 或 `test` |
| `instances_json` | 根因实体数组，支持一个或多个根因 |
| `instance_type` | `service`、`pod` 或 `node` |
| `anomaly_type` | 故障类型 |

图上的 `root` 是多热标签。单根因案例有一个值为 1 的节点；网络类双根因案例有两个值为 1 的服务节点。

### 6.3 `raw/` 输出

```text
raw/metrics.json      每个案例的指标事件
raw/traces.json       每个案例的调用链事件
raw/logs.json         每个案例的日志模板事件
raw/nodes.json        每个案例的有序节点名称
raw/edges.json        每个案例的局部下标边
raw/node_types.json   节点名称到 service/pod/node 的映射
```

六个 JSON 都以案例 UUID 为键，并同时包含 320 个训练案例和 80 个测试案例。

## 7. FastText 事件编码

执行 `main.py --reconstruct` 时，`EventProcess` 为 Metric、Trace、Log 分别训练一个 FastText 编码器。

### 7.1 FastText 训练阶段

只有训练集事件进入 FastText 训练语料。对每个训练案例的每个图节点：

- 节点是根因：标签为“节点 + 当前故障类型”；
- 节点不是根因：标签为“节点 + normal”；
- 空事件文档使用 `__empty__` 训练占位符。

FastText 训练文本保存在根目录 `tmp/`：

```text
tmp/metric-train.txt
tmp/metric-train-aug.txt
tmp/trace-train.txt
tmp/trace-train-aug.txt
tmp/log-train.txt
tmp/log-train-aug.txt
```

这些文件是训练语料，不是 MGDiag 的节点向量。

### 7.2 全部案例编码阶段

FastText 训练完成后，冻结词向量，再分别编码训练集和测试集。每个节点得到默认 128 维向量；没有对应事件的节点得到全零向量。

结果保存为：

```text
data/aiops2025/tmp/metric.pkl
data/aiops2025/tmp/trace.pkl
data/aiops2025/tmp/log.pkl
```

每个文件的逻辑结构为：

```python
{
    "case_uuid": [
        node_0_embedding,
        node_1_embedding,
        # ...
    ]
}
```

测试集只被编码，不用于训练或更新 FastText 参数。

## 8. DGL 图数据集构建

`EventProcess.build_dataset()` 将三个模态的节点向量挂载到图上：

```text
graph.ndata["metric"]
graph.ndata["trace"]
graph.ndata["log"]
graph.ndata["root"]
graph.ndata["node_type"]
```

随后根据 `label.csv.data_type` 分流：

```text
train → train_data
test  → test_data
```

只有 `train_data` 会执行节点丢弃增强。增强过程保护所有真实根因节点，不会把单根因或双根因标签删除。增强图保存于内存中的 `aug_data`，不会写回原始数据。

## 9. MGDiag 模型训练

模型使用 `train_data` 和从训练集生成的 `aug_data`，测试集不参与反向传播、优化器更新或早停判断。

训练目标包括：

- RCL：多正样本根因定位损失；
- FTI：故障类型交叉熵；
- TO：任务导向监督对比损失；
- CM：跨模态无监督对比损失；
- Dynamic Weight：可学习的多任务损失权重。

其中多根因定位损失为：

```text
logsumexp(全部节点分数) - logsumexp(所有真实根因节点分数)
```

单根因时等价于普通 softmax 交叉熵；双根因时允许模型把概率质量分配给任一或两个真实根因。

训练阶段会根据训练损失执行 early stopping，并保存：

```text
logs/aiops2025/TVDiag.pt
```

## 10. 测试集评估

训练完成后加载 `TVDiag.pt`，逐案例对 `test_data` 前向推理。测试阶段：

- 不执行反向传播；
- 不更新模型权重；
- 不更新 Metric/Trace 基线；
- 不更新 Drain；
- 不更新 FastText。

根因定位指标：

| 指标 | 含义 |
|---|---|
| `HR@K` | Top-K 是否命中至少一个真实根因 |
| `RootRecall@K` | Top-K 覆盖的真实根因比例，适用于双根因案例 |
| `MRR@3` | 第一个真实根因在前三名内的倒数排名 |
| `avg@3` | HR@1、HR@2、HR@3 的平均值 |

故障类型指标采用加权 `precision`、`recall` 和 `F1`。

每个测试案例的结果导出到：

```text
logs/aiops2025/predictions.csv
```

主要字段包括预测根因、预测根因层级、预测故障类型、Top-5 根因、真实根因和真实故障类型。

## 11. 训练集和测试集参与环节汇总

| 环节 | 训练集 | 测试集 | 是否产生可学习参数 |
|---|---|---|---|
| 读取 Parquet、切分 normal/abnormal | 参与 | 参与 | 否 |
| 拟合 Metric 正常基线 | 仅 normal 参与 | 不参与 | 是 |
| 拟合 Trace 正常基线 | 仅 normal 参与 | 不参与 | 是 |
| 训练 Drain 模板 | 仅 normal 参与 | 不参与 | 是 |
| 抽取 abnormal Metric 事件 | 参与 | 参与，仅 transform | 否 |
| 抽取 abnormal Trace 事件 | 参与 | 参与，仅 transform | 否 |
| 比较案例内 normal/abnormal 日志率 | 参与 | 参与，作为单案例上下文 | 否 |
| 构建案例拓扑 | 参与 | 参与 | 否 |
| 训练 FastText | 参与 | 不参与 | 是 |
| 使用冻结 FastText 生成节点向量 | 参与 | 参与，仅 transform | 否 |
| 生成图增强样本 | 参与 | 不参与 | 否 |
| 训练 MGDiag | 参与 | 不参与 | 是 |
| Early stopping | 参与，依据训练损失 | 不参与 | 间接影响训练轮次 |
| 最终推理和指标计算 | 不参与 | 参与 | 否 |

换言之，测试集可以经过与训练集相同的确定性预处理，也可以使用自身 normal 窗口作为当前案例的历史上下文，但不能用于拟合全局检测器、日志模板、FastText 或 MGDiag 权重。

## 12. 当前离线评测协议说明

当前实现面向有完整 ground truth 的离线监督评测，因此有以下行为需要明确：

1. `extractor.pipeline` 会读取训练集和测试集的 `ground_truth.json`，以生成统一 `label.csv`；
2. 构建拓扑时会校验真实根因是否在候选节点中，并在遥测/部署数据缺失时将真实根因作为候选节点兜底加入；
3. `EventProcess` 当前从完整 `label.csv` 推导故障类型全集，并为全部案例建立根因集合编号。

这些操作不会让测试遥测参与参数拟合，但使用了测试标签，因此属于“离线闭集评测协议”，不能直接等同于无标签生产推理。

如果部署到生产环境，应当：

- 只从训练标签确定故障类型集合；
- 只从部署关系和实际遥测构造候选节点，不使用真实根因兜底；
- 推理输入不包含 `instances_json` 和 `anomaly_type`；
- 将预测结果作为输出，而不是再读取 ground truth 做指标计算。

## 13. 安装与运行

项目当前以 Python 3.8、PyTorch 2.1.1、CUDA 11.8 和 DGL 2.2.1 为目标环境。

安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 13.1 重新生成抽取器、事件和拓扑

```powershell
python -m extractor.pipeline --dataset aiops2025
```

等价的显式路径写法：

```powershell
python -m extractor.pipeline `
  --dataset aiops2025 `
  --input data/aiops2025 `
  --output data/aiops2025
```

这一步会重建：

```text
artifacts/*.pkl
raw/*.json
label.csv
```

### 13.2 首次训练或重建事件向量

```powershell
python main.py --dataset aiops2025 --reconstruct
```

`--reconstruct` 会重新训练 FastText 并覆盖 `data/aiops2025/tmp/*.pkl`，不会重新执行 Parquet 事件抽取。

### 13.3 复用已有节点向量

```powershell
python main.py --dataset aiops2025
```

此时直接复用 `data/aiops2025/tmp/*.pkl`，然后重新训练和评估 MGDiag。

### 13.4 快速冒烟测试

```powershell
python main.py `
  --dataset aiops2025 `
  --epochs 1 `
  --aug-times 0 `
  --log-dir logs/aiops2025-smoke
```

如果尚未生成 `data/aiops2025/tmp/*.pkl`，需要同时添加 `--reconstruct`。

## 14. 配置文件

### `extractor_config.json`

控制事件抽取：

| 配置 | 默认值 | 含义 |
|---|---:|---|
| `metric_threshold` | 3.5 | 普通指标异常阈值 |
| `metric_severe_threshold` | 6.0 | 单点严重异常阈值 |
| `metric_min_points` | 2 | 普通异常最少采样点数 |
| `trace_threshold` | 3.5 | Trace 异常阈值 |
| `trace_window` | `30s` | Trace 聚合窗口 |
| `log_rare_quantile` | 0.2 | 日志低频模板分位数 |
| `log_rate_multiplier` | 3.0 | abnormal/normal 日志率倍数阈值 |
| `log_rate_delta` | 0.5 | abnormal 相对 normal 的最小增量 |

### `dataset_config.json`

覆盖数据集专属模型参数。AIOps2025 当前配置为：

```json
{
  "batch_size": 128,
  "patience": 20,
  "feat_drop": 0.1,
  "aggregator": "mean"
}
```

故障类型数量 `ft_num` 不需要手工填写，`EventProcess` 会根据标签自动设置。

## 15. 缓存重建规则

| 修改内容 | 需要重建 artifacts/raw | 需要 `--reconstruct` | 需要重训 MGDiag |
|---|---:|---:|---:|
| 修改 Parquet、时间窗口或 ground truth | 是 | 是 | 是 |
| 修改事件抽取阈值 | 是 | 是 | 是 |
| 修改 Metric/Trace/Log 抽取逻辑 | 是 | 是 | 是 |
| 只修改 FastText 参数或事件拼接方式 | 否 | 是 | 是 |
| 只修改图增强参数 | 否 | 否 | 是 |
| 只修改 MGDiag 网络或训练参数 | 否 | 否 | 是 |

## 16. 接入其他数据集

新增数据集时，实现 `DatasetAdapter` 的三个接口：

```python
cases()          # 返回案例、split、时间窗口和标签
load_case()      # 转换为统一的 Metric/Trace/Log DataFrame
build_topology() # 返回节点、局部下标边和节点类型
```

然后使用注册装饰器：

```python
@register_adapter("new_dataset")
class NewDatasetAdapter(DatasetAdapter):
    ...
```

文件命名为 `extractor/new_dataset_adapter.py` 后，即可执行：

```powershell
python -m extractor.pipeline --dataset new_dataset
python main.py --dataset new_dataset --reconstruct
```

## License

项目代码许可证见 [LICENSE](./LICENSE)。AIOps Challenge 2025 数据集遵循其数据目录中说明的许可协议。
