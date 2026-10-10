# Qwen SuperPos → RFA 梯度迁移

> 当前实现不是 training-free：target prompt 通过无标签 RFA 目标做梯度下降。项目同时
> 提供单次 full-basis `pipeline.py`，以及固定 support 的多 K `sweep_k.py`。本文末尾
> 折叠区里的旧 candidate-selection 代码仅是历史记录，不属于这两个主流程。

## 当前完整方法

1. 在 Qwen2.5-1.5B 上用 SST-2 label 训练 SuperPos source prompt：
   `alpha_s = softmax(z_s / T)`，`P_s = alpha_s E_s`。
2. 检查 1.5B/7B tokenizer 的每个 token ID 一致，并令 `z_t(0) = z_s`，
   `P_t = softmax(z_t / T) E_t`。
3. 固定无标签 probe。分别计算 source 和 target 的
   `delta(v)=log p(v|P,x)-log p(v|x)`；实际代码使用等价的 logit 差，避免重复
   log-softmax。
4. 优化且只优化 `z_t`：
   `L = mean_x JS(softmax(delta_s/tau), softmax(delta_t/tau))
        + lambda KL(alpha_t || alpha_s)`。
5. 单独加载 label，离线比较 direct target prompt 与 refined target prompt 的准确率。

默认使用全部对齐的非 special token。显存或速度受限时可用 source-effect anchor：
按照 source RFA effect distribution `q_s=softmax(delta_s/tau)` 的概率质量选择覆盖
95% 的 token（限制 512–8192），在优化前固定，并加入 `OTHER` 概率质量桶。启用
参数是 `--anchor-mode source-effect`。checkpoint 会保存实际 token ID 和实际覆盖率。

## 一键运行

```bash
python -m qwen_cross_model.pipeline \
  --output-dir outputs/run_seed42 \
  --num-probes 64 \
  --refine-epochs 5 \
  --stay-weight 1.0 \
  --anchor-mode full \
  --dtype bfloat16
```

若已经有本项目或旧 clean 项目的 softmax SuperPos checkpoint，可以跳过 source
训练：

```bash
python -m qwen_cross_model.pipeline \
  --source-checkpoint path/to/best_superpos.pt \
  --output-dir outputs/run_from_checkpoint
```

输出为：

```text
outputs/run_seed42/
├── source/source_prompt.pt
├── source/training_history.json
├── refinement/refined_target.pt
├── refinement/refinement_history.json
└── evaluation/
    ├── target_evaluation.json
    └── target_evaluation.csv
```

单个 K 的 7B refinement 默认且已按代码路径检查的是单卡。`--device-map auto` 只允许纯 GPU
分片，CPU/disk offload 会直接拒绝；该梯度路径在完成真实多卡测试前视为实验功能。
pipeline 用独立进程串行执行各阶段，因此 1.5B 和 7B 不会同时常驻显存。多卡 DDP
主要用于 source 训练：`torchrun --nproc_per_node=N -m
qwen_cross_model.train_superpos --output-dir ...`。

默认从 SST-2 train split 只抽取文本作为 probe，并在 validation 上做最终标签评测，
避免在同一批 validation 输入上 refinement 后再报告其 accuracy。依赖见
`requirements.txt`；refinement checkpoint 同时保存 probe 原文、source checkpoint
SHA-256、模型 revision（仓库提供时）和运行库版本。

分别运行各阶段和全部参数：

```bash
python -m qwen_cross_model.train_superpos --help
python -m qwen_cross_model.refine_target --help
python -m qwen_cross_model.evaluate --help
python -m qwen_cross_model.pipeline --help
```

## 固定 support 的 K sweep 与无标签选 K

旧实验的 direct 最优 K 不应直接沿用，因为每个 K 经 RFA refine 后的最优点可能
改变。新 sweep 固定测试：

```text
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 18 20 24 28 32 40 50 64 96 128
```

每个 prompt 位置单独取 source coordinate logit 的 Top-K，并在整个 refine 过程中
严格固定这 K 个 token；只优化对应的 `[prompt_length, K]` logits。所有 K 都和同一
个 full source prompt functional signature 比较，KL 则相对于该 K 内截断并重新
归一化的 source coordinates。K=1 没有可优化自由度，只评测 direct RFA；K=128
保持原始 basis 顺序，应复现原来的 full direct prompt。

probe 默认从 SST-2 train 文本一次性、确定性地拆成互不重叠的 64 条 refine-train
和 128 条 held-out RFA-validation。先缓存 source full effect、target no-prompt
reference、tokenized probes 及 source-only anchors。最终用 validation RFA 的
one-standard-error rule 选择 K：以最小 mean RFA 对应的 `mean + SE` 为阈值，选择
阈值内最小的 K。此阶段不加载 label 或 accuracy。

四张 GPU 做任务并行（每个 worker 加载一次 7B，并顺序处理自己分到的 K）：

```bash
python -m qwen_cross_model.sweep_k \
  --source-checkpoint /root/outputs_qwen/superpos/pipeline.pt \
  --output-dir /root/outputs_qwen/k_sweep_seed42 \
  --gpus 0 1 2 3 \
  --anchor-mode full \
  --dtype bfloat16 \
  --batch-size 1 \
  --epochs 5
```

`--gpus` 使用当前进程可见的逻辑 CUDA 编号。若作业调度器已经设置
`CUDA_VISIBLE_DEVICES=2,3,4,5`，仍应写 `--gpus 0 1 2 3`；程序不会覆盖调度器的
设备隔离。共享 cache 在一次性子进程中准备，退出并释放 CUDA context 后才启动
各 K worker。

中断后可以在同一命令末尾添加 `--resume`。程序只跳过 checkpoint 和结果文件均
存在、且 cache SHA-256 一致的 K；不会静默接受半成品。若只想先准备共享缓存，
添加 `--cache-only`。

主要输出：

```text
k_sweep_seed42/
├── rfa_sweep_cache.pt
├── k_001/refined_target_k001.pt
├── ...
├── k_128/refined_target_k128.pt
├── k_sweep_rfa.json
├── k_sweep_rfa.csv
└── k_selection.json
```

K 选定后，才在独立脚本中读取 validation labels，统计所有 K 的 direct/refined
accuracy、`Spearman(-RFA, refined accuracy)` 和 refined Top-1 selection regret：

```bash
python -m qwen_cross_model.evaluate_sweep \
  --sweep-dir /root/outputs_qwen/k_sweep_seed42 \
  --output-dir /root/outputs_qwen/k_sweep_seed42_eval \
  --dtype bfloat16 \
  --device cuda:0
```

多卡这里采用 K 级任务并行，不是 DDP；一个 K 内的 prompt 参数不跨卡同步，因此
每张参与的 GPU 都必须能单独容纳一份 7B 模型及该 K 的反向传播显存。

新增 CLI：

```bash
python -m qwen_cross_model.sweep_k --help
python -m qwen_cross_model.evaluate_sweep --help
```

## 已停用的旧 selection 实验

<details>
<summary>仅保留作历史对照；不属于当前方法，也不被 pipeline 调用</summary>

独立实现 Qwen2.5-1.5B → Qwen2.5-7B 的 training-free Relative Functional
Alignment（RFA）实验。项目不会依赖或修改旧的 `prompt-transfer-main*` 目录。

## 方法

每个 SuperPos checkpoint 定义 source prompt：

```text
P_source = alpha @ E_source
P_target = alpha @ E_target
```

对固定无标签 probe `x`，计算：

```text
delta(v) = log p(v | P, x) - log p(v | x)
q(v) = softmax(delta(v) / temperature)
RFA = mean_x JS(q_source, q_target)
```

RFA 越低，表示迁移后的 prompt 越好地保留了 source prompt 对生成分布造成的
相对影响。代码不读取 label 或 accuracy，不更新模型，不反向传播，也不优化
prompt。

## 输入 checkpoint

`--checkpoints-dir` 直接放多次 source SuperPos 训练得到的 checkpoint：

```text
outputs/source_prompts/
├── seed_1.pt
├── seed_2.pt
├── seed_3.pt
└── seed_4.pt
```

每个文件至少需要：

```python
{
    "prompt_weights": Tensor[prompt_length, basis_size],
    "sampled_ids": Tensor[basis_size],
    "weight_mode": "softmax",  # 也支持 raw / l1_normalized_raw
    "temperature": 0.5,         # softmax checkpoint 必须保存
    "seed": 1,
    "source_model": "Qwen/Qwen2.5-1.5B",  # 推荐保存
}
```

程序会从每个 checkpoint 自动得到 source prompt 和 migrated target prompt；不需要
另行保存 `[100, 3584]` target tensor。多个 `--bridge-k` 会从每个 checkpoint
产生多个 target candidate，例如 `seed_1.pt::k=10`。

同一个 checkpoint 的不同 K 严格共享同一个 source functional signature。不同
source seed 则分别与各自的 source signature 比较，因此此时 RFA 衡量的是每次
训练结果的“迁移保真度”；它不能替代 source prompt 本身的任务质量检查。

老 checkpoint 如果确实没有保存 temperature，必须显式传
`--legacy-temperature 0.5`；程序不会静默猜测。

## 主实验：完整共享词表

```bash
python -m qwen_cross_model.rfa \
    --checkpoints-dir outputs/source_prompts \
    --source-model Qwen/Qwen2.5-1.5B \
    --target-model Qwen/Qwen2.5-7B \
    --bridge-k 1 5 10 128 \
    --num-probes 128 \
    --anchor-mode full \
    --temperature 2.0 \
    --dtype bfloat16 \
    --device cuda:0 \
    --batch-size 1 \
    --output-dir outputs/rfa_full
```

程序会逐 ID 验证两个 tokenizer 完全一致，排除 special/control token，并在全部
对齐的普通 token 上计算 RFA。source 模型完成后会被释放，之后才加载 target
模型，因此 1.5B 和 7B 不会同时占显存。

多卡模型分片可将设备参数替换为：

```bash
--device-map auto --batch-size 1
```

## 稀疏 source-only anchor

```bash
python -m qwen_cross_model.rfa \
    --checkpoints-dir outputs/source_prompts \
    --source-model Qwen/Qwen2.5-1.5B \
    --target-model Qwen/Qwen2.5-7B \
    --bridge-k 1 5 10 128 \
    --num-probes 128 \
    --anchor-mode source-js \
    --anchor-coverage 0.95 \
    --min-anchor-tokens 512 \
    --max-anchor-tokens 8192 \
    --temperature 2.0 \
    --dtype bfloat16 \
    --device cuda:0 \
    --batch-size 1 \
    --output-dir outputs/rfa_sparse
```

对每个 probe，代码先计算所有 source checkpoint 的 prompted/reference 逐 token
JS contribution，再取均值。按贡献从高到低选择覆盖 95% source functional JS
的最小集合，并限制为 512–8192 个 token。

这个 anchor 集合具有三个约束：

- 只看 source 模型，不看任何 target candidate；
- target 评分开始前固定，所有 candidate 完全共用；
- RFA 分布加入 `OTHER` 桶，保留未选 token 的总概率质量。

因此它不再是 candidate-dependent 的 source/target Top-K union。

## Probe bank

默认从 SST-2 validation 的 `sentence` 列按 seed 固定抽取 128 条，只读取文本列，
不读取 `label`。这种设置是 training-free，但不是 data-free。

也可以使用自己的无标签 TXT、JSON 或 JSONL：

```bash
--probe-file probes/sst2_unlabeled.txt
```

TXT 每行一个输入；JSON/JSONL 可以是字符串，也可以是含 `text` 字段的对象。
报告记录 probe 索引、SHA-256、seed 和全部评分参数。

## Accuracy 离线验证

RFA selection 完成后再准备 accuracy 文件。candidate 名必须和 selection CSV 中的
`candidate` 完全一致：

```csv
candidate,accuracy
seed_1.pt::k=10,0.8429
seed_2.pt::k=10,0.8555
```

然后运行：

```bash
python -m qwen_cross_model.analyze_rfa \
    --selection-json outputs/rfa_full/rfa_selection_<timestamp>.json \
    --accuracy-file outputs/candidate_accuracy.csv \
    --output-dir outputs/rfa_analysis
```

输出：

```text
Spearman(-RFA, accuracy)
Top-1 selection regret = oracle_best_accuracy - selected_accuracy
```

accuracy 分析代码与 RFA selection 完全分离，不能改变排名。

## 验证

以下验证不下载模型或数据：

```bash
python -m unittest discover -s qwen_cross_model -p "test_*.py" -v
python -m compileall qwen_cross_model
python -m qwen_cross_model.rfa --help
python -m qwen_cross_model.analyze_rfa --help
```

</details>
