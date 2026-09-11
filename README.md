# prompt-transfer

这是整理后的独立工作副本。原项目目录保持不变。

## 代码结构

- `qwen_cross_model/projector.py`：projector 主线的模型、训练和评估实现。
- `qwen_cross_model/superpos_trainer.py`：源 prompt 和 SuperPos prompt 训练。
- `qwen_cross_model/pipeline.py`：SuperPos prompt 训练、迁移和总流程入口。
- `qwen_cross_model/eval_direct.py`：纯 `prob @ target_embedding` 的直接迁移评估。
- `qwen_cross_model/eval_soft_bridge.py`：basis constrained 的 soft bridge 评估。
- `qwen_cross_model/eval_baseline.py`：普通 baseline 评估。
- `qwen_cross_model/align.py`：projector 和 soft bridge 共用的 embedding 提取与对齐工具。
- `qwen_cross_model/utils.py`：共享的 prompt、checkpoint、数据和复现工具。

direct 和 soft 两条迁移路线分别由 `eval_direct.py` 和
`eval_soft_bridge.py` 完整负责；不再保留单独的 bridge 实现文件。

## 服务器训练命令

下面的命令会通过旧兼容入口启动同一条 SuperPos projector 训练逻辑。输出文件名
跟随整理后的目录名：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 HF_ENDPOINT=https://hf-mirror.com \
nohup torchrun --nproc_per_node=4 --master_port=29501 \
    -m qwen_cross_model.projector \
    --src-prompt outputs_qwen/superpos_prompt_transfer_main_clean/prompt_transfer_main_clean.pt \
    --src-model Qwen/Qwen2.5-1.5B --tgt-model Qwen/Qwen2.5-7B --dataset sst2 \
    --output-dir outputs_qwen/prompt_transfer_main_clean \
    --save-name prompt_transfer_main_clean.pt \
    > outputs_qwen/prompt_transfer_main_clean/prompt_transfer_main_clean.log 2>&1 &
```

如果使用 `pipeline.py --superpos` 先训练源 prompt，则沿用原参数和
`--save-name`；checkpoint 会保存 SuperPos temperature、seed 和 sampled token
信息，评估时自动读取这些元数据。

从头训练源 SuperPos prompt 时，命令为：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 HF_ENDPOINT=https://hf-mirror.com \
nohup torchrun --nproc_per_node=4 --master_port=29501 \
    -m qwen_cross_model.pipeline \
    --superpos --src Qwen/Qwen2.5-1.5B --dataset sst2 --prompt-epochs 5 \
    --save-name prompt_transfer_main_clean.pt \
    --output-dir outputs_qwen/superpos_prompt_transfer_main_clean \
    > outputs_qwen/superpos_prompt_transfer_main_clean/prompt_transfer_main_clean.log 2>&1 &
```

## 验证

```bash
python -m compileall qwen_cross_model
```

该命令只做依赖无关的语法检查。模型和数据下载不作为静态验证步骤。
