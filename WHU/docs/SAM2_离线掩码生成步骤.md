# 服务器SAM2离线掩码生成

用户已确认：项目 `/project/code/WHU_base/WHU_STMI`，数据 `/project/code/data/WHU-MARS`，分割环境 `wsam`（Python3.10、torch2.5.1+cu124、3090），训练环境 `clipreid`。SAM2源码commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`，权重 `/project/code/third_party/sam2/checkpoints/sam2_hiera_large.pt`。

## 1. 同步新增脚本

将本地 `WHU/tools/generate_sam2_masks.py` 同步至服务器项目的同名相对位置。服务器上此前同步的版本可能没有该文件。此工具不导入UAD、timm或yacs，不需要在clipreid安装SAM2。

## 2. 先运行72张训练集预览

```bash
conda activate wsam
cd /project/code/WHU_base/WHU_STMI/WHU

CUDA_VISIBLE_DEVICES=0 python tools/generate_sam2_masks.py \
  --image-root /project/code/data/WHU-MARS \
  --mask-root /project/code/data/WHU-MARS_masks_preview_v1 \
  --checkpoint /project/code/third_party/sam2/checkpoints/sam2_hiera_large.pt \
  --model-config configs/sam2/sam2_hiera_l.yaml \
  --precision bf16 --preview-per-group 12 --seed 1234
```

配置名是SAM2包内的相对配置标识，不是项目目录的文件路径。该权重对应SAM2配置，不能换成sam2.1配置。GPU编号可换成空闲卡；进程内部使用映射后的cuda:0。

每类RGB/IR/Thermal × ground/aerial从训练集抽12张，固定随机种子。每张独立使用“整图框+中心前景点”作为初始提示，返回多候选后选择SAM2预测IoU最高的一个；这只是针对行人裁剪图的初始空间假设，不等同于语义人体检测。禁用额外后处理、膨胀和空洞过滤；BF16与所有规则记录在provenance中。

掩码PNG位于输出根下镜像的 `train/<模态>/`；`previews/train/<模态>/` 保存原图、二值图、绿色叠加图三联预览。掩码始终为原图尺寸，只有预览图缩放显示。`preview_selection.json`保存选图清单，`generation.jsonl`记录每图预测分数和面积比例，`provenance.json`记录源码commit、权重/config/script哈希与生成规则。

先检查全部72张预览，尤其是IR/Thermal、航拍、偏离中心或遮挡图。目标应为主要行人且背景较少；近乎全前景、只分出一小块或选中背景都需要回看原图。此轮不根据测试集检索结果选择生成规则。

如果质量不好，先停止并统一调整生成规则，再在新的输出目录重跑训练集预览；不要手工修正部分测试掩码后与旧规则混合。

## 3. 预览通过后全量生成

```bash
CUDA_VISIBLE_DEVICES=0 python tools/generate_sam2_masks.py \
  --image-root /project/code/data/WHU-MARS \
  --mask-root /project/code/data/WHU-MARS_masks_sam2_v1 \
  --checkpoint /project/code/third_party/sam2/checkpoints/sam2_hiera_large.pt \
  --model-config configs/sam2/sam2_hiera_l.yaml \
  --precision bf16 --seed 1234
```

不传 `--preview-per-group` 即生成train/query/test所有图像。与预览完全相同的模型/精度/提示/选择规则，不为NIR、TIR复用RGB掩码。按用户当前统计，共应处理192147个图像路径。具体速度、显存和真实分割质量必须以服务器执行为准，本地没有运行SAM2权重。

如中断，原命令末尾增加 `--resume`；脚本要求provenance一致，并验证已有PNG的尺寸与二值编码后跳过。损坏掩码或读取/推理错误会停止，不填全零掩码掩盖错误。正常推理得到的空掩码仍按真实结果保存。不要在同一个mask-root同时启动多个进程。

## 4. 切回clipreid全量审计

```bash
conda activate clipreid
cd /project/code/WHU_base/WHU_STMI/WHU

python tools/audit_masks.py \
  --image-root /project/code/data/WHU-MARS \
  --mask-root /project/code/data/WHU-MARS_masks_sam2_v1 \
  --manifest /project/code/data/WHU-MARS_masks_sam2_v1/manifest.json \
  --encoding 0/255 --build --suffix .png \
  --provenance /project/code/data/WHU-MARS_masks_sam2_v1/provenance.json \
  --report logs/mask_audit_sam2_v1.json
```

已有manifest重新审计时去掉 `--build --suffix --provenance`。全量掩码审计不能对只有72张的预览目录执行，否则会因其余掩码缺失而报错。

## 5. 开始训练

沿用主说明中的单卡/DDP命令，使用：

```text
DATASETS.MASK_ROOT /project/code/data/WHU-MARS_masks_sam2_v1
DATASETS.MASK_MANIFEST /project/code/data/WHU-MARS_masks_sam2_v1/manifest.json
DATASETS.MASK_ENCODING 0/255
```

训练、验证、测试均在clipreid执行，无须加载SAM2权重。

## 数据数量提醒

用户服务器train每模态30711张、共92133张；论文统计每模态30771张、共92313张，相差每模态60张。不要自行增删或重命名数据。先确认原baseline是否使用同一份数据；若是，则本轮模型对照保持一致。与论文做严格数量对照时，应另外核实差异原因。

## 本地检查范围

`python -m unittest discover -s tests -p test_sam2_masks.py -v` 已通过2项辅助逻辑检查；真实SAM2/CUDA推理与人体掩码质量尚待服务器预览验证。
