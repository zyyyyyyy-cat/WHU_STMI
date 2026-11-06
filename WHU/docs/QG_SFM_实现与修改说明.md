# UAD + QG-SFM 实现与修改说明

最新训练准备复核见 [训练前复核与运行步骤](训练前复核与运行步骤.md)：包含 R08 正式生成、300 张像素等价校验、全量审计和真实 CUDA/NCCL 短步入口；以下早期通用掩码路径示例应以该文档的冻结 R08 路径为准。

日期：2026-11-03。实现依据优先级：本目录上级 `guide.md` > 提供的 STMI 论文 > STMI 源码。当前 `WHU/` 为主框架，`STMI/`、两篇 PDF 和 `guide.md` 未修改。

## 1. 已实现的模型

```text
每图像独立的 Image + Binary Mask
  -> 联合 Resize / Flip / Pad / Crop
  -> 图像 Normalize + 原 RandomErasing；掩码不擦除
  -> 掩码 AvgPool16/stride16 -> 128维覆盖率 rho、有效性 v
  -> 原共享 ViT-B/16：128 Patch + 1 CLS，768维
  -> 12个 Transformer Block：原 QKV -> QG-SFM -> Scale -> Softmax -> AV
  -> 原 Final LayerNorm -> 768维 pre-BN CLS f
       |-> 原 BNNeck -> 共享 Classifier -> CE
       |-> WRT
       |-> ProCA + GPD
  -> 测试：pre-BN CLS -> 原评测器 L2归一化与距离计算
```

三模态不是三个分支；训练时仍按 RGB、IR、Thermal 在 batch 维拼接。每张图像输出自己的768维描述子，测试的 Query 和 Gallery 都使用掩码。

设原始、尚未缩放的 logits 为 `S=QK^T`，`p=[1,rho]`。每层两个原始标量 `a_l,b_l`，全部初始化为0：

\[
\alpha_l=0.2\tanh(a_l/0.2),\qquad
\beta_l=0.2\tanh(b_l/0.2).
\]

\[
\Delta S_{b,h,i,j}=v_b p_{b,i}
\left[\alpha_l p_{b,j}(S^{max}_{b,h,i}-S_{b,h,i,j})
-\beta_l(1-p_{b,j})(S_{b,h,i,j}-S^{min}_{b,h,i})\right].
\]

\[
A=\operatorname{softmax}((S+\Delta S)/\sqrt{64}),\qquad O=AV.
\]

行极值沿 Key 维计算、保留梯度。调制及行极值使用 FP32、非原地计算，再转回原 logits 类型后使用原缩放、Softmax、Dropout、Projection。原 QK 矩阵乘法仍服从 UAD 的 AMP 设置；FP32转换不能修复已经发生的 QK 溢出。

全12层独立、Head共享，共 **24个新增可学习标量**；另有不可学习的检查点配置元数据 buffer。没有新分类分支、损失、教师、STR、CHI、文本编码器或分割网络。数学上有效系数范围是 `(-0.2,0.2)`，允许负值；不将负值解释为逐图掩码纠错。

## 2. 方案复核与补齐的决策

1. **遵循完整公式，而非照抄 STMI 原地更新。** STMI 源码先对 logits 视图做前景更新、再做背景更新；在软覆盖率下直接照搬会引入顺序依赖。这里统一使用原始 `S` 构造正负增量，再一次性相加。
2. **逐图映射不改变采样记录。** 用独立 `MaskStore` 将具体图像相对路径映射到掩码，不修改原四元组，也不改 PID、Camera、Modality 或 PKM。启动时检查映射覆盖和文件存在，读样本时检查解码、原始尺寸、单通道及二值编码。另提供训练前全量审计工具。
3. **掩码必须与原始图像同分辨率。** 固定分割预处理若输出其他分辨率，须在离线准备阶段按既定规则还原到原图坐标。训练代码不替错误尺寸做静默补救。
4. **空掩码在联合几何变换后判定。** 裁剪可能把全部前景移出画面，此时同样 `v=0`；CLS权重仍为1，但整图增量关闭。空样本仍走可微计算，避免DDP新增 unused 参数。
5. **编码由整个 manifest 声明。** 支持二值 `0/1` 或 `0/255`，不对每张图片猜编码；不接受软概率、RGB掩码或有损 JPEG。合法空掩码允许，缺失/损坏掩码报错。全前景掩码仍有效。
6. **增强和基线保持一致。** 图像训练 Resize 为 bicubic，测试为原默认 bilinear；掩码始终 nearest。共享一次 Flip 决策和一次 Crop 坐标，维持原图像侧随机数调用顺序。RandomErasing只作用于归一化图像。
7. **不提供无效配置。** 全12层、零初始化、Head共享、无随机扩张及空掩码关闭策略固定在实现里；只暴露真正有用的开关、边界和消融项。默认边界仍为0.2。
8. **保留基线入口。** defaults 中默认关闭模块，`configs/UAD.yml` 开启，原 YAML 另存为 `UAD_baseline.yml`。禁用后无需掩码，批次字段数量和原调用格式恢复。
9. **训练权重严格加载。** ImageNet初始化沿用原 backbone loader；QG参数保持0。加载 ReID 检查点时要求全部24个参数及匹配的配置元数据，不能拿原UAD权重冒充已训练的QG模型。推理仍允许跳过 classifier 以适配评测身份数。
10. **不因卡数自动改学习率。** YAML仍为0.008。旧README显式0.032属于已有历史启动参数；若与该历史模型比较，双方应采用相同实际LR，不能将0.008与0.032混作仅模块差异。

这些修补未改变 guide 的核心公式或研究变量。论文中 STMI 的增益、参数共享消融来自其三分支 CLIP/配对检索设置，不能当作当前共享UAD必然提升的证据。

## 3. 原损失与梯度

\[
L=L_{CE}+L_{WRT}+0.01L_{ProCA}+L_{GPD}.
\]

| 损失 | 输入及原有机制 | 权重 |
|---|---|---|
| CE | `classifier(BN(f))`，普通交叉熵、无 label smoothing | 1 |
| WRT | 未做L2归一化的 pre-BN `f`；原加权正负距离、SoftMarginLoss | 1 |
| ProCA | 图像L2归一化 -> 模态中心 -> 身份中心；身份中心detach | 0.01 |
| GPD | L2图像特征对全身份原型库，温度0.03；冷启动逻辑保留 | 1 |

所有损失通过最终CLS回传至共享ViT及24个QG标量。BNNeck和classifier由CE直接更新。掩码和覆盖率不更新；GPD原型库无梯度，仍在step后用 `Norm(0.8*old+0.2*center)` 更新，保留原DDP汇总。

`loss/` 的所有源文件未改。Processor中的 `PCA`、`GPD`、`PrototypeBank` 计算定义及原型更新调用保留。没有调整现有AMP、GradScaler、全局梯度裁剪1.0、SGD、warmup、cosine或原型库设计。

## 4. 修改与新增文件

共修改9个已有文件、新增10个实现/配置/检查文件（含后续补齐的SAM2离线生成、多规则对照工具及检查）；另外新增说明、测试证据和review补丁。没有删除源文件。

| 类型 | 文件 | 功能与目的 |
|---|---|---|
| 修改 | `config/defaults.py` | 注册QG开关、边界、消融、诊断和掩码配置，默认兼容原UAD |
| 修改 | `configs/UAD.yml` | 开启QG、明确mask入口、使用独立输出目录，原训练超参数不变 |
| 修改 | `datasets/bases.py` | 逐图读取自身掩码；训练和测试追加覆盖率/有效性；检查tuple身份一致 |
| 修改 | `datasets/make_dataloader.py` | 建立MaskStore；同步增强；扩展collate且保留原字段与顺序；覆盖Query/Gallery/normal loader |
| 修改 | `model/backbones/vit_pytorch.py` | 每层两个标量；有界QG公式；FP32调制；12层传递掩码；零初始化和空掩码梯度保留 |
| 修改 | `model/make_model.py` | 三模态空间输入按图像同序拼接；测试传递；检查点24参数和配置校验；仍返回原三项训练输出 |
| 修改 | `processor/processor.py` | 训练、周期验证、独立推理传递空间输入；可选记录调制参数和实际幅度；损失和DDP分配逻辑保留 |
| 修改 | `solver/make_optimizer.py` | 所有QG标量一个独立组：base LR、WD=0；其余参数组规则不变 |
| 修改 | `README.md` | 指向本说明，区分新增QG配置和原baseline命令 |
| 新增 | `configs/UAD_baseline.yml` | 修改前UAD配置原样保留，用于无掩码基线 |
| 新增 | `datasets/masks.py` | 严格PNG/NPY加载、显式映射、联合几何增强、行优先覆盖率和有效性 |
| 新增 | `tools/audit_masks.py` | 从镜像目录建立manifest、或审计已有manifest；统计各split/模态/平台掩码分布 |
| 新增 | `tools/verify_reference.py` | 对保存的修改前源码检查完整模型零初始化等价和四项损失一致性 |
| 新增 | `tests/test_qg_sfm.py` | 无数据CPU检查：公式、梯度、增强、collate、空掩码、消融、优化器、检查点等 |
| 新增 | `tests/ddp_qg_smoke.py` | 两进程CPU/Gloo检查：PKM、梯度同步、全空掩码、原型库一致性、不同次数验证前向 |
| 新增 | `tools/generate_sam2_masks.py` | 在wsam环境离线逐图生成掩码、六场景训练集抽样预览、固定provenance及断点续跑 |
| 新增 | `tests/test_sam2_masks.py` | 离线工具辅助逻辑检查：提示坐标、候选选择、六场景抽样、尺寸与预览 |
| 新增 | `tools/compare_sam2_rules.py` | 固定训练集样本比较15类提示/输入规则与两种候选排序，保存所有候选和并排预览 |
| 新增 | `tests/test_sam2_rules.py` | 对照工具合成检查：坐标、补白、编码复用、分支输出及统计 |

`train.py`、`test.py`、`datasets/whu_mars.py`、两种采样器、scheduler、loss源码和 `utils/metrics.py` 均未修改。原单卡/DDP启动逻辑、原mINP扩展、同相机排除、九模态方向、AG和详细场景评测保留。

## 5. 掩码准备与审计

本地没有WHU图像、生成好的掩码或分割模型权重。本实现不虚构已生成掩码，也不将SAM加入训练。后续已根据用户服务器的wsam环境补齐离线工具，详见 `SAM2_离线掩码生成步骤.md`。正式训练前需用固定分割流程为 **train/query/test全部三模态的每一张图像** 生成掩码；不能把RGB掩码复制给IR/Thermal。

推荐镜像存储：

```text
/project/code/data/WHU-MARS/train/RGB/1000_c1_....jpg
/project/code/data/WHU-MARS_masks/train/RGB/1000_c1_....png
# 同理包含 train/IR、train/Thermal、query/*、test/*。
```

manifest格式如下；键相对于 `ROOT_DIR/DATASET_DIR`，值相对于 `MASK_ROOT`。路径使用 `/`，禁止越出根目录，也禁止多个图像映射到同一个掩码文件。路径检查可以发现映射错误，但不能代替检查分割语义是否真属于该行人。

```json
{
  "version": 1,
  "encoding": "0/255",
  "provenance": {
    "generator": "填写实际模型与版本",
    "checkpoint_sha256": "填写实际权重哈希",
    "prompt_rule": "填写固定的人体选择规则",
    "settings": {"填写实际推理参数": "实际值"},
    "binarization": "填写实际二值化规则及恢复到原图坐标的方法"
  },
  "images": {
    "train/RGB/1000_c1_example.jpg": "train/RGB/1000_c1_example.png",
    "train/IR/1000_c1_example.jpg": "train/IR/1000_c1_example.png"
  }
}
```

上例仅说明格式。请将实际provenance另存为JSON文件，不把占位文字用于正式实验。生成模型、权重、参数、二值化和目标选择规则在训练/测试间固定；不按测试指标挑选或改写特定图像掩码。

在 `WHU/` 下执行：

```bash
python tools/audit_masks.py \
  --image-root /project/code/data/WHU-MARS \
  --mask-root /project/code/data/WHU-MARS_masks \
  --manifest /project/code/data/WHU-MARS_masks/manifest.json \
  --encoding 0/255 --build --suffix .png \
  --provenance /project/code/data/WHU-MARS_masks/provenance.json \
  --report logs/mask_audit_1000.json
```

已有manifest去掉 `--build --suffix --provenance` 即可全量检查。非镜像目录可自行制作逐图映射，然后运行同一审计命令。NPY使用 `--suffix .npy`，若存储0/1编码，工具和配置同时改成 `0/1`。

工具检查所有文件能解码、原始尺寸匹配和严格二值；输出每个split下三模态×两平台中实际存在组的数量、空掩码比例、平均前景比例及10档patch覆盖率直方图。原camera1–5为ground、6–7为aerial。审计通过只表示工程格式合格；还应抽查六种场景的可视化分割质量，尤其是大量空掩码的模态。

## 6. 运行命令

以下命令在 `WHU/` 下运行，使用Linux服务器的原UAD环境。没有引入STMI环境依赖，也没有修改 `requirements.txt`。本次CPU检查并非在目标CUDA版本上完成。

单卡1000：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/UAD.yml \
  MODEL.DIST_TRAIN False \
  MODEL.PRETRAIN_PATH /project/code/data/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.MASK_ROOT /project/code/data/WHU-MARS_masks \
  DATASETS.MASK_MANIFEST /project/code/data/WHU-MARS_masks/manifest.json \
  SOLVER.IMS_PER_BATCH 64 SOLVER.BASE_LR 0.008 SOLVER.SEED 1234 \
  OUTPUT_DIR logs/UAD_QG_SFM_1000_single
```

四卡DDP1000（全局256 tuple，每卡64 tuple=192图像）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py --config_file configs/UAD.yml MODEL.DIST_TRAIN True \
  MODEL.PRETRAIN_PATH /project/code/data/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.MASK_ROOT /project/code/data/WHU-MARS_masks \
  DATASETS.MASK_MANIFEST /project/code/data/WHU-MARS_masks/manifest.json \
  SOLVER.IMS_PER_BATCH 256 SOLVER.BASE_LR 0.008 SOLVER.SEED 1234 \
  OUTPUT_DIR logs/UAD_QG_SFM_1000_DDP
```

基线使用同样启动条件，将配置改为 `configs/UAD_baseline.yml`，省略MASK参数并使用独立OUTPUT_DIR。已有历史baseline若实际用0.032，应让新的对照也明确使用0.032，或两者均重跑0.008。

独立推理（不必重复加载ImageNet初始化，检查点会完整恢复骨干）：

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/UAD.yml \
  MODEL.DIST_TRAIN False MODEL.PRETRAIN_CHOICE none \
  DATASETS.MASK_ROOT /project/code/data/WHU-MARS_masks \
  DATASETS.MASK_MANIFEST /project/code/data/WHU-MARS_masks/manifest.json \
  TEST.WEIGHT logs/UAD_QG_SFM_1000_DDP/transformer_120.pth \
  TEST.DETAILED_DOMAIN_EVAL True \
  OUTPUT_DIR logs/UAD_QG_SFM_1000_eval
```

1000GD评测追加 `DATASETS.FILTER_GD True`，使用同一epoch120权重。2337独立训练时将 `DATASETS.DATASET_DIR` 改为 `WHU-MARS-2337`，同时使用2337自己的MASK_ROOT/manifest和独立OUTPUT_DIR。不要将1000GD当成新的训练任务。

开启模块且缺少mask配置会直接报错；原UAD检查点不能用QG开启配置加载。加载完的QG模型也不能在测试时省略掩码。

## 7. 消融与诊断

| 实验 | 相对主配置的覆盖项 |
|---|---|
| 原UAD | `MODEL.QG_SFM.ENABLED False`，或用baseline YAML |
| 连续软SFM | `MODEL.QG_SFM.MODE soft_sfm` |
| 均匀覆盖率 | `MODEL.QG_SFM.UNIFORM_COVERAGE True` |

软SFM的背景项为 `-beta*(1-p_i*p_j)*(S-Smin)`，其他参数化、空掩码策略及训练条件相同；不是STMI原始二值实现。均匀覆盖率用每张图自己的 `rho.mean()` 替换全部patch权重，CLS仍为1。

每个消融须独立训练。检查点保存mode/uniform/bound/尺寸/stride/掩码编码及provenance的元数据，加载时校验一致，防止错误配置下悄悄换公式。绝对数据路径和全量映射不嵌入权重，便于迁移及同一生成流程下更换评测子集。

默认每个log周期，rank0在该次前向记录各层 `a,b,alpha,beta` 及

\[
R=\frac{\operatorname{RMS}(\Delta S-\operatorname{mean}_j\Delta S)}
{\operatorname{RMS}(S-\operatorname{mean}_jS)+10^{-12}}.
\]

这是rank0当前本地batch的前向诊断，参数与R来自同一次前向，并非跨rank/全epoch平均。训练原有四项损失日志保留。设置 `MODEL.QG_SFM.DIAGNOSTICS False` 可关闭额外统计。

新增参数虽仅24个，FP32调制需要额外的注意力中间张量，训练显存/耗时会增加。本机无GPU，不能给出服务器192图像/卡的峰值显存或吞吐承诺；正式120轮前应先做小步真实CUDA训练和原批量显存检查。

## 8. 本地自检证据与限制

可复跑：

```bash
python -m compileall -q .
python -m unittest discover -s tests -p test_qg_sfm.py -v
python tools/verify_reference.py --original-root ../tmp/original/WHU
python tests/ddp_qg_smoke.py
```

`verify_reference` 需要保留的修改前源码树，本工作区为 `../tmp/original/WHU`；它不是训练依赖。单元检查与DDP脚本不依赖该快照。

已通过的检查：

- 逐元素独立参考公式与极值梯度对照；零初始化logits/概率/attention输出一致。
- 完整ViT-B与实际修改前UAD源码对照：12层attention输出、block输出、CLS、BNNeck、classifier在FP32下完全一致。
- 相同特征和原型状态下，原CE/WRT/ProCA/GPD及原型更新完全一致。
- 12层24个标量及骨干正常反传；零参数仍有梯度；空掩码有梯度张量但调制梯度为0。
- 图像与掩码联合几何变换；固定seed下图像增强与原pipeline完全一致；全零和裁剪后变空的mask正确关闭调制。
- 软SFM、均匀覆盖率、低精度logits转FP32调制；三模态collate与训练/全部评测loader；baseline旧批次格式。
- 严格缺失/损坏/错误尺寸/三通道/错误编码检查；PNG/NPY；离线manifest创建与审计工具合成数据端到端。
- 独立优化器组、scheduler基础LR一致；全部24个标量恢复和加载后特征一致；缺参数或元数据不符时拒绝加载。
- 两进程CPU/Gloo：PKM身份分组、24参数梯度同步、一rank空掩码/全rank空掩码、原型库一致、不同次数的unwrapped验证前向。

单元和完整模型对照环境为本机 `torch 2.8.0+cpu / torchvision 0.23.0+cpu / timm 1.0.15`。该环境的Gloo初始化失败：`makeDeviceForHostname(): unsupported gloo device`。改用本机已有 `DL` 环境的 `torch 2.5.1+cu121`，在CPU上用 `USE_LIBUV=0` 后，两进程Gloo测试通过。没有修改训练入口来适应本机Windows。PyTorch的[Gloo设备工厂源码](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/distributed/c10d/GlooDeviceFactory.cpp)用于核对该环境错误的初始化位置。

**未验证：** 真正CUDA FP16 AMP、NCCL多GPU、真实WHU分割质量、真实批量峰值显存、ImageNet文件实际加载、120轮训练、各协议最终检索指标。本次没有真实训练日志、权重和数据可供读取，因此不声称提升、不推测指标。正确性检查通过不等于研究效果已经成立。

本目录 `verification/` 保存检查输出，`source_review.json` 记录源文件变更和未改文件的哈希，`QG_SFM_code_changes.patch` 提供逐行代码差异。
