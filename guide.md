# WHU-MARS UAD-QG-SFM 方案设计与代码修改指导文档（复核修订版）

**文档用途：** 指导在WHU-MARS官方UAD代码基础上接入QG-SFM，明确网络结构、数据处理、正向特征流、反向梯度流、损失函数、参数优化、训练初始化、推理评测及实现检查。

**设计原则：**

- 以WHU-MARS官方UAD代码为唯一主框架。
- 以STMI的SFM作为注意力调制机制的参考来源。
- 只迁移和改造SFM，不迁移STMI的STR、CHI、文本生成及其他多分支设计。
- 原UAD的Backbone、BNNeck、分类器、四项损失及检索特征定义全部保留。
- 新增的24个调制参数是唯一新增的可学习参数。
- 不通过新增损失、可靠性网络、教师模型或额外训练阶段复杂化第一版方案。

---

# 第一章 方案全貌与最终设计边界

## 1.1 研究目标

在WHU-MARS多模态、多平台行人重识别任务中，研究人体空间先验能否通过调整Transformer内部的Token交互关系，改善UAD最终统一身份特征的判别能力。

研究对象不是独立的前景分割网络，也不是多模态特征融合网络。

我们需要验证的是：

> 在保留UAD单分支统一身份表征及原有身份监督体系的情况下，将人体空间信息引入Self-Attention，能否改善最终768维CLS身份特征的检索质量。

需要区分：

1. 掩码提供空间先验；
2. QG-SFM决定空间先验如何参与Attention；
3. UAD四项损失决定这种Attention变化是否有利于身份表征；
4. 最终异构Gallery检索结果判断整个设计是否有效。

不能把注意力变化、参数更新或训练损失下降直接等同于检索性能提升。

## 1.2 最终网络结构

采用原始UAD的ViT-B/16作为统一特征提取器。

原始结构：

```text
RGB / NIR / TIR Images
          |
          v
      Shared ViT
          |
          v
      Final CLS
          |
          v
        BNNeck
          |
          v
      ID Classifier

Pre-BN CLS
   |     |      |
  WRT  ProCA   GPD
```

修改后：

```text
RGB / NIR / TIR Images
          |
          v
    Image-Mask Pairing
          |
          v
    Joint Geometric Augmentation
          |
          v
    Patch Foreground Coverage
          |
          v
    Shared ViT-B/16
          |
    +-----+------+
    |            |
    | 12 Transformer Blocks
    |            |
    |  QK Attention Logits
    |            |
    |     QG-SFM
    |            |
    |   Softmax
    |            |
    |  Attention × V
    |            |
    |  Residual + MLP
    |            |
    +------------+
          |
          v
    Final LayerNorm
          |
          v
    Pre-BN CLS (768)
          |
     +----+----+
     |         |
    BNNeck    Pre-BN
     |         |
     CE        WRT
               |
          +----+----+
          |         |
         ProCA     GPD
```

QG-SFM不增加独立的特征提取分支。

不改变：

- ViT的层数；
- Token数量；
- Embedding Dimension；
- QKV投影；
- Value计算；
- Residual；
- MLP；
- CLS提取方式；
- BNNeck；
- 分类器；
- UAD四项损失。

## 1.3 最终参数与结构

| 项目 | 最终设计 |
|---|---|
| Backbone | ViT-B/16 |
| 输入尺寸 | 256×128 |
| Patch尺寸 | 16×16 |
| Patch数量 | 128 |
| CLS数量 | 1 |
| Token总数 | 129 |
| Embedding Dimension | 768 |
| Transformer层数 | 12 |
| Attention Heads | 12 |
| Head Dimension | 64 |
| QG-SFM作用位置 | 全12层 |
| 每层参数 | α、β各1个 |
| Head间参数 | 共享 |
| 参数总量 | 24个 |
| 参数初始化 | 原始参数全部为0 |
| 参数范围 | (-0.2, 0.2) |
| 参数化 | 0.2×tanh(raw/0.2) |
| Query策略 | 前景覆盖率门控 |
| Key策略 | 软前景/背景调制 |
| 空掩码 | 关闭该图调制 |
| 掩码扩张 | 不使用 |
| 额外损失 | 不增加 |
| 最终特征 | Pre-BN CLS，768维 |

## 1.4 最终采用的注意力调制公式

定义第l层、第b张图像、第h个Head、第i个Query、第j个Key对应的原始Attention Logits：

\[
S_{l,b,h,i,j}
=
Q_{l,b,h,i}K_{l,b,h,j}^{T}
\]

对于每一行：

\[
S_{l,b,h,i}^{max}
=
\max_j S_{l,b,h,i,j}
\]

\[
S_{l,b,h,i}^{min}
=
\min_j S_{l,b,h,i,j}
\]

Patch覆盖率：

\[
p_b=[1,\rho_{b,1},\ldots,\rho_{b,128}]
\]

有效性标记：

\[
v_b\in\{0,1\}
\]

最终调制：

\[
\boxed{
\begin{aligned}
\Delta S_{l,b,h,i,j}
={}&v_bp_{b,i}\Big[
\alpha_lp_{b,j}
(S_{l,b,h,i}^{max}-S_{l,b,h,i,j})\\
&-\beta_l(1-p_{b,j})
(S_{l,b,h,i,j}-S_{l,b,h,i}^{min})
\Big]
\end{aligned}
}
\]

得到：

\[
\widetilde S_l=S_l+\Delta S_l
\]

然后：

\[
\boxed{
A_l=
Softmax\left(
\frac{\widetilde S_l}{\sqrt{64}}
\right)
}
\]

\[
O_l=A_lV_l
\]

该公式是本方案的最终实现依据。

## 1.5 对前一版方案的必要修正

以下内容必须在代码实现和论文描述中保持一致。

**修正一：掩码必须与图像同步执行几何增强。**

不照搬STMI官方代码中掩码独立读取、未同步变换的处理方式。

**修正二：每张图像独立使用掩码。**

不将同身份RGB掩码复制给NIR、TIR，也不假设三模态图像存在像素级空间对应关系。

**修正三：QG-SFM必须采用非原地Logits更新。**

使用新的调制张量和Logits张量，不直接对QK矩阵乘法结果的视图执行原地加减。

**修正四：行最大值、最小值不能detach。**

保留其计算图，使调制参数和QK特征能够通过完整公式反向传播。

**修正五：官方UAD学习率不进行自动线性放大。**

官方配置及论文使用的基础学习率为0.008。若当前实际基线采用其他启动参数，则必须保持基线与改进模型一致，不能单独改变QG-SFM的训练条件。

**修正六：保留当前评测器。**

特别是当前实验已经使用的mINP扩展，不应因迁移官方评测代码而丢失。

---

# 第二章 论文原始机制、官方代码和本方案自适应设计的对应关系

## 2.1 STMI的原始设计

参考：

Xu et al., STMI: Segmentation-Guided Token Modulation with Cross-Modal Hypergraph Interaction for Multi-Modal Object Re-Identification，AAAI 2026。

原论文第3.1节，公式（1）—（8）。

原始流程：

1. 使用SAM生成像素级二值掩码；
2. 将图像划分成Patch；
3. 生成CLS加Patch的二值Token掩码；
4. 计算Attention Logits；
5. 计算行最大值和最小值；
6. 构造前景与背景调制矩阵；
7. 使用可学习α、β调整Attention；
8. 执行Softmax。

原始Token掩码：

\[
m=[1,m_1,\ldots,m_N]
\]

原始前景关系：

\[
R_{ij}=m_im_j
\]

原始正负调制矩阵：

\[
M_{ij}^{pos}=S_i^{max}-S_{ij}
\]

\[
M_{ij}^{neg}=S_{ij}-S_i^{min}
\]

原始调制：

\[
S^{STMI}
=
\alpha R\odot M^{pos}
-
\beta(1-R)\odot M^{neg}
\]

这里的α、β是STMI中每个Transformer层使用的可学习标量。

STMI官方代码的`ResidualAttentionBlock`中，每个Block都有独立的`adapter_alpha`和`adapter_beta`，不同Head共享该层参数。

官方代码初值为：

\[
\alpha,\beta\approx0.1+\mathcal N(0,0.05^2)
\]

注意，这只是代码初始化，不是本方案的初始化方式。

## 2.2 STMI官方代码必须重点审查的地方

相关文件：

- `modeling/clip/model.py`
- `data/datasets/bases.py`
- `data/datasets/make_dataloader.py`
- `modeling/make_model.py`
- `configs/RGBNT201/STMI.yml`
- `config/defaults.py`

### （1）掩码读取

STMI使用：

`read_ndy(ndy_path, threshold=128)`

将像素掩码转换成16×8的二值Token掩码。

其计算方式是：

\[
m_j=
\begin{cases}
1,&\sum_{(u,v)\in P_j}M(u,v)\geq128\\
0,&\text{otherwise}
\end{cases}
\]

但这里存在必须注意的问题：

如果原始掩码是0/1编码，128代表至少128个前景像素。

如果原始掩码是0/255编码，128可能只意味着Patch中出现一个前景像素。

因此，不能脱离原始掩码编码解释128阈值。

本方案不使用这一阈值，而是直接计算归一化像素掩码的平均覆盖率。

### （2）图像与掩码增强不同步

STMI的`ImageDataset.__getitem__()`先读取图像和掩码，再分别对RGB、NIR、TIR图像调用图像Transform。

掩码在读取后直接追加到返回结果中，没有同步执行Flip、Padding和Crop。

因此，本方案必须重新实现图像与掩码联合几何增强。

### （3）一个掩码传入三个模态分支

STMI的RGBNT201数据以三模态配对样本为基础，数据集中每个Triplet对应一个SAM2掩码。

在`modeling/make_model.py`中，同一个mask分别传入RGB、NI、TI三个Backbone调用。

WHU-MARS-2337包含大量不要求同步的跨模态图像，不能采用同样的数据关联方式。

本方案改为：

\[
I_{RGB}\leftrightarrow M_{RGB}
\]

\[
I_{NIR}\leftrightarrow M_{NIR}
\]

\[
I_{TIR}\leftrightarrow M_{TIR}
\]

每张图像独立关联自己的掩码。

### （4）Attention计算

STMI代码采用Sequence-first：

\[
[L,B,D]
\]

经过QKV计算后，将Head展开为：

\[
[B\times H,L,d_h]
\]

而UAD使用Batch-first：

\[
[B,L,D]
\]

因此不能直接复制STMI的QKV维度重排代码。

只迁移调制数学关系，在UAD原始Attention布局中实现。

### （5）原地Logits修改

STMI代码：

```python
patch_attn = attn_logits[:, :, :]
```

这里的`patch_attn`是原始Logits的视图。

后续通过`+=`和`-=`直接修改。

本方案不采用这种原地写法，而是单独构造：

\[
\Delta S
\]

然后：

\[
\widetilde S=S+\Delta S
\]

这样可以避免对原始QK计算结果进行原地修改，也更方便验证零初始化等价性。

### （6）随机掩码扩张

STMI论文允许训练时以概率p将部分背景Token转成前景。

官方RGBNT201配置：

\[
p=0.01
\]

推理时设置为0。

本方案首轮关闭随机掩码扩张。

原因是本方案已经将二值掩码改为连续覆盖率，同时又引入Query门控。第一轮不再增加额外的掩码扰动变量。

## 2.3 UAD原始设计

参考：

Zhao et al., WHU-MARS: A Multispectral Aerial-Ground Benchmark Towards Any-Scenario Person Re-Identification，CVPR 2026。

论文第4.2—4.5节。

UAD的主要结构：

- Shared Backbone；
- BNNeck和共享Classifier；
- Weighted Regularization Triplet；
- Progressive Center Alignment；
- Global Prototype Discrimination。

原始UAD的输入不要求三模态图像帧同步。

它使用统一共享Backbone学习身份空间，而不是分别建立RGB、NIR、TIR三个独立特征网络。

本方案必须完整保留这一结构。

## 2.4 最终技术来源划分

| 设计内容 | 来源 | 本方案处理 |
|---|---|---|
| 行最大值和最小值调制 | STMI公式4、5 | 保留 |
| 前景与背景两类调制 | STMI公式7 | 保留思想 |
| 每层独立调制参数 | STMI代码 | 保留 |
| Head共享参数 | STMI代码 | 保留 |
| 二值Token掩码 | STMI公式2 | 改为软覆盖率 |
| 前景关系矩阵R | STMI公式6 | 改为Query门控结构 |
| 随机背景扩张 | STMI公式7及代码 | 关闭 |
| α、β随机正值初始化 | STMI代码 | 改为零初始化 |
| α、β有界参数化 | 本方案 | 新增 |
| Query前景门控 | 本方案 | 新增 |
| 空掩码有效性标记 | 本方案 | 新增 |
| 图像与掩码联合增强 | 本方案 | 新增 |
| 每图像独立掩码 | WHU数据特点下的适配 | 新增 |
| 共享ViT | UAD | 保留 |
| CE/WRT/ProCA/GPD | UAD | 全部保留 |
| 最终CLS检索 | UAD | 保留 |

---

# 第三章 数据集与掩码数据流修改

## 3.1 数据集组织

UAD官方：

`datasets/whu_mars.py`

数据按照模态组织：

```text
train/
    RGB/
    IR/
    Thermal/

query/
    RGB/
    IR/
    Thermal/

test/
    RGB/
    IR/
    Thermal/
```

概念上分别对应：

- RGB；
- NIR；
- TIR。

官方数据集记录形式为：

\[
(img\_path,pid,camid,modality\_id)
\]

其中第四个字段是模态编号。

不要将这个字段误当成Aerial/Ground平台编号。

当前UAD配置没有启用SIE Camera和SIE View，模型不根据相机或平台标签进行特征条件化。

## 3.2 增加掩码路径关联

每条图像数据增加一个对应的掩码路径。

逻辑上：

\[
(img\_path,pid,camid,modality\_id,mask\_path)
\]

掩码路径必须通过具体图像文件关联，而不是通过PID关联。

推荐建立明确的映射关系：

\[
ImagePath\rightarrow MaskPath
\]

不建议使用容易产生路径歧义的字符串替换作为唯一映射依据。

数据加载时必须验证：

- 图像存在；
- 掩码存在；
- 掩码可正常读取；
- 掩码尺寸正确；
- 掩码值域正确；
- 图像与掩码属于同一原始图像。

如果掩码文件缺失或损坏，直接报错。

不能静默返回全零掩码。

## 3.3 掩码生成与训练边界

掩码生成模型属于外部固定先验生成过程。

它不参与UAD训练。

不将SAM或其他分割模型放入QG-SFM反向传播图。

正式训练前固定：

- 分割模型版本；
- 掩码生成参数；
- 二值化规则；
- 掩码文件映射；
- 掩码存储方式。

训练、验证、测试均使用相同的掩码生成流程。

不要根据测试结果选择性修改某些测试图像的掩码。

## 3.4 像素掩码处理

定义：

\[
M_b\in\{0,1\}^{256\times128}
\]

其中：

- 1表示人体前景；
- 0表示背景。

如果原始掩码使用0/255编码，需要先按照实际文件编码归一化。

\[
M_b=\frac{M_b^{raw}}{255}
\]

仅当原始文件确实采用0/255编码时才使用上述转换。

如果原始文件是0/1，则直接使用。

不能把不同编码的掩码混在一起处理。

## 3.5 图像和掩码联合增强

这是数据修改中最重要的部分之一。

UAD原始训练增强：

1. Resize；
2. RandomHorizontalFlip；
3. Pad；
4. RandomCrop；
5. ToTensor；
6. Normalize；
7. RandomErasing。

需要构造一个联合变换过程。

| 操作 | 图像 | 掩码 |
|---|---|---|
| Resize | 原始UAD插值设置 | Nearest |
| HorizontalFlip | 执行 | 同步执行 |
| Pad | 原始UAD设置 | 同步填充0 |
| RandomCrop | 原始UAD裁剪坐标 | 使用相同坐标 |
| ToTensor | 执行 | 转为0/1浮点 |
| Normalize | 执行 | 不执行 |
| RandomErasing | 执行 | 不执行 |

特别注意：

### RandomHorizontalFlip

图像翻转时，掩码必须使用同一个翻转决策。

### Padding

图像和掩码使用相同的Padding宽度。

掩码新增区域全部填充0。

### RandomCrop

图像和掩码必须使用相同的裁剪坐标。

不能分别调用两个独立的RandomCrop。

### RandomErasing

继续保留UAD原始RandomErasing。

RandomErasing发生在图像Tensor及Normalize之后。

掩码保持原始人体语义定义，不随擦除区域变成背景。

这意味着被擦除的人体区域仍然属于人体空间先验。

第一版不引入额外可见性网络，也不改变原始RandomErasing策略。

## 3.6 每个模态独立增强

UAD的PKM采样不要求不同模态图像像素级同步。

因此：

同一个身份的RGB、NIR、TIR图像可以拥有不同的随机翻转、裁剪参数。

但是每一张图像必须与自己的掩码同步增强。

即：

\[
(I_{RGB},M_{RGB})\rightarrow Aug_{RGB}
\]

\[
(I_{NIR},M_{NIR})\rightarrow Aug_{NIR}
\]

\[
(I_{TIR},M_{TIR})\rightarrow Aug_{TIR}
\]

不需要让三个模态共用同一套随机几何参数。

## 3.7 Patch级软覆盖率

在图像和掩码完成联合几何变换后，计算Patch覆盖率。

图像输入：

\[
H=256,\quad W=128
\]

Patch：

\[
16\times16
\]

Patch网格：

\[
16\times8
\]

Patch总数：

\[
N=128
\]

对于第j个Patch：

\[
\boxed{
\rho_{b,j}
=
\frac{1}{256}
\sum_{(u,v)\in P_j}M_b(u,v)
}
\]

因此：

\[
\rho_{b,j}\in[0,1]
\]

推荐使用：

\[
AvgPool2d(kernel=16,stride=16)
\]

生成：

\[
\rho_b\in\mathbb R^{16\times8}
\]

再按照原始Patch Embedding的行优先顺序展平：

\[
\rho_b\in\mathbb R^{128}
\]

注意Patch覆盖率必须在联合几何增强之后计算。

不能先对原始掩码池化，再对已经池化的16×8矩阵独立进行像素级随机Crop。

## 3.8 CLS权重

CLS没有实际图像区域。

固定：

\[
p_0=1
\]

完整Token权重：

\[
\boxed{
p_b=[1,\rho_{b,1},\ldots,\rho_{b,128}]
}
\]

维度：

\[
p_b\in\mathbb R^{129}
\]

## 3.9 空掩码有效性

定义：

\[
v_b=
\begin{cases}
1,&\sum M_b>0\\
0,&\sum M_b=0
\end{cases}
\]

当掩码有效时：

\[
v_b=1
\]

当文件存在、格式正确，但分割结果为空时：

\[
v_b=0
\]

此时：

\[
\Delta S_b=0
\]

注意CLS仍然保持前景权重1，但有效性标记会关闭整张图像的调制。

这样可以避免出现：

\[
p=[1,0,\ldots,0]
\]

却仍然对CLS行进行前景增强和背景抑制的情况。

不允许以下处理：

- 缺失文件转换为全零；
- 读取失败转换为全零；
- 尺寸不匹配时自动忽略；
- 掩码异常时静默关闭模块。

近乎全前景的有效掩码不自动判为无效。

## 3.10 数据加载输出

建议数据集经过处理后，为每张图像提供：

\[
(X_b,\rho_b,v_b,pid,camid,modality\_id)
\]

其中：

- \(X_b\)：增强后的图像；
- \(\rho_b\)：128维Patch覆盖率；
- \(v_b\)：有效性标记；
- PID、相机和模态标签保持UAD原有定义。

为了减少GPU传输开销，可以在DataLoader的CPU侧完成AvgPool2d，直接向模型传递128维覆盖率和有效性标记。

也可以传递完整掩码、在模型侧池化。

两种实现数学上等价，但必须保证池化发生在联合几何增强之后。

第一种更适合当前固定16×8的Patch布局。

---

# 第四章 QG-SFM模块的具体实现指导

## 4.1 修改的核心位置

主要修改：

`CVPR26_UAD/model/backbones/vit_pytorch.py`

重点涉及：

- Attention；
- Block；
- TransReID；
- forward_features。

不另行建立独立Transformer。

在原Attention内部增加QG-SFM计算即可。

## 4.2 原始UAD Attention

UAD原始Attention：

\[
QKV=Linear(X)
\]

经过重排：

\[
Q,K,V\in
\mathbb R^{B\times12\times129\times64}
\]

计算：

\[
S=QK^T
\]

然后：

\[
A=Softmax(S/\sqrt{64})
\]

最后：

\[
O=AV
\]

原始Attention还包括：

- Attention Dropout；
- 输出Projection；
- Projection Dropout。

这些操作全部保留。

## 4.3 QG-SFM插入顺序

必须严格使用：

```text
QKV Projection
      |
      v
Split Q / K / V
      |
      v
QK^T
      |
      v
Raw Attention Logits
      |
      v
Row-wise Max / Min
      |
      v
QG-SFM Modulation
      |
      v
Logits + Delta
      |
      v
Attention Scale
      |
      v
Softmax
      |
      v
Attention Dropout
      |
      v
Attention × V
      |
      v
Output Projection
```

不能把调制放到：

- Softmax之后；
- Attention概率之后；
- Value聚合之后；
- CLS输出之后。

否则就不是当前确定的QG-SFM。

## 4.4 调制矩阵的具体计算

对每个样本、每个Head、每个Query行：

\[
S_i^{max}=\max_jS_{ij}
\]

\[
S_i^{min}=\min_jS_{ij}
\]

最大值和最小值沿最后一个Key维度计算。

其维度为：

\[
S_i^{max},S_i^{min}
\in\mathbb R^{B\times12\times129\times1}
\]

Patch权重拆分：

\[
p_i\in\mathbb R^{B\times129\times1}
\]

\[
p_j\in\mathbb R^{B\times1\times129}
\]

实际实现需要增加Head维度进行广播。

有效性标记：

\[
v\in\mathbb R^{B\times1\times1\times1}
\]

计算：

\[
\Delta S^{pos}
=
v\alpha_lp_ip_j(S_i^{max}-S)
\]

\[
\Delta S^{neg}
=
v\beta_lp_i(1-p_j)(S-S_i^{min})
\]

\[
\boxed{
\Delta S=\Delta S^{pos}-\Delta S^{neg}
}
\]

再得到：

\[
\widetilde S=S+\Delta S
\]

其中所有矩阵均针对当前图像独立计算。

不同图像之间不共享最大值、最小值或空间权重。

## 4.5 QG-SFM参数组织

在每个Attention层内注册两个原始参数：

\[
a_l,b_l
\]

一共12层：

\[
\{a_l,b_l\}_{l=1}^{12}
\]

共24个标量。

有效参数：

\[
\boxed{
\alpha_l=0.2\tanh(a_l/0.2)
}
\]

\[
\boxed{
\beta_l=0.2\tanh(b_l/0.2)
}
\]

初始化：

\[
a_l=b_l=0
\]

所以：

\[
\alpha_l=\beta_l=0
\]

每层的α、β被本层全部12个Head共享。

不能在每个Head内重复注册独立参数。

也不能将12层参数合并成全网络共享的两个参数。

## 4.6 为什么使用零初始化

零初始化意味着：

\[
\Delta S=0
\]

从而：

\[
\widetilde S=S
\]

模型的初始前向计算与原UAD完全一致。

随着训练进行，24个参数通过身份损失获得梯度，逐步学习是否使用空间先验。

参数导数：

\[
\frac{\partial\alpha_l}{\partial a_l}
=
1-\tanh^2(a_l/0.2)
\]

零点：

\[
\left.
\frac{\partial\alpha_l}{\partial a_l}
\right|_{a_l=0}=1
\]

β同理。

因此不会在初始化时人为缩小有效学习尺度。

## 4.7 正负系数的实际作用

α、β都允许正负变化。

当α为正：

前景Key Logits向行最大值靠近。

当α为负：

前景Key Logits远离行最大值，前景组内部差异可能扩大。

当β为正：

背景Key Logits向行最小值靠近。

当β为负：

背景Key Logits远离行最小值，背景组内部差异可能扩大。

因此，不能在代码中将α、β限制为非负数。

也不能将负值直接解释为掩码纠错。

它们表达的是全数据集共享的可学习注意力偏好。

## 4.8 FP32计算与AMP处理

UAD训练使用AMP。

QK矩阵乘法仍沿用原始UAD的AMP计算方式。

得到原始Logits后：

1. 将Logits转换为FP32；
2. 计算行最大值和最小值；
3. FP32计算调制增量；
4. FP32完成Logits相加；
5. 转回原始Logits数据类型；
6. 沿用原始Attention Scale和Softmax。

示意：

```python
logits_raw = q @ k.transpose(-2, -1)

logits_fp32 = logits_raw.float()

row_max = logits_fp32.amax(dim=-1, keepdim=True)
row_min = logits_fp32.amin(dim=-1, keepdim=True)

delta = qg_modulation(
    logits_fp32,
    row_max,
    row_min,
    patch_weights,
    valid_flags
)

logits_mod = (logits_fp32 + delta).to(logits_raw.dtype)

attn = (logits_mod * self.scale).softmax(dim=-1)
```

这里仅为计算流程示意。

不要在原始QK矩阵乘法已经发生低精度溢出后，认为转换FP32能够修复Inf或NaN。

同时，不要使用：

```python
if alpha == 0 and beta == 0:
    return original_attention
```

这样的条件分支。

零初始化必须保留完整可微计算路径，否则可能导致调制参数初始梯度被切断，也可能增加DDP的Unused Parameters处理问题。

## 4.9 为什么不使用原地修改

推荐：

\[
S_{mod}=S+\Delta S
\]

而不是：

\[
S+=\Delta S
\]

原因：

- 原始Logits仍可用于诊断；
- 避免视图原地操作；
- 避免Autograd版本计数问题；
- 更方便检查零初始化等价性；
- 方便分别记录实际调制量。

## 4.10 QG-SFM与原Transformer Block的衔接

原UAD：

\[
X_l'=X_l+Attention(LN(X_l))
\]

\[
X_{l+1}=X_l'+MLP(LN(X_l'))
\]

修改后：

\[
X_l'=X_l+Attention_{QG}(LN(X_l),p,v)
\]

\[
X_{l+1}=X_l'+MLP(LN(X_l'))
\]

只有Attention内部发生变化。

不能把掩码直接乘到Token特征上。

不能将背景Token设置为0。

不能删除背景Token。

不能改变CLS Token。

---

# 第五章 Backbone与最终特征的完整数据流

## 5.1 输入Batch

设每个模态本地Batch包含B个tuple样本。

图像：

\[
X_R,X_N,X_T
\in\mathbb R^{B\times3\times256\times128}
\]

覆盖率：

\[
\rho_R,\rho_N,\rho_T
\in\mathbb R^{B\times128}
\]

有效性：

\[
v_R,v_N,v_T
\in\mathbb R^B
\]

按照UAD原有模态顺序拼接：

\[
X=[X_R;X_N;X_T]
\]

\[
\rho=[\rho_R;\rho_N;\rho_T]
\]

\[
v=[v_R;v_N;v_T]
\]

因此：

\[
X\in\mathbb R^{3B\times3\times256\times128}
\]

\[
\rho\in\mathbb R^{3B\times128}
\]

\[
v\in\mathbb R^{3B}
\]

必须保证三者拼接顺序完全一致。

## 5.2 Patch Embedding

原始Patch Embedding：

\[
E(X)\in\mathbb R^{3B\times128\times768}
\]

CLS扩展：

\[
CLS\in\mathbb R^{3B\times1\times768}
\]

拼接：

\[
T_0=[CLS;E(X)]
\]

得到：

\[
T_0\in\mathbb R^{3B\times129\times768}
\]

加入原始位置编码：

\[
T_0=T_0+PE
\]

再执行原始Position Dropout。

注意：

Patch覆盖率不需要加入Token Embedding，也不需要与768维特征拼接。

它只作为Attention调制的外部空间权重。

## 5.3 逐层Transformer计算

对每一层：

\[
T_l\rightarrow LN_1\rightarrow QKV
\]

\[
QK^T\rightarrow QG\text{-}SFM
\]

\[
Softmax\rightarrow AV
\]

\[
Residual\rightarrow MLP\rightarrow Residual
\]

第l层使用：

\[
(\alpha_l,\beta_l)
\]

全部12层完成后：

\[
T_{12}\in\mathbb R^{3B\times129\times768}
\]

## 5.4 Final LayerNorm与CLS

经过原始Final LayerNorm：

\[
\hat T=LN(T_{12})
\]

提取第0个Token：

\[
\boxed{
f=\hat T[:,0,:]
}
\]

因此：

\[
f\in\mathbb R^{3B\times768}
\]

这就是UAD最终的Pre-BN全局身份特征。

## 5.5 BNNeck与Classifier

\[
z=BN(f)
\]

其中：

\[
z\in\mathbb R^{3B\times768}
\]

分类：

\[
s=W_{cls}z
\]

其中：

\[
s\in\mathbb R^{3B\times C_{train}}
\]

这里：

- \(f\)是pre-BN特征；
- \(z\)是BNNeck特征；
- \(s\)是身份分类Logits。

不允许把三者混为一谈。

## 5.6 不进行额外特征融合

本方案不使用：

\[
[f_{RGB};f_{NIR};f_{TIR}]
\]

作为最终检索特征。

三个模态的图像仍然分别通过同一套共享Backbone。

三模态Batch拼接只是为了批量计算。

ProCA、GPD通过身份标签和中心构造进行跨模态监督，不是通过特征拼接实现融合。

最终每张图像仍然只有：

\[
768\text{维}
\]

身份描述子。

---

# 第六章 UAD四项损失的代码实现与特征来源

本章是实现时最不能随意修改的部分。

UAD论文给出了CE、WRT、ProCA、GPD的总体关系，但WRT的具体权重计算和若干数值细节需要以官方代码为准。

## 6.1 CE：身份分类

### 输入特征

\[
z=BN(f)
\]

分类：

\[
s=W_{cls}z
\]

### 损失

当前配置：

`IF_LABELSMOOTH: off`

因此使用普通交叉熵：

\[
\boxed{
L_{CE}
=
-\frac1{3B}
\sum_i
\log
\frac{\exp(s_{i,y_i})}
{\sum_c\exp(s_{i,c})}
}
\]

分类目标：

\[
Y_{rep}=Y.repeat(3)
\]

与RGB、NIR、TIR拼接后的特征顺序对应。

参数：

\[
\lambda_{CE}=1
\]

### 反向传播

\[
L_{CE}
\rightarrow Classifier
\rightarrow BNNeck
\rightarrow CLS
\rightarrow ViT
\rightarrow QG\text{-}SFM
\]

CE能够通过最终CLS特征间接优化24个QG-SFM参数。

CE不直接优化掩码。

## 6.2 WRT：加权正负样本排序

官方文件：

`loss/wrt_loss.py`

### 输入

使用pre-BN特征：

\[
f\in\mathbb R^{3B\times768}
\]

**不执行L2归一化。**

官方调用：

```python
triplet(feat, target)[0]
```

`normalize_feature`默认False。

### 欧氏距离

\[
d_{ij}
=
\sqrt{
\max(
\|f_i\|_2^2+
\|f_j\|_2^2-
2f_i^Tf_j,
10^{-12})
}
\]

距离矩阵使用FP32计算。

### 正负样本掩码

\[
I_{ij}^{+}=\mathbb 1(y_i=y_j)
\]

\[
I_{ij}^{-}=\mathbb 1(y_i\neq y_j)
\]

正样本包括Anchor自身。

### 正样本权重

\[
\boxed{
w_{ij}^{+}
=
\frac{
\exp(d_{ij}-d_i^{+,\max})I_{ij}^{+}
}{
\sum_k\exp(d_{ik}-d_i^{+,\max})I_{ik}^{+}
+10^{-6}
}
}
\]

其中：

\[
d_i^{+,\max}
=
\max_j(d_{ij}I_{ij}^{+})
\]

### 负样本权重

\[
\boxed{
w_{ij}^{-}
=
\frac{
\exp(-d_{ij}-d_i^{-,\max})I_{ij}^{-}
}{
\sum_k\exp(-d_{ik}-d_i^{-,\max})I_{ik}^{-}
+10^{-6}
}
}
\]

其中：

\[
d_i^{-,\max}
=
\max_j(-d_{ij}I_{ij}^{-})
\]

### 加权距离

\[
d_i^+=\sum_jw_{ij}^+d_{ij}
\]

\[
d_i^-=\sum_jw_{ij}^-d_{ij}
\]

### 最终损失

官方使用：

`nn.SoftMarginLoss()`

其目标标签为+1：

\[
\boxed{
L_{WRT}
=
\frac1{3B}
\sum_i
\log(1+\exp(d_i^+-d_i^-))
}
\]

参数：

\[
\lambda_{WRT}=1
\]

不增加固定Margin。

### 必须保留的代码细节

- 距离为欧氏距离；
- 输入为pre-BN特征；
- 不执行L2归一化；
- 正样本包含自身；
- 正负样本采用不同的Softmax权重；
- 分母加入\(10^{-6}\)；
- 使用SoftMarginLoss；
- 不修改原有批次正负样本范围。

### 梯度传播

\[
L_{WRT}\rightarrow f\rightarrow ViT
\]

包括QG-SFM的α、β。

WRT不直接更新BNNeck和Classifier。

## 6.3 ProCA：Progressive Center Alignment

官方文件：

`processor/processor.py`

### 输入

\[
f\in\mathbb R^{3B\times768}
\]

按照原始模态拼接顺序恢复：

\[
f\rightarrow(f_R,f_N,f_T)
\]

其中每个模态：

\[
f_m\in\mathbb R^{B\times768}
\]

### 第一步：图像特征L2归一化

\[
\hat f_{y,m,k}
=
Norm(f_{y,m,k})
\]

注意这里归一化的是每一张图像的pre-BN特征。

### 第二步：模态中心聚合

每个身份、每种模态有K=4张图像：

\[
\tilde\mu_{y,m}
=
\frac14
\sum_{k=1}^{4}\hat f_{y,m,k}
\]

随后：

\[
\boxed{
\mu_{y,m}=Norm(\tilde\mu_{y,m})
}
\]

### 第三步：身份中心聚合

三种模态中心：

\[
\tilde\mu_y
=
\frac13
\sum_{m=1}^{3}\mu_{y,m}
\]

随后：

\[
\boxed{
\mu_y=Norm(\tilde\mu_y)
}
\]

### 第四步：停止梯度

\[
\mu_y=\operatorname{detach}(\mu_y)
\]

身份中心作为固定参照。

不能让身份中心通过ProCA损失反向更新所有模态中心。

### 第五步：ProCA损失

\[
\boxed{
L_{ProCA}
=
\frac1P
\sum_y
\frac13
\sum_m
\|\mu_{y,m}-\mu_y\|_2^2
}
\]

参数：

\[
P=16,\quad M=3,\quad K=4
\]

损失权重：

\[
\boxed{\lambda_{ProCA}=0.01}
\]

### 第六步：身份ID提取

官方代码：

```python
ids = target.reshape(-1, group_size)[:, 0].long()
```

其成立条件是同一身份的K张图像在Batch中连续排列。

因此：

**不得在DataLoader、collate或模型拼接过程中改变原始样本顺序。**

### 梯度传播

\[
L_{ProCA}
\rightarrow\mu_{y,m}
\rightarrow\hat f_{y,m,k}
\rightarrow f
\rightarrow ViT
\]

身份中心不接受梯度。

ProCA不直接监督前景区域。

## 6.4 GPD：Global Prototype Discrimination

官方文件：

`processor/processor.py`

### 原型库

\[
P\in\mathbb R^{C_{train}\times768}
\]

每个训练身份维护一个768维原型。

原型库通过：

```python
register_buffer('prototypes', ...)
```

注册为Buffer。

不属于模型可学习参数。

### 当前特征归一化

\[
\hat f_i=Norm(f_i)
\]

### 原型归一化

\[
\hat P_y=Norm(P_y)
\]

在计算GPD时使用当前原型库的副本。

整个原型读取过程位于no_grad环境中。

### 冷启动

若某个身份的原型尚未初始化：

\[
\|P_y\|_2<10^{-6}
\]

则在当前GPD计算中，用当前Batch对应的detach身份中心替代该身份的零原型。

不改变原型库本身的梯度属性。

### 相似度

\[
s_{iy}
=
\frac{\hat f_i^T\hat P_y}{0.03}
\]

温度：

\[
\boxed{\tau=0.03}
\]

### GPD损失

\[
\boxed{
L_{GPD}
=
-\frac1{3B}
\sum_i
\log
\frac{
\exp(\hat f_i^T\hat P_{y_i}/0.03)
}{
\sum_y\exp(\hat f_i^T\hat P_y/0.03)
}
}
\]

全部训练身份原型均参与竞争。

### 原型更新

官方配置：

\[
GPD\_MOMENTUM=0.2
\]

实际更新：

\[
\boxed{
P_y
\leftarrow
Norm(0.8P_y+0.2\mu_y)
}
\]

注意：

- 旧原型权重0.8；
- 新身份中心权重0.2；
- 与论文使用的EMA动量0.8完全一致。

更新过程：

1. 计算GPD损失；
2. 完成反向传播；
3. 完成Optimizer Step；
4. 汇总当前Batch的身份中心；
5. DDP汇总不同Rank的身份中心；
6. 对相同身份的中心求平均；
7. 执行EMA；
8. 归一化更新后的原型。

### 梯度传播

\[
L_{GPD}\rightarrow\hat f\rightarrow f\rightarrow ViT
\]

原型库不参与反向传播。

GPD通过原型提供全局身份判别参照。

## 6.5 最终总损失

\[
\boxed{
L_{total}
=
L_{CE}
+
L_{WRT}
+
0.01L_{ProCA}
+
L_{GPD}
}
\]

四项损失均使用原始UAD机制。

QG-SFM没有独立损失。

也不增加任何Attention拟合、前景保持或背景抑制损失。

---

# 第七章 完整梯度传播与参数更新流程

## 7.1 整体反向传播

```text
                   L_total
                      |
         +------------+------------+
         |            |            |
         CE           WRT       ProCA / GPD
         |            |            |
       BNNeck       Pre-BN CLS   Pre-BN CLS
         |            |            |
         +------------+------------+
                      |
                Final CLS
                      |
               Final LayerNorm
                      |
             Transformer Block 12
                      |
             QG-SFM Attention 12
                      |
             Transformer Block 11
                      |
             QG-SFM Attention 11
                      |
                      ...
                      |
             Transformer Block 1
                      |
             QG-SFM Attention 1
                      |
                Patch Embedding
```

所有损失最终通过CLS特征回传至共享Backbone。

## 7.2 QG-SFM内部梯度

\[
S=QK^T
\]

\[
\widetilde S=S+\Delta S
\]

\[
A=Softmax(\widetilde S/\sqrt{64})
\]

\[
O=AV
\]

反向传播：

\[
\frac{\partial L}{\partial O}
\rightarrow
\frac{\partial L}{\partial A}
\rightarrow
\frac{\partial L}{\partial\widetilde S}
\]

之后分别进入：

- Q分支；
- K分支；
- α参数；
- β参数。

同时Value分支：

\[
\frac{\partial L}{\partial V}
\]

正常传播。

因此，QG-SFM不会阻断原始QKV的梯度。

## 7.3 α、β的梯度来源

以α为例：

\[
\frac{\partial L}{\partial a_l}
=
\frac{\partial L}{\partial\widetilde S_l}
\frac{\partial\widetilde S_l}{\partial\alpha_l}
\frac{\partial\alpha_l}{\partial a_l}
\]

其中：

\[
\frac{\partial\alpha_l}{\partial a_l}
=
1-\tanh^2(a_l/0.2)
\]

β同理。

由于α、β作用于Attention Logits，因此其梯度最终来自最终身份表征的损失。

梯度包含：

- 当前图像；
- 当前层全部Head；
- 当前Query行；
- 当前Key位置；
- 全部四项身份损失。

这些贡献共同形成该层参数的梯度。

## 7.4 行最大值和最小值的梯度

必须保留：

\[
S_i^{max}=\max_jS_{ij}
\]

\[
S_i^{min}=\min_jS_{ij}
\]

在计算图中。

不要执行：

```python
row_max = row_max.detach()
row_min = row_min.detach()
```

STMI的原始机制就是基于当前Attention Logits计算行极值。

本方案保留该计算关系。

## 7.5 哪些参数参与更新

| 对象 | 更新方式 |
|---|---|
| ViT所有原始参数 | 反向传播 |
| QG原始参数a、b | 反向传播 |
| 有效α、β | 由a、b计算 |
| BNNeck | CE反向传播 |
| Classifier | CE反向传播 |
| 掩码 | 不更新 |
| Patch覆盖率 | 不更新 |
| 有效性标记 | 不更新 |
| ProCA身份中心 | Stop Gradient |
| GPD原型库 | EMA更新 |
| QKV投影 | 反向传播 |
| MLP | 反向传播 |
| LayerNorm | 反向传播 |

## 7.6 梯度冲突处理

第一版不增加：

- 梯度投影；
- GradNorm；
- 动态损失权重；
- 梯度手术；
- 单独的前景梯度；
- 参数稀疏正则。

原因是四项损失已经共同作用于最终身份特征。

目前没有实际证据证明必须修改原UAD的优化策略。

如果训练后发现调制参数长期不更新，再针对参数梯度进行诊断，而不是预先增加复杂优化机制。

---

# 第八章 归一化操作的严格顺序

以下内容必须严格按照实际特征来源实现。

| 阶段 | 操作 | 对象 | 归一化方式 |
|---|---|---|---|
| 数据处理 | 掩码值域转换 | 像素掩码 | 0/1 |
| 数据处理 | Patch覆盖率 | 像素掩码 | 16×16平均 |
| Attention | QK计算 | Q、K | 不做L2 |
| Attention | 行最大值/最小值 | 原始Logits | 不做L2 |
| Attention | QG调制 | 原始Logits | 不做L2 |
| Attention | Scale | 调制Logits | 除以8 |
| Attention | Softmax | Logits | Key维Softmax |
| Backbone | LayerNorm | Token特征 | LN |
| Backbone | Final LayerNorm | Token特征 | LN |
| 特征输出 | CLS | 768维特征 | 不做L2 |
| CE | BNNeck | CLS特征 | BatchNorm |
| WRT | pre-BN CLS | 768维 | 不做L2 |
| ProCA | 单图像特征 | pre-BN CLS | L2 |
| ProCA | 模态中心 | 聚合特征 | L2 |
| ProCA | 身份中心 | 聚合特征 | L2+detach |
| GPD | 图像特征 | pre-BN CLS | L2 |
| GPD | 原型 | Prototype | L2 |
| 推理 | CLS特征 | pre-BN CLS | L2 |

尤其注意：

**WRT不能因为GPD使用L2归一化，就跟着改为L2归一化。**

**ProCA不能直接使用BNNeck后的特征。**

**GPD不能直接使用未归一化的CLS特征计算相似度。**

**推理不能将BNNeck后的特征替代原始pre-BN CLS。**

---

# 第九章 代码文件修改方案

本章给出具体修改位置和修改职责，不提供大段可直接替换的代码。

## 9.1 配置文件

涉及：

- `CVPR26_UAD/config/defaults.py`
- `CVPR26_UAD/configs/UAD.yml`

### 修改目的

为QG-SFM增加必要的配置入口。

建议新增一组独立配置：

| 配置 | 设定 |
|---|---|
| QG_SFM_ENABLED | True |
| QG_SFM_LAYERS | 12 |
| QG_SFM_BOUND | 0.2 |
| QG_SFM_INIT | 0 |
| QG_SFM_HEAD_SHARED | True |
| QG_SFM_MASK_ENABLED | True |
| QG_SFM_RANDOM_EXPANSION | False |
| QG_SFM_EMPTY_MASK_POLICY | Disable |

上述配置属于模块实现参数。

不将α、β有效值直接写死在配置文件中，因为它们是训练参数。

### 保留原配置

以下设置不因新增模块而改变：

- `TRANSFORMER_TYPE='vit_base_in'`；
- `STRIDE_SIZE=[16,16]`；
- `SIZE_TRAIN=[256,128]`；
- `SIZE_TEST=[256,128]`；
- `NUM_INSTANCE=4`；
- `PCA_LOSS_WEIGHT=0.01`；
- `GPD_MOMENTUM=0.2`；
- `LOSS_TYPE='base+pca+gpd'`；
- `MAX_EPOCHS=120`；
- 原始Seed。

另外，官方UAD YAML将图像均值和标准差设置为0.5，而`defaults.py`中存在ImageNet均值和标准差的默认值。

最终以合并后的UAD YAML有效配置为准，不能只查看defaults.py。

## 9.2 数据集文件

涉及：

`CVPR26_UAD/datasets/whu_mars.py`

### 修改职责

保留原始PID、Camera、Modality数据解析方式。

新增每张图像的掩码路径关联。

不改变：

- 身份重新编号；
- 模态编号；
- Camera编号；
- Query/Gallery划分；
- PKM采样索引。

不能将掩码按照身份编号关联。

## 9.3 Dataset与Transform

涉及：

`CVPR26_UAD/datasets/bases.py`

### 修改职责

原始ImageDataset只返回图像、PID、Camera。

需要扩展为同时返回：

- 图像；
- 对应掩码或Patch覆盖率；
- 有效性标记；
- PID；
- Camera；
- 模态信息。

新增联合几何增强逻辑。

确保每张图像和掩码使用相同的Resize、Flip、Pad、Crop参数。

RandomErasing只作用于图像。

### 验收条件

对于任意训练样本：

\[
Image_b\leftrightarrow Mask_b
\]

不能出现跨身份、跨图像、跨模态错误关联。

## 9.4 DataLoader与Collate

涉及：

`CVPR26_UAD/datasets/make_dataloader.py`

### 修改职责

原始`train_collate_fn`分别堆叠三个模态图像。

新增对应的三个模态掩码覆盖率和有效性标记。

保持：

\[
X=[X_R;X_N;X_T]
\]

\[
p=[p_R;p_N;p_T]
\]

\[
v=[v_R;v_N;v_T]
\]

严格保证顺序一致。

### 验证DataLoader

原始验证数据集只返回图像和检索元数据。

必须同时返回Query和Gallery对应的掩码。

不能只在训练Dataset中接入掩码。

## 9.5 Backbone Attention

涉及：

`CVPR26_UAD/model/backbones/vit_pytorch.py`

这是整个修改的核心文件。

### Attention

新增：

- 两个原始可学习标量；
- 有界参数化；
- QG-SFM计算；
- 有效性标记输入；
- 软覆盖率输入。

保持原始：

- QKV；
- Scale；
- Softmax；
- Attention Dropout；
- Projection；
- Projection Dropout。

### Block

负责将同一图像的空间权重传递到对应Attention。

不改变Block残差结构和MLP。

### TransReID

负责：

- 接收Patch覆盖率；
- 接收有效性标记；
- 在12个Block之间传递；
- 最终仍返回CLS。

### 位置编码

不修改位置编码。

继续使用原始预训练权重及位置编码插值方式。

## 9.6 模型构建文件

涉及：

`CVPR26_UAD/model/make_model.py`

原始训练前向：

```python
x = torch.cat(list(x), dim=0)
global_feat = self.base(x)
feat = self.bottleneck(global_feat)
cls_score = self.classifier(feat)
```

修改后仍然保持该逻辑，只是额外将掩码信息按照同样顺序传给Backbone。

训练时：

\[
(imgs,masks)\rightarrow base
\]

返回：

\[
(cls\_score,global\_feat,feat)
\]

其中三个输出的定义不变。

推理时：

\[
(img,mask)\rightarrow base
\]

继续根据`TEST.NECK_FEAT`返回pre-BN或after-BN特征。

本方案仍然使用：

`TEST.NECK_FEAT='before'`

## 9.7 Processor

涉及：

`CVPR26_UAD/processor/processor.py`

### 训练部分

原始：

```python
for imgs, vid, camids in train_loader:
```

需要扩展为同时读取：

- 模态图像列表；
- 模态掩码列表；
- 模态有效性列表；
- PID；
- Camera。

模型调用时将掩码作为额外输入。

但是：

以下计算全部保留原样：

- `loss_fn`；
- `PCA`；
- `GPD`；
- `PrototypeBank`；
- 四项损失组合；
- GradScaler；
- 梯度裁剪；
- Optimizer Step；
- 原型库更新。

不要在Processor中重新实现一套ProCA或GPD。

### 验证部分

每个模态的验证Loader也需要返回掩码。

验证时调用模型必须传入对应掩码。

不能在验证时将QG-SFM关闭。

### 推理部分

`do_inference`也需要接收Query和Gallery掩码。

推理过程中不计算损失、不更新α、β。

只执行固定参数的前向特征提取。

## 9.8 Optimizer

涉及：

`CVPR26_UAD/solver/make_optimizer.py`

这是另一个容易遗漏的关键文件。

官方代码对每个可学习参数设置Optimizer Parameter Group。

新增的α、β原始参数默认不属于bias，因此如果不额外处理，会继承普通参数的Weight Decay。

本方案要求：

\[
LR_{SFM}=LR_{base}
\]

\[
WD_{SFM}=0
\]

Momentum：

\[
0.9
\]

因此必须将QG-SFM参数单独设置为一个Optimizer Parameter Group。

必须在创建Scheduler之前完成。

不能通过修改全局Weight Decay为0来实现，否则会影响整个UAD骨干。

## 9.9 Scheduler

涉及：

`CVPR26_UAD/solver/scheduler_factory.py`

不需要修改调度策略。

新增参数组使用与Backbone相同的基础学习率和Scheduler。

原始：

\[
LR_{warmup}=0.01LR_{base}
\]

\[
LR_{min}=0.002LR_{base}
\]

Warmup 5 Epoch，Cosine 120 Epoch。

不额外设置QG-SFM专用Warmup。

## 9.10 损失函数文件

涉及：

- `CVPR26_UAD/loss/wrt_loss.py`
- `CVPR26_UAD/loss/make_loss.py`
- `CVPR26_UAD/processor/processor.py`

第一版不需要修改WRT和CE的定义。

ProCA、GPD继续保留在Processor中。

不要因为增加了QG-SFM而重新封装所有损失函数。

## 9.11 评测文件

涉及：

`CVPR26_UAD/utils/metrics.py`

第一版不修改距离计算、相机过滤和评测协议。

保留：

- Pre-BN特征；
- L2归一化；
- Euclidean Distance；
- 同相机过滤；
- 九种有向模态组合；
- 当前实验使用的mINP扩展。

注意：如果使用的是官方当前版本，该文件本身没有mINP实现。因此必须以你实际运行的评测版本为准，不能用没有mINP的版本覆盖现有评测代码。

## 9.12 Train与Checkpoint

涉及：

`CVPR26_UAD/train.py`

训练主流程不需要重构。

只需要确保配置能够正确启用QG-SFM。

Checkpoint必须保存全部24个QG参数。

加载ImageNet预训练权重时：

- 原始Backbone参数正常加载；
- Position Embedding正常插值；
- QG参数保持零初始化。

加载训练完成的QG-SFM模型时：

- 必须检查24个参数均存在；
- 不允许缺失后静默使用零参数；
- 确保掩码配置与训练时一致。

原型库属于独立训练状态，不属于最终推理特征路径。第一轮不改变原UAD原型库的设计。

---

# 第十章 训练初始化与完整实验设置

## 10.1 初始化顺序

必须按照以下顺序：

1. 读取配置；
2. 创建UAD模型；
3. 创建ViT-B/16；
4. 创建12层QG-SFM；
5. 初始化24个原始参数为0；
6. 加载ImageNet预训练权重；
7. 初始化BNNeck；
8. 初始化Classifier；
9. 创建Loss；
10. 创建Optimizer；
11. 创建Scheduler；
12. 初始化PrototypeBank；
13. 开始训练。

不要在加载ImageNet权重后，对整个模型再次执行统一随机初始化。

否则可能覆盖预训练权重。

## 10.2 ImageNet权重

沿用原始：

`jx_vit_base_p16_224-80ecf9dd.pth`

位置编码继续使用UAD原始插值函数。

QG-SFM的24个参数不在ImageNet权重中，因此保持初始化零值。

## 10.3 ViT配置

| 参数 | 数值 |
|---|---|
| Backbone | vit_base_in |
| Patch Size | 16 |
| Stride | 16 |
| Embed Dim | 768 |
| Depth | 12 |
| Heads | 12 |
| MLP Ratio | 4 |
| QKV Bias | True |
| Drop Path | 0.1 |
| Dropout | 0 |
| Attention Dropout | 0 |
| LayerNorm Epsilon | 1e-6 |

## 10.4 原始UAD训练设置

| 参数 | 官方配置 |
|---|---|
| Optimizer | SGD |
| Base LR | 0.008 |
| Momentum | 0.9 |
| Weight Decay | 1e-4 |
| Bias Weight Decay | 1e-4 |
| Bias LR Factor | 2 |
| Max Epochs | 120 |
| Warmup Epochs | 5 |
| Warmup | Linear |
| LR Minimum | 0.002×Base LR |
| Warmup Initial LR | 0.01×Base LR |
| Gradient Clip | 1.0 |
| Sampler | PKM |
| P | 16 |
| M | 3 |
| K | 4 |
| Num Instance | 4 |

官方YAML的`IMS_PER_BATCH=64`对应每卡64个tuple样本。

因为每个tuple包含三模态图像，所以实际进入Backbone的图像数量为192。

四卡DDP时，若使用全局256个tuple样本，每卡对应64个tuple，即192张实际图像。

官方论文报告四卡DDP训练、每卡192张图像及0.008基础学习率。

不要仅根据GPU数量自动放大学习率。

如果当前基线使用的是不同的实际启动参数，则QG-SFM必须使用同样的参数，保证公平对照。

## 10.5 损失设置

```yaml
METRIC_LOSS_TYPE: wrt
IF_LABELSMOOTH: off
PCA_LOSS_WEIGHT: 0.01
GPD_MOMENTUM: 0.2
LOSS_TYPE: base+pca+gpd
```

保持：

\[
L_{total}
=
L_{CE}+L_{WRT}+0.01L_{ProCA}+L_{GPD}
\]

## 10.6 图像增强设置

继续使用原始UAD：

- Resize 256×128；
- HorizontalFlip 0.5；
- Padding 10；
- RandomCrop 256×128；
- Normalize mean/std 0.5；
- RandomErasing 0.5；
- RandomErasing mode=pixel；
- max_count=1。

新增掩码只同步几何变换，不参与Normalize和RandomErasing。

## 10.7 Seed

继续采用当前UAD实验使用的Seed=1234。

所有对照实验保持一致。

## 10.8 DDP

新增参数必须在DDP包装之前完成注册。

各Rank分别接收本地掩码和图像。

QG-SFM参数通过DDP正常同步梯度。

GPD原型库继续使用原始all_gather机制。

不要将掩码变成全局共享Tensor，也不要跨Rank重新匹配PID。

## 10.9 AMP

保留原始GradScaler。

反向传播：

```text
Total Loss
    |
    v
Scaler.scale(loss)
    |
    v
Backward
    |
    v
Unscale
    |
    v
Global Grad Clip = 1.0
    |
    v
Optimizer Step
    |
    v
Scaler Update
    |
    v
GPD Bank Update
```

QG-SFM参数包含在`model.parameters()`中，因此参与全局梯度裁剪。

---

# 第十一章 推理特征与检索流程

## 11.1 推理必须继续使用掩码

训练和测试必须保持一致：

\[
Image+Mask\rightarrow QG\text{-}SFM
\]

不能在测试阶段关闭QG-SFM。

不能只对Query使用掩码而对Gallery不使用掩码。

Query和Gallery必须采用相同的掩码生成方式和图像预处理规则。

## 11.2 测试数据预处理

UAD原始测试增强：

- Resize到256×128；
- ToTensor；
- Normalize。

掩码对应执行：

- Resize；
- Nearest插值；
- Patch覆盖率计算；
- CLS权重补充；
- 空掩码有效性判断。

不执行训练随机增强。

## 11.3 特征获取

UAD模型：

```python
global_feat = self.base(x)
feat = self.bottleneck(global_feat)

if self.neck_feat == 'after':
    return feat
else:
    return global_feat
```

本方案继续设置：

```yaml
NECK_FEAT: before
FEAT_NORM: yes
```

最终：

\[
\boxed{
f_{test}=CLS_{pre-BN}\in\mathbb R^{768}
}
\]

不使用：

- BNNeck后特征；
- 分类Logits；
- Patch Token拼接；
- 多模态CLS拼接；
- 局部特征；
- 重排序；
- TTA。

## 11.4 L2归一化

\[
\hat f=\frac{f}{\|f\|_2}
\]

在评测器中执行。

## 11.5 距离计算

\[
d(q,g)=\|\hat f_q-\hat f_g\|_2
\]

官方代码使用平方欧氏距离进行排序，平方操作不会改变距离排序。

## 11.6 Gallery过滤

继续使用原始同相机过滤。

不能因为增加掩码而改变Query和Gallery协议。

## 11.7 九种模态方向

继续输出：

| Query | Gallery |
|---|---|
| RGB | RGB |
| RGB | NIR |
| RGB | TIR |
| NIR | RGB |
| NIR | NIR |
| NIR | TIR |
| TIR | RGB |
| TIR | NIR |
| TIR | TIR |

同时保留Ground/Aerial场景评测。

注意：模态方向指标与平台方向指标是不同的统计维度，不能将九种模态组合误当作六种平台—模态场景的全部独立评测。

---

# 第十二章 必须完成的代码正确性验证

本章是正式训练前的实现验收，不是新增研究实验。

## 12.1 数据关联检查

检查：

- 每张图像均有对应掩码；
- RGB、NIR、TIR分别匹配自身掩码；
- PID没有错配；
- Camera没有错配；
- 模态编号没有改变；
- 掩码和图像尺寸一致；
- 掩码值域正确；
- 掩码不包含意外的三通道数据。

对六种模态—平台场景分别统计：

- 掩码数量；
- 空掩码比例；
- 前景面积比例；
- Patch覆盖率分布。

如果某个模态大量出现空掩码，需要先检查掩码生成和映射，不要直接进入训练。

## 12.2 联合增强检查

固定一张图像和对应掩码。

执行翻转、Padding、Crop。

检查图像中的人体位置与掩码中的人体位置是否始终对应。

重点检查：

- 左右翻转；
- Padding区域；
- Crop边界；
- 图像与掩码最终尺寸；
- Patch网格顺序。

## 12.3 Patch覆盖率检查

必须满足：

\[
0\leq\rho_j\leq1
\]

\[
p\in\mathbb R^{129}
\]

\[
p_0=1
\]

Patch数必须为128。

不能出现16×8与8×16的展平顺序错误。

## 12.4 空掩码检查

构造一张有效全零掩码。

确认：

\[
v=0
\]

\[
\Delta S=0
\]

同时模型仍然正常输出CLS。

构造一张非空掩码，确认：

\[
v=1
\]

能够正常进入QG-SFM。

## 12.5 零初始化等价性

使用同一份模型权重、同一张图像、同一掩码，关闭随机Dropout等训练随机性。

分别运行：

- 原始UAD；
- QG-SFM零初始化模型。

检查：

\[
f_{QG,0}\approx f_{UAD}
\]

并比较：

- 每层Attention Logits；
- 每层Attention概率；
- 每层Transformer输出；
- 最终CLS；
- BNNeck输出；
- 分类Logits。

零初始化时：

\[
\Delta S=0
\]

如果输出不一致，首先检查QKV布局、缩放顺序、掩码广播、AMP数据类型和残差位置。

## 12.6 梯度检查

在非空且包含前景和背景Patch的Batch上执行Backward。

检查：

- α原始参数有梯度；
- β原始参数有梯度；
- QKV正常回传；
- Value正常回传；
- Backbone各层正常回传；
- BNNeck正常回传；
- 分类器正常回传；
- PrototypeBank没有梯度；
- ProCA身份中心没有梯度；
- 掩码没有梯度。

不要求每层参数在每个Batch都具有非零梯度。

但必须保证完整计算图存在且没有NaN/Inf。

## 12.7 原始UAD损失对照

使用同一份固定特征，分别调用原UAD和修改后的损失计算流程。

确认：

\[
L_{CE}^{new}=L_{CE}^{old}
\]

\[
L_{WRT}^{new}=L_{WRT}^{old}
\]

\[
L_{ProCA}^{new}=L_{ProCA}^{old}
\]

\[
L_{GPD}^{new}=L_{GPD}^{old}
\]

这里的比较应在QG参数为零、输入特征一致、Bank状态一致的条件下完成。

不要因为模型加入掩码就顺手修改损失实现。

## 12.8 DDP检查

使用至少两个Rank进行小Batch测试。

检查：

- 图像和掩码本地对应；
- QG参数同步；
- 每层梯度正常；
- PrototypeBank一致；
- 不发生通信死锁；
- 不出现Unused Parameters异常。

## 12.9 Checkpoint检查

训练保存后重新加载。

确认：

- Backbone权重一致；
- α、β全部24个参数恢复；
- BNNeck一致；
- Classifier一致；
- 推理掩码正常读取；
- 加载后的CLS特征一致。

---

# 第十三章 正式实验安排

## 13.1 第一阶段：主实验

对比：

- UAD Baseline；
- UAD + QG-SFM。

严格保持：

- 数据划分；
- Seed；
- ImageNet初始化；
- PKM采样；
- 图像增强；
- Loss；
- Optimizer；
- Scheduler；
- 训练Epoch；
- 评测器；
- 相机过滤。

唯一研究变量为QG-SFM。

### 数据集安排

第一轮：

WHU-MARS-1000完整训练。

使用同一模型评测：

- WHU-MARS-1000；
- WHU-MARS-1000GD。

1000GD不单独训练。

完成1000后，再进行WHU-MARS-2337独立训练。

## 13.2 第二阶段：Query门控消融

比较：

### 对照A：软化的STMI关系

\[
\boxed{
\Delta S_{ij}^{soft}
=
\alpha p_ip_j(S_i^{max}-S_{ij})
-
\beta(1-p_ip_j)(S_{ij}-S_i^{min})
}
\]

### 对照B：QG-SFM

\[
\boxed{
\Delta S_{ij}^{QG}
=
p_i[
\alpha p_j(S_i^{max}-S_{ij})
-
\beta(1-p_j)(S_{ij}-S_i^{min})
]
}
\]

二者使用相同的掩码、参数化、初始化和训练配置。

注意，这个软SFM对照是本方案定义的连续化消融，不是STMI官方代码的原始实现。

主要回答：

背景Query是否需要接受直接调制？

## 13.3 第三阶段：空间分布消融

采用均匀Patch覆盖率：

\[
\bar\rho_b=
\frac1{128}\sum_j\rho_{b,j}
\]

令：

\[
p_0=1
\]

\[
p_j=\bar\rho_b
\]

其余部分不变。

该对照保留整体前景面积比例，但消除Patch之间的空间差异。

如果QG-SFM优于均匀覆盖率，说明空间分布提供了额外价值。

如果二者接近，则不能过度强调人体空间定位的贡献。

## 13.4 评价指标

必须同时观察：

- mAP；
- Rank-1；
- Rank-5；
- Rank-10；
- mINP。

以及：

- 九种有向模态组合；
- Ground/Aerial评测；
- 各模态分组表现；
- 参数有效值；
- 实际调制幅度。

特别注意：

不能出现mAP略升、Rank-1明显下降，却仍然判定实验成功的情况。

## 13.5 训练过程诊断

建议记录：

### 参数

每层：

\[
\alpha_l,\beta_l
\]

以及：

\[
a_l,b_l
\]

### 实际调制幅度

为了排除Softmax行平移不影响注意力的情况，定义：

\[
\Delta S^c
=
\Delta S-\operatorname{mean}_j(\Delta S)
\]

记录：

\[
\boxed{
R_l=
\frac{RMS(\Delta S_l^c)}
{RMS(S_l^c)+\epsilon}
}
\]

这个指标用于判断调制是否真正改变了有效注意力竞争。

### 损失

记录：

- CE；
- WRT；
- ProCA；
- GPD；
- Total Loss。

### 训练稳定性

记录：

- α、β是否长期接近边界；
- 参数是否持续更新；
- AMP是否出现跳步；
- 梯度是否异常；
- 各模态训练表现是否分化。

不需要默认计算所有损失对所有24个参数的独立梯度。

---

# 第十四章 需要明确禁止的额外修改

为保证第一轮实验变量清晰，以下内容不纳入当前实现。

1. 不增加前景身份分类器。
2. 不增加背景分类器。
3. 不增加局部Token分类分支。
4. 不增加STR。
5. 不增加CHI。
6. 不增加文本编码器。
7. 不增加教师蒸馏。
8. 不增加逐图可靠性预测网络。
9. 不增加模态专属α、β。
10. 不增加平台专属α、β。
11. 不增加前景Attention拟合损失。
12. 不增加背景保持损失。
13. 不修改WRT。
14. 不修改ProCA。
15. 不修改GPD。
16. 不修改最终CLS提取方式。
17. 不修改推理距离。
18. 不增加重排序。
19. 不增加TTA。
20. 不关闭测试阶段的QG-SFM。

这些不是说它们永远没有价值，而是它们不属于本轮已经确定的研究假设。

---

# 第十五章 最终代码修改任务清单

| 序号 | 文件 | 主要任务 | 是否修改 |
|---|---|---|---|
| 1 | config/defaults.py | 新增QG配置 | 是 |
| 2 | configs/UAD.yml | 设置QG参数及掩码配置 | 是 |
| 3 | datasets/whu_mars.py | 增加图像—掩码映射 | 视实现方式 |
| 4 | datasets/bases.py | 联合增强、覆盖率生成 | 是 |
| 5 | datasets/make_dataloader.py | 训练和测试掩码Collate | 是 |
| 6 | model/backbones/vit_pytorch.py | Attention接入QG-SFM | 是 |
| 7 | model/make_model.py | 模型前向传递掩码 | 是 |
| 8 | processor/processor.py | 训练、验证、推理传递掩码 | 是 |
| 9 | solver/make_optimizer.py | 新增独立参数组 | 是 |
| 10 | solver/scheduler_factory.py | 原调度策略 | 否 |
| 11 | loss/wrt_loss.py | 原始WRT | 否 |
| 12 | loss/make_loss.py | 原始CE/WRT组合 | 否 |
| 13 | utils/metrics.py | 原始评测逻辑 | 否 |
| 14 | train.py | 保持原训练入口 | 原则上否 |

---

# 第十六章 最终完整执行顺序

```text
阶段一：数据准备
    |
    +-- 固定掩码生成方式
    |
    +-- 建立每图像掩码映射
    |
    +-- 检查六种场景掩码
    |
    +-- 检查掩码值域与尺寸
    |
    v
阶段二：数据管线
    |
    +-- Image-Mask联合增强
    |
    +-- 计算Patch软覆盖率
    |
    +-- 构造CLS权重
    |
    +-- 构造有效性标记
    |
    +-- 扩展Train/Test Collate
    |
    v
阶段三：模型修改
    |
    +-- ViT Attention接入QG-SFM
    |
    +-- 全12层独立参数
    |
    +-- 24个零初始化参数
    |
    +-- 有界tanh参数化
    |
    +-- 保留原始CLS输出
    |
    v
阶段四：训练与优化
    |
    +-- ImageNet初始化
    |
    +-- QG参数独立Optimizer Group
    |
    +-- 原始SGD与Scheduler
    |
    +-- 原始CE/WRT/ProCA/GPD
    |
    +-- AMP与DDP
    |
    v
阶段五：正确性检查
    |
    +-- 数据对齐
    |
    +-- 零初始化等价性
    |
    +-- 梯度检查
    |
    +-- 损失对照
    |
    +-- DDP检查
    |
    v
阶段六：正式训练
    |
    +-- WHU-MARS-1000
    |
    +-- 1000GD评测
    |
    +-- WHU-MARS-2337独立训练
    |
    v
阶段七：消融与结果复盘
    |
    +-- QG与软SFM比较
    |
    +-- 真实空间与均匀空间比较
    |
    +-- 九种模态方向分析
    |
    +-- 参数与实际调制分析
    |
    v
最终判断QG-SFM是否真正改善UAD
```

---

# 第十七章 参考文献与代码定位

## 1. STMI论文

Xu, X., Liu, Z., Zhou, W., et al.

**STMI: Segmentation-Guided Token Modulation with Cross-Modal Hypergraph Interaction for Multi-Modal Object Re-Identification.**

AAAI 2026, 40(14), 11433–11441.

- [AAAI论文主页](https://ojs.aaai.org/index.php/AAAI/article/view/38125)
- [论文PDF](https://ojs.aaai.org/index.php/AAAI/article/download/38125/42087)
- [STMI官方GitHub](https://github.com/young6man/STMI)

重点阅读：

| 位置 | 内容 |
|---|---|
| Sec. 3.1，Eq. 1–3 | Token和原始Attention |
| Eq. 4–6 | 行极值及前景关系 |
| Eq. 7–8 | 原始SFM调制 |
| Sec. 5 | 实验及消融 |
| modeling/clip/model.py | SFM实际计算 |
| data/datasets/bases.py | 掩码读取 |
| modeling/make_model.py | 掩码传入方式 |

## 2. UAD论文

Zhao, Y., Zhou, Z., Yang, B., et al.

**WHU-MARS: A Multispectral Aerial-Ground Benchmark Towards Any-Scenario Person Re-Identification.**

CVPR 2026, 25461–25471.

- [CVPR Open Access论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhao_WHU-MARS_A_Multispectral_Aerial-Ground_Benchmark_Towards_Any-Scenario_Person_Re-Identification_CVPR_2026_paper.pdf)
- [WHU-MARS官方GitHub](https://github.com/msm8976/WHU-MARS)

重点阅读：

| 位置 | 内容 |
|---|---|
| Sec. 4.2 | 统一Backbone及PKM采样 |
| Sec. 4.3 | ProCA |
| Sec. 4.4 | GPD |
| Sec. 4.5 | 总损失 |
| Sec. 5.1 | 训练设置 |

## 3. UAD官方代码

- [ViT Backbone](https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/model/backbones/vit_pytorch.py)
- [Model Construction](https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/model/make_model.py)
- [WRT Loss](https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/loss/wrt_loss.py)
- [Loss Construction](https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/loss/make_loss.py)
- [Training Processor](https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/processor/processor.py)
- [Dataset Loader](https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/datasets/bases.py)
- [DataLoader](https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/datasets/make_dataloader.py)
- [PKM Sampler](https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/datasets/sampler.py)
- [DDP PKM Sampler](https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/datasets/sampler_ddp.py)
- [Optimizer](https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/solver/make_optimizer.py)
- [Scheduler](https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/solver/scheduler_factory.py)
- [Evaluation Metrics](https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/utils/metrics.py)

---

# 最终方案结论

本轮代码修改的核心不是在UAD上简单增加一个注意力模块，而是将外部人体空间先验以可学习、连续、受约束的方式引入UAD原始Transformer特征交互过程。

完整设计可以概括为：

**每图像独立掩码 → 联合几何增强 → Patch软覆盖率 → 前景Query门控 → 行极值注意力调制 → 共享ViT统一CLS → 原始四项身份监督 → 统一异构Gallery检索。**

整个方案只有24个新增可学习标量。

原始UAD身份学习体系、特征维度、损失监督和推理描述子全部保持不变。

本方案真正需要验证的是空间先验与身份监督的协同作用，而不是掩码是否让注意力图看起来更集中。

**直击核心总结：以UAD为主干、以STMI的SFM为机制来源、以QG-SFM为自适应改造，优先保证掩码空间对齐、Attention数学顺序、特征归一化及梯度传播完全正确，再进行120 Epoch正式训练。**