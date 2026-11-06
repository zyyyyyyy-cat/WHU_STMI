import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/training_log_review_20261105'
s=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
b=json.loads((OUT/'baseline_comparison.json').read_text(encoding='utf-8'))
names={'1':'RGB','2':'NIR','3':'TIR'}
def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+
                     ['|'+'|'.join(map(str,row))+'|' for row in rows])

overall=table(['指标','论文 UAD','四卡 UAD 复现','QG-SFM','QG−复现（百分点）'],
              [[r['metric'],f"{r['paper']:.2f}",f"{r['baseline']:.2f}",f"{r['qg']:.2f}",f"{r['delta']:+.2f}"] for r in b['overall']])
pairs=table(['Query→Gallery','论文 R1/mAP/mINP','复现 R1/mAP/mINP','QG R1/mAP/mINP','ΔR1','ΔmAP','ΔmINP'],
            [[names[r['direction'][0]]+'→'+names[r['direction'][2]],
              '/'.join(f"{r['paper_'+m]:.2f}" for m in ('rank1','map','minp')),
              '/'.join(f"{r['baseline_'+m]:.2f}" for m in ('rank1','map','minp')),
              '/'.join(f"{r['qg_'+m]:.2f}" for m in ('rank1','map','minp')),
              *[f"{r['delta_'+m]:+.2f}" for m in ('rank1','map','minp')]] for r in b['pairs']])
ag=table(['方向/指标','论文','UAD 复现','QG','Δ（百分点）'],
         [[r['direction']+'/'+r['metric'],f"{r['paper']:.2f}",f"{r['baseline']:.2f}",f"{r['qg']:.2f}",f"{r['delta']:+.2f}"] for r in b['ag']])
layer=table(['层','α 有效值','β 有效值','R','dα/draw'],
            [[r['layer'],f"{r['alpha']:.6f}",f"{r['beta']:.6f}",f"{r['ratio']:.6f}",f"{r['alpha_derivative']:.4f}"] for r in s['final_qg']])
table3=table(['数据集/指标','论文','UAD 复现','本轮 QG'],
             [[r['dataset']+'/'+r['metric'],f"{r['paper']:.2f}",f"{r['baseline']:.2f}",
               f"{r['qg']:.2f}" if r['qg'] is not None else '未提供'] for r in b['table3']])

text=f'''# WHU-MARS UAD + QG-SFM 完整训练日志分析（2026-11-05）

## 结论

本轮已完成真实四卡 120 轮训练与一次第 40→41 轮的完整断点恢复。QG 参数确实学习、调制确实非零，
但相对用户提供的四卡 UAD 复现，当前是接近 baseline 的小幅混合变化，尚未形成实质、全面的检索提升。
总体 Rank-1 +0.25、mAP −0.02、Rank-5 +0.22、Rank-10 −0.37 个百分点。
跨平台 A→G/G→A 的 Rank-1 +0.81/+0.78 值得保留，但 mAP −0.04/−0.17，不构成整体成功。

本轮没有修改任何训练、模型、损失、数据集或测试源码，只新增本分析目录和 tmp 下分析脚本。

## 1. 输入证据与对照选择

- 当前 QG 日志：logs/train_log_20261105_0937.txt（1451 行，epoch 1–40）；
  logs/train_log_20261105_1106.txt（2798 行，恢复后 epoch 41–120）。共 4249 行。
- 基线目录：D:/CODE/Doing/WHU_Code/WHU/logs。已读取其中六份日志，共 3711 行。
  主对照取 1000/1000_train_log_20260911_2145.txt 和 1000/1000_test_log_20260912_1037.txt。
  训练末轮验证和独立测试的总体、九方向、AG 指标全部一致。
- 1000/2337train_1000test_log_20261025_1457.txt 是 2337 训练、1000 测试，不能充当 1000 训练基线。
  本报告明确没有混用它；1000GD/2337 日志仅用于对应论文表 3 的相应数据集行。
- 当前 guide.md、WHU 模型/数据/processor/优化器/调度器/评测/恢复代码；STMI 论文及 modeling/clip/model.py、
  modeling/make_model.py、RGBNT201 配置；WHU 论文表 3–6 已同时检查 PDF 原页及提取文本。
- 每份日志的 SHA256、配置差异、完整统计已导出到 summary.json、baseline_comparison.json 和 CSV。

## 2. 是否真正使用了可比配置

两次训练的主要设置一致：四卡；global tuple batch=256；每卡 64 组、192 张图；LR=0.032；
seed=1234；SGD momentum=0.9；120 轮；warmup=5；worker=8；PKM K=4；共享 ViT-B；
256×128；pre-BN CLS；L2 测试；不 rerank、不 top-k 截断；ProCA 权重 0.01、GPD 新中心权重 0.2。
所有共有配置中实际差异为：预训练路径写法、输出目录和测试 batch（UAD 512，QG 1024）。
预训练文件同名，但没有服务器文件哈希，不能严格证明两个文件二进制相同。

基线独立测试日志中的 BASE_LR=0.008、IMS_PER_BATCH=64 是测试脚本的默认配置，不能据此误判其训练条件。
对应训练日志明确是四卡、256、0.032，独立测试加载的是 UAD_1000_DDP/transformer_120.pth。

当前 WHU 源码与该基线目录的 whu_mars、单卡/DDP sampler、make_loss、WRT、scheduler_factory、metrics
逐字节一致。模型/processor 的 QG、AMP 保护、断点保存和恢复是本轮新增。

仍需承认对照边界：两次运行时间/硬件负载不同，基线没有新版 AMP 跳步日志与恢复逻辑，
原基线在 AMP 跳过 SGD 时仍更新原型，而新版会同时跳过原型更新。并且测试 batch 不同。
所以这是主要训练配置对齐的历史对照，还不是同版本、同硬件、多 seed 的严格单变量统计实验。
这些边界不会把接近零的差异变成已证明的大幅提升，也不能凭一个 seed 宣称显著差异。

## 3. 总体结果与论文表 3

数值均为百分比，差值为百分点。

{overall}

论文报告本身只保留一位小数，应避免对其尾数做过度解释。
从论文看 Rank-1 +0.46；但相对实际复现仅 +0.25。不能将全部论文差距归因 QG。
当前 mAP 与 baseline 基本持平，Rank-10 略低，不满足“主要指标共同提升”的主实验目标。

论文三个数据集设置的对照：

{table3}

本次 QG 的 FILTER_GD=False，尚无其 1000GD 测试或 2337 独立训练日志；不能填写未做的结果。
1000GD 应使用同一个 1000-trained final checkpoint 测试，不额外训练。

## 4. 九种模态方向、表 5/6 与 AG 表 4

{pairs}

相对实际复现：Rank-1 为 3 个方向上升、5 个下降、1 个持平；mAP 为 3 个上升、6 个下降；
mINP 为 3 个上升、5 个下降、1 个持平。没有形成跨模态方向的一致优势。

一个特别容易误读的例子：RGB→RGB 相对论文 R1 +1.50，但基线复现原本已比论文 +2.01；
实际加入 QG 后是 42.48→41.97，下降 0.51。因此不能把 +1.50 报为模块贡献。

VI 两个主要方向：NIR→RGB 的 Rank-1 不变、mAP +0.03、mINP +0.08；
TIR→RGB 的 Rank-1 +0.23，但 mAP −0.06、mINP −0.03。
所以当前不能声称热红外被 QG 大幅损害，也不能声称它已系统解决热红外检索问题。
此前仅看到相对论文的落后时，不足以进行这种归因；实际复现证明多数差距已经存在。

论文 VI 部分还有 144×288 的尺寸描述，当前代码及两份实际训练均固定 256×128。
因此本报告以实际复现作为模块效果主对照，论文表值仅做参考，不擅自改变用户现有协议。

{ag}

AG 的 Rank-1 是较值得保留的正向信号，但 mAP 没有同步提高。
这与“部分 query 的第一名变好，更多正样本的整体排序没有一起变好”相容；
并不证明每张 query 的变化都如此，需要保存检索结果逐 query 对比才能定位。
混合 gallery 的总体 Rank-1 与限制单模态 gallery 的九格指标并非同一排名问题，不能互相求平均替代。
总体 mINP 未在日志输出，也不能拿九格 mINP 的平均值代替。

## 5. 训练与续训稳定性

- 120 个 epoch 的完整保存记录连续；总尝试 13704 次，成功优化 13696 次，AMP 跳过 8 次，占约 0.0584%。
- 跳步出现在 epoch 1（3 次）、42、61、81、99、111，均非连续失控；日志没有 NaN/Inf 损失或崩溃堆栈。
- 我按原生 scaler 的 2000 次增长间隔、×2 增长、×0.5 回退，重建整个历史；每轮记录的 scale 和成功更新数
  都能一致对上，包括恢复边界。不能把后面回到 32768 后发生的少量跳步误判成持续发散。
- epoch40 保存 successful_updates=4564、skips=3、scale=32768；恢复日志明确 completed=40、next=41、scale=32768。
  epoch41 累计更新继续为 4678。LR=0.0240→0.0237，而不是重启 warmup。
- epoch40→41 的 GPD=0.283→0.270、QG 参数也平滑接续，没有原型冷启动或模型重置的迹象。
- 这些证据支持真实服务器续训流程已接续运行。但没有收到 checkpoint 文件，不能逐张量验证其存储内容，
  也不承诺跨 GPU/进程重建的逐位一致性；物理 GPU 编号不在这些日志里。
- 两份日志没有复现此前 rendezvous Broken pipe，不能反推先前失败根因已经确定。

## 6. 损失几乎与 UAD 重合，不能只用低 loss 证明检索有效

第 120 轮第 100 批的 rank0 running mean：

|项|UAD 复现|QG|
|---|---|---|
|Total|0.290|0.287|
|CE|0.100|0.098|
|WRT|0.111|0.110|
|ProCA（未乘权重）|0.197|0.196|
|GPD|0.078|0.077|
|训练 Acc|97.8%|97.9%|

QG 的重建值：0.098+0.110+0.01×0.196+0.077=0.28696，与记录 0.287 一致。
所有 240 个损失快照均按此关系检查，残差只在日志舍入量级。
ProCA 的真实加权值约 0.00196，不能把日志原始 0.196 当成总损失中同等大小的一项。
这符合原设计权重，不是据此自动增大权重的理由。

训练曲线显示模型已能拟合训练身份；并没有出现 loss 不下降/分类头未学习等明显实现故障。
但训练身份分类准确率和未见身份的跨场景检索是不同任务，97.9% 不能换算成检索应接近 97.9%。
验证 mAP 从 5.84→11.00 持续增加，110→120 轮 mAP 仍 +0.02、R1 只 −0.03，
没有“训练越来越好、验证持续恶化”的明确证据，不能武断定性为严重过拟合。
同时末段增益已经很小，简单延长到 200 轮不是当前最有依据的处理。

## 7. QG 在计算上生效，但学到的不是统一前景抬升

下面是 epoch120、iteration100 的训练前向快照，不是最终保存权重的精确数值，
也不是三模态/全数据的统计均值；它仅对应 rank0 当时的混合模态 batch。

{layer}

全部 α 为负，β 在第 6、12 层也为负。方案本来允许正负值，代码也没有把符号接反：

ΔS_ij = v p_i [α p_j (M_i−S_ij) − β (1−p_j)(S_ij−m_i)]。

R 的定义是去除行均值后的 delta RMS / 原 logits RMS。
当前各层约 0.013–0.126，说明不是“只有行平移而被 softmax 完全抵消”的无效操作；
但 R=0.126 不等于准确率提高 12.6%，也不等于前景 attention mass 提高 12.6%。

对有效前景 Query、纯前景 Key（p_i=p_j=1）：S'_j=(1−α)S_j+αM。
α<0 时，低于最大值的前景 logit 会下调，前景内部高低差被放大，而非把所有前景都抬高。
纯背景 Key 在 β>0 时向行最小值收缩。两者相对竞争不能只看 α 或 β 单个符号。

第 6/12 层 α、β 都为负时，对上述纯前/背景 Key，前景 logit 下调、背景 logit 上调。
这已偏离“每层都增强前景、压制背景”的口头直觉，但不是偏离允许双符号的数学方案。
它可能反映保留细粒度差异、利用上下文或对错误掩码的一种全局适应；现有日志不能区分这些原因，
更不能把负值直接说成逐图掩码纠错。

全前景时 p=1，有 softmax(S')=softmax((1−α)S)，退化成温度调节。
接近全图时也可能主要体现这种效应；但已有 R08 抽样并非普遍全图，不能声称本次整体就是全图退化。

第 6 层 α≈−0.190682，约达到边界幅度的 95.3%，dα/draw≈0.091；
它受到一定 tanh 梯度压缩，但没有全部参数卡死。绝大部分 R 稳定在非零水平。
不能只因出现 −0.19 就立即放大 bound、去掉有界化，或违反方案将参数强制非负。

## 8. Query 门控与 CLS 读出的衔接，是后续消融的关键

本模型最终只读 pre-BN CLS。CLS 的 p_i 始终为 1，并且不受空 mask 外的额外质量评价控制。
对相同输入、相同参数，QG 与连续化 soft-SFM 的差值为：

ΔS_QG − ΔS_soft = v β (1−p_i)(S−m)。

因此 CLS 行（p_i=1）直接调制完全相同；差别集中在 p_i<1 的 patch Query 行。
Query 门控必须通过改变 patch 的表示，再在后续层影响 CLS，体现额外价值。
尤其最后一层在输入固定时，改变背景 Query 行不会影响同层 CLS 输出，因为注意力按 Query 行计算、
后面的投影/MLP 又逐 token 处理。不是“每个新增门控在每层都直接改了最终 CLS”。

这不证明 Query 门控无效，也不构成应改代码的 bug，但解释了为什么“QG 有动作”不等于
“Query 门控贡献已被证明”。需要 guide 第 13.2 节中已经规划的 soft-SFM 对照。

## 9. STMI 的迁移边界与掩码质量

STMI 的核心 SFM 在 QK 后、scale/softmax 前，按层独立且 head 共享，这些连接在当前方案中保留。
但其 CLIP 视觉主干、约 0.1 的随机正初值、硬 token mask、背景→前景小概率扰动、
STR/CHI 和多模态融合/文本监督，与当前 ImageNet 共享 ViT、零初始化、软覆盖率、
无扰动、仅 CLS、UAD 四损失并不相同。
STMI 的组件实验或全系统结果，不能当成 WHU 上只迁移 QG-SFM 就一定提升的保证。

STMI 论文自己说明边界歧义和误分割，并以背景 token 随机翻转增强鲁棒性；它也不保证 SAM 是真值。
当前按 guide 没有加入掩码扩张、没有质量预测器；valid 仅检查增强后是否非空。
所以这里 QG 是 Query-gated，不是 Quality-gated。非空错掩码不会自动被识别并关闭。

先前 300 张 R08 复核已发现背包/局部人体等失败例子，提示这种风险真实存在；
但没有分割真值、没有逐模态 attention/梯度/coverage 统计，不能把当前的小幅混合结果直接归罪于 SAM。
α、β 是跨身份/模态/平台共享的全局参数，也无法直接为某一张 Thermal 错 mask 单独纠错。

当前数据流与 guide 一致：逐图独立 mask、图像/掩码同步几何增强、mask 最近邻和零 padding、
每模态独立随机增强、平均池化 16×16、CLS=1、空 mask 关调制、RandomErasing 不改语义 mask，
四损失仍分别接 BN classifier 和 pre-BN 特征；没有发现日志与这些连接相矛盾的确定性错误。
但日志不能替代服务器实际文件哈希和逐样本可视化，不能据此宣称分割语义都正确。

## 10. 运行代价

历史 UAD 训练 loop 合计 {b['baseline_train_seconds']:.1f} 秒，QG 为 {s['training_seconds']:.1f} 秒，
观察值约增加 {(s['training_seconds']/b['baseline_train_seconds']-1)*100:.1f}%。
非验证轮从 epoch done 到完整保存成功，中位数约 7.23 秒/轮。
QG 虽然只新增 24 个参数，但逐层全注意力矩阵运算、掩码 PNG 读取和新增 AMP 梯度检查都有代价，
不能把“参数量小”写成“计算和训练开销可以忽略”。
两组不是同时同负载的性能基准，此差异不能全部归因 QG，更不能当作 FLOPs 或精确推理延迟结论。

## 11. 下一步的优先顺序

1. 固定保存本次 final120 结果，不将 epoch110 的最高 Rank-1 与 epoch120 的最高 mAP 拼成一套结果。
   先用同一 final checkpoint 独立测试 full1000 与 1000GD，测试 batch 可先统一为 baseline 的 512。
   检查 standalone 与训练末轮验证的一致性，并补全论文表 3 的 GD 列；这一步无需重新训练或重新生成掩码。
2. 对同版本实现做最小必要对照。现有历史 UAD 是主要参照；如果要把 0.1–0.3 的差异做因果结论，
   再用当前修复后的 AMP/processor、QG 关闭重跑 baseline，并保持相同测试 batch 和恢复安排。
   不需要为历史 baseline 先反复跑很多 seed 来掩盖本轮总体缺少收益的问题。
3. 优先执行 guide 已有的 uniform coverage 消融：保持每图面积、去掉 patch 空间差异。
   如果 real mask 没有优于 uniform，就没有空间定位收益证据，应优先解决空间先验利用问题，
   而不是继续微调 tanh bound 或损失权重。
4. 执行已有 soft-SFM vs QG 比较：相同 mask、初值、有界化、优化器和训练协议。
   它回答保护背景 Query 是否有额外收益。若 soft-SFM 更好，说明这部分门控假设需要修订；
   不能把消融结果倒写成先验必然结论。
5. 如仍没有明显收益，再用难例和逐模态诊断决定是否引入可靠性/区域读出等新机制。
   这些属于下一版研究方案，须有单独消融，不在本轮静默加入，不直接堆 STR/CHI 或任意新损失。
6. 目前不建议先铺开 2337 全量训练、重复一轮相同掩码生成、盲目修改 .032 学习率或延长训练。
   先建立本轮的空间先验与 Query 门控是否提供额外价值的证据。

## 12. 导出材料

- baseline_training_comparison.png：真实 UAD 与 QG 的总体指标、训练损失和准确率曲线。
- matched_baseline_delta.png：九方向相对真实复现的变化；paper_reference_delta.png 仅相对论文。
- training_dashboard.png：全部训练轨迹与逐层 α/β/R 热图。
- matched_baseline_*.csv、table3_three_way.csv：可直接审阅的两/三方对照表。
- training.csv、qg_layers.csv、checkpoints.csv、evaluation_*.csv、amp_overflows.csv：完整解析记录。
- summary.json、baseline_comparison.json：源文件摘要、计数、配置与时序校验。

总体判断：当前实现和训练流程有真实运行证据，QG 也有真实计算作用；
但第一轮并未证明它改善了 UAD 的整体检索能力。保留跨平台 Rank-1 的正向信号，
用已有的空间/Query 消融确定价值来源，是比继续宣称“机制预期有效”更可靠的下一步。
'''
(OUT/'训练日志综合分析.md').write_text(text,encoding='utf-8')
print('Report written:',OUT/'训练日志综合分析.md',len(text),'characters')
