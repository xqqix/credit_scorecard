# 信用评分卡 A 卡（申请评分）建模

一个从零实现的申请评分卡（Application Scorecard）项目。使用 UCI 德国信贷数据集，完整走通 **WOE 分箱 → IV 筛选 → 逻辑回归 → 分数转换 → KS/AUC 评估** 的标准流程。

## 项目结构

```
.
├── A卡.py              # 完整建模脚本（6 个步骤，逐段注释）
├── README.md           # 本文件
├── LEARNING_GUIDE.md   # 分步动手指南（学习用）
└── data/
    ├── german_credit.csv              # 原始数据（UCI，1000 条）
    ├── german_credit_readable.csv     # 可读列名版（建模用这个）
    └── scorecard.csv                  # 产出的评分卡表（每箱分值）
```

## 快速开始

```bash
pip install pandas numpy scikit-learn
python A卡.py
```

脚本会打印各步骤结果，并在 `data/` 下生成 `scorecard.csv`。

## 建模流程

| 步骤 | 内容 | 关键点 |
|---|---|---|
| 1. 读数据 | 加载德国信贷数据 | 1000 条，20 个特征，好坏标签 |
| 2. 切分 | 训练/测试 7:3 | `stratify=y` 保持好坏比例一致 |
| 3. WOE 函数 | `calc_woe_iv()` | 按箱统计好坏占比，加 0.5 平滑 |
| 4. 分箱与筛选 | 连续变量 qcut 分箱，类别变量直接编码 | **分箱边界和 WOE 只在训练集计算**，测试集复用 |
| 5. 评分转换 | 逻辑回归系数 → 每箱分值 | `score = A − B × ln(odds)`，PDO=20 |
| 6. 评估 | KS / AUC / 评分分层 | 分数越高越安全（注意 AUC 方向） |

## 模型效果

| 指标 | 训练集 | 测试集 |
|---|---|---|
| KS | 0.5231 | 0.5079 |
| AUC | 0.8238 | 0.7984 |

评分分层（测试集按分数分 5 档）坏客户率单调递减：

| 档位 | 1（最低分） | 2 | 3 | 4 | 5（最高分） |
|---|---|---|---|---|---|
| 坏客户率 | 65.0% | 45.0% | 20.0% | 13.3% | 6.7% |

## 核心概念

- **WOE（Weight of Evidence）**：`ln(该箱好客户占比 / 该箱坏客户占比)`，正值表示该箱好人多、安全。
- **IV（Information Value）**：各箱 `(好占比 − 坏占比) × WOE` 之和，衡量变量整体的区分能力，本项目的筛选阈值是 0.02。
- **评分卡转换**：把逻辑回归的线性系数拆成每个分箱的分值，最终得分 = 基准分 + 各特征所在箱的分值之和。业务人员无需懂模型也能解释"这个人为什么多少分"。

## 数据来源

UCI Machine Learning Repository — [Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)

原始数据 class=2（坏客户）已转换为 `target_default=1`，class=1 → 0。

## 适用场景

适合入门信用风控建模，理解评分卡从数据到分数的完整链路。生产环境需要考虑更多因素（特征稳定性监控、跨时间验证、拒绝推断等）。

## License

MIT
