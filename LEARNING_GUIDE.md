# A 卡建模 · 自己动手指南

这个项目只有两样东西：`data/`（数据）和 `A卡.py`（你写代码的地方）。
不要一次写完，按下面的步骤一步一步来——**每一步先自己尝试写，卡住了再看提示，还不行再来问我。**

数据文件：`data/german_credit_readable.csv`（1000 行，21 列）
目标变量：`target_default`（1=坏客户，0=好客户）

---

## 第 1 步：读数据，认识数据 ← 从这里开始

**目标**：把数据读进来，看清楚它长什么样。

**要写的**：
1. `import pandas as pd`
2. 用 `pd.read_csv()` 读 `data/german_credit_readable.csv`
3. 打印这三个东西：
   - 数据形状（多少行、多少列）
   - 前 5 行
   - `target_default` 这一列的值分布（`value_counts()`）

**思路提示**：相对路径写 `'data/german_credit_readable.csv'`（假设你在项目根目录运行）。

**自检问题**：
- 数据有多少行、多少列？
- 好客户和坏客户各多少？比例大概是多少？

---

## 第 2 步：切分训练集和测试集

**目标**：像你做负荷预测那样，把数据切成训练集和测试集。

**要写的**：
1. `from sklearn.model_selection import train_test_split`
2. 把 `target_default` 作为 y，其余列作为 X
3. 用 `train_test_split` 切分，`test_size=0.3`，`random_state=42`
4. **新增**（A 卡特有）：加参数 `stratify=y`——保证训练集和测试集里好坏客户的比例一致
5. 打印训练集和测试集各自的行数、坏客户占比

**自检问题**：为什么 A 卡要加 `stratify=y`，而负荷预测不用？（提示：想想回归和分类的区别）

---

## 第 3 步：写分箱 + WOE 函数（核心）

**目标**：理解 WOE，并写成函数。这一步是 A 卡最难也最重要的地方。

**先搞懂公式**：
```
WOE = ln( 该箱好客户数 / 全部好客户数  ÷  该箱坏客户数 / 全部坏客户数 )
```
- WOE > 0：这箱好人多，安全
- WOE < 0：这箱坏人多，危险

**要写的**：
写一个函数 `calc_woe_iv(bins, y)`，其中 `bins` 是分箱后的 Series（每个样本属于哪个箱），`y` 是 0/1 目标值。
函数内部：
1. 按箱分组，统计每箱的 `好客户数`、`坏客户数`、`总数`
2. 每箱的坏客户数、好客户数各加 0.5（防止某箱没有坏客户时 ln 算成无穷）
3. 算 `bad_pct` = 该箱坏客户 / 全部坏客户，`good_pct` = 该箱好客户 / 全部好客户
4. 算 `woe = np.log(good_pct / bad_pct)`
5. 算 `iv_part = (good_pct - bad_pct) * woe`，IV = 所有箱的 `iv_part` 之和
6. 返回（每箱的统计表, IV 值）

**思路提示**：用 `pd.DataFrame({'bin': bins.values, 'y': y.values})` 装起来，然后 `groupby('bin')['y'].agg(...)`。

**自检问题**：随便找一个变量（比如 `checking_account_status`）算一次，看它的 WOE 是正还是负？符合直觉吗？

---

## 第 4 步：分箱、算 IV、筛变量

**目标**：对每个变量做分箱 → 算 WOE → 用 IV 筛掉弱变量。

**要写的**：
1. 定义一个连续变量列表 `NUMERIC_COLS = ['duration_months', 'credit_amount', 'age']`
2. 对**连续变量**：用 `pd.qcut(series, q=5, retbins=True, duplicates='drop')` 算出分箱边界 `edges`，再用 `pd.cut(series, bins=edges, include_lowest=True).astype(str)` 切箱
3. 对**类别变量**：直接 `.astype(str)` 当作箱
4. 对每个变量调用第 3 步的 `calc_woe_iv`，记录 WOE 表和 IV
5. 筛出 `IV >= 0.02` 的变量
6. 把原始数据里每个变量的值，替换成它所在箱的 WOE 值

**⚠️ 最容易踩的坑**：分箱边界和 WOE 只能在**训练集**上算，测试集套用训练集的规则。否则数据泄漏，结果虚高。

**自检问题**：
- 哪些变量 IV 最高？
- 有没有变量被筛掉（IV < 0.02）？

---

## 第 5 步：训练逻辑回归，转评分卡

**目标**：训练模型，并把模型变成一张"每个箱加多少分"的表。

**要写的**：
1. `from sklearn.linear_model import LogisticRegression`
2. 用第 4 步得到的 WOE 特征训练逻辑回归（注意：这里不用 XGBoost，因为评分卡必须能用线性系数拆成每箱的分数）
3. 评分转换：
   - `B = PDO / ln(2)`，其中 `PDO = 20`
   - `A = 基准分 + B * ln(基准赔率)`，基准分=600，基准赔率=1/20
   - `基准分_最终 = A - B * 截距`
   - 每个箱的分值 = `-B * 系数 * 该箱WOE`
4. 生成一张评分卡表（变量、箱、WOE、分值），存成 `data/scorecard.csv`

**自检问题**：
- 找一个 WOE 为正的箱，它的分值是正还是负？（应该为正——好人加分）

---

## 第 6 步：评估效果

**目标**：看模型好不好。

**要写的**：
1. 算 `KS`：把样本按分数排序，分别算累计坏客户占比和累计好客户占比，两者最大差距就是 KS。自己写个函数实现。
2. `from sklearn.metrics import roc_auc_score` 算 AUC
3. 打印训练集和测试集的 KS、AUC
4. 把测试集按分数分 5 档（`pd.qcut`），看每档的坏客户率是否从高到低递减

**自检问题**：
- 测试集 KS 大概多少？（> 0.3 算可用）
- 5 档的坏客户率是不是单调递增？

---

## 完成后

全部跑通后，把你的 `A卡.py` 和控制台输出发给我，我帮你检查：
- IV 筛选是否合理
- KS/AUC 是否正常
- 评分分层是否符合预期

有任何一步卡住，随时来问——告诉我你在第几步、报什么错或者哪里想不通。
