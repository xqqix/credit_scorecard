# -*- coding: utf-8 -*-
"""
A 卡建模 - 自己动手版
按照 README.md 的步骤，一步一步把代码写在这个文件里。
每一步都写了任务说明，代码留空由你来填。
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
# ============================================================
# 第 1 步：读数据，认识数据
data = pd.read_csv("D:/python_learning/评分卡·/data/german_credit_readable.csv")
#   2. 打印数据形状
# print(data.shape)
#   3. 打印前 5 行
# print(data.head(5))
#   4. 打印 target_default 列的分布
# print(data['target_default'].value_counts())      # 1坏用户 ，0好用户
# ============================================================
# 第 2 步：切分训练集和测试集
x = data.drop('target_default', axis=1)
y = data['target_default']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42,stratify=y)
# print(x_train.shape)
# print(y_train.shape)
# ============================================================
# 第 3 步：写分箱 + WOE 函数
def calc_woe_iv(bins, y):
    # 第1步：拼表（bins 和 y 都是 Series，取 .values 对齐）
    df = pd.DataFrame({'bin': bins.values, 'y': y.values})

    # 第2步：按箱统计。坏人数 = y求和(1就是坏)，总数 = 计数
    g = df.groupby('bin')['y'].agg(bad='sum', total='count')

    # 第3步：好人数 = 总数 - 坏人数；再各加 0.5 平滑
    g['good'] = g['total'] - g['bad']
    g['bad'] = g['bad'] + 0.5
    g['good'] = g['good'] + 0.5

    # 第4步：分母是"全部坏/全部好"的总数（不是总样本数）
    bad_total = g['bad'].sum()
    good_total = g['good'].sum()
    g['bad_pct'] = g['bad'] / bad_total
    g['good_pct'] = g['good'] / good_total

    # 第5步：WOE 和 IV
    g['woe'] = np.log(g['good_pct'] / g['bad_pct'])
    g['iv_part'] = (g['good_pct'] - g['bad_pct']) * g['woe']
    iv = g['iv_part'].sum()

    return g, iv

# 试算一个变量，验证函数是否正确
# test_bins = x_train['checking_account_status'].astype(str)
# table, iv = calc_woe_iv(test_bins, y_train)
# print(table[['bad', 'good', 'woe']])
# print('IV =', iv)
# ============================================================
# 第 4 步：分箱、算 IV、筛变量、转 WOE
numeric_cols = ['duration_months', 'credit_amount', 'age']

# 三个字典：把每变量算出的"尺子"存起来，测试集要复用
iv_dict = {}       # 变量名 → IV
bin_edges = {}     # 变量名 → 分箱边界（连续变量才有）
woe_maps = {}      # 变量名 → {箱: WOE} 字典

# 阶段1：对每个变量，在训练集上做尺子、算 IV
for col in x_train.columns:
    if col in numeric_cols:
        # 连续变量：qcut 算边界 → cut 切箱
        series = x_train[col]
        _, edges = pd.qcut(series, q=5, retbins=True, duplicates='drop')
        bin_edges[col] = edges
        bins_train = pd.cut(series, bins=edges, include_lowest=True).astype(str)
    else:
        # 类别变量：本身就是箱，直接转字符串
        bin_edges[col] = None
        bins_train = x_train[col].astype(str)

    table, iv = calc_woe_iv(bins_train, y_train)
    iv_dict[col] = iv
    woe_maps[col] = dict(zip(table.index, table['woe']))
    # print('%-28s IV = %.4f' % (col, iv))

# 阶段2：筛出 IV >= 0.02 的变量
selected = [col for col, iv in iv_dict.items() if iv >= 0.02]
# print('\n保留的变量:', selected)

# 阶段3：把原始数据换成 WOE 值（训练/测试共用上面的尺子）
def to_woe(x):
    out = pd.DataFrame(index=x.index)
    for col in selected:
        if bin_edges[col] is not None:
            bins = pd.cut(x[col], bins=bin_edges[col], include_lowest=True).astype(str)
        else:
            bins = x[col].astype(str)
        out[col] = bins.map(woe_maps[col]).fillna(0.0)
    return out

x_train_woe = to_woe(x_train)
x_test_woe = to_woe(x_test)
# print('\n转换后前5行（全是WOE数值）:')
# print(x_train_woe.head())
# ============================================================
# 第 5 步：训练逻辑回归，转评分卡
estimator = LogisticRegression()
estimator.fit(x_train_woe, y_train)
# print(f'系数:{estimator.coef_}')
# print(f'截距:{estimator.intercept_}')
PDO = 20
B = PDO / np.log(2)
A = 600 + B * np.log(1/20)
intercept = estimator.intercept_[0]
base_score = A - B * intercept
# print('基准分:', round(base_score, 2))

rows = []
for i, col in enumerate(selected):
    for bin_name, woe in woe_maps[col].items():
        points = -B * estimator.coef_[0][i] * woe
        rows.append({'变量': col, '箱': bin_name, 'WOE': round(woe, 4), '分值': round(points, 2)})

scorecard = pd.DataFrame(rows)
scorecard.to_csv('data/scorecard.csv', index=False, encoding='utf-8-sig')
# print(scorecard.head(20))

# ============================================================
# 第 6 步：评估效果（KS / AUC / 评分分层）
train_score = A - B * estimator.decision_function(x_train_woe)
test_score = A - B * estimator.decision_function(x_test_woe)

print('训练集前5个分数:', train_score[:5])
print('分数范围:', round(train_score.min(), 1), '~', round(train_score.max(), 1))

def calc_ks(y,score):
    # 1. 拼表并按分数排序
    df = pd.DataFrame({'y': np.asarray(y), 'score': np.asarray(score)})
    df = df.sort_values('score').reset_index(drop=True)
    # 2. 累计占比：cumsum 除以各自总数
    df['cum_bad'] = (df['y'] == 1).cumsum() / (df['y'] == 1).sum()
    df['cum_good'] = (df['y'] == 0).cumsum() / (df['y'] == 0).sum()

    # 3. 最大差距
    return (df['cum_bad'] - df['cum_good']).abs().max()
ks_train = calc_ks(y_train, train_score)
ks_test = calc_ks(y_test, test_score)
print('KS 训练集:', round(ks_train, 4))
print('KS 测试集:', round(ks_test, 4))

auc_train = roc_auc_score(y_train, -train_score)
auc_test = roc_auc_score(y_test, -test_score)
print('AUC 训练集:', round(auc_train, 4))
print('AUC 测试集:', round(auc_test, 4))

test_y_values = np.asarray(y_test)
test_df = pd.DataFrame({'score': test_score, 'y': test_y_values})
test_df['档位'] = pd.qcut(test_df['score'], 5, labels=[1, 2, 3, 4, 5])

tier = test_df.groupby('档位', observed=True)['y'].agg(人数='count', 坏客户数='sum', 坏客户率='mean')
print(tier)
