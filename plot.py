import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体（根据系统自行调整，如 SimHei, Microsoft YaHei 或 CJK 字体）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_length_vs_ppl(df):
    # 1. 计算 Token 长度 (字符数)
    df['Token_Len'] = df['Text'].apply(len)

    # 删除 PPL <= 0 的 Exact match 节点
    df = df[df['PPL'] > 0]

    # 2. 这里的 PPL 可能会有离群值，为了绘图美观，限制一下范围
    plot_df = df[df['PPL'] < 1.3].copy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # --- 图表 1: 散点图 + 趋势线 (查看个体分布与整体趋势) ---
    sns.regplot(data=plot_df, x='Token_Len', y='PPL',
                scatter_kws={'alpha':0.2, 's':10},
                line_kws={'color':'red'}, ax=ax1)
    ax1.set_title('Token 长度与 PPL 的原始分布及趋势', fontsize=14)
    ax1.set_xlabel('Token 长度 (字符数)')
    ax1.set_ylabel('PPL (越低越好)')

    # --- 图表 2: 分组平均 PPL (查看特定长度的健康度) ---
    # 计算每个长度的平均 PPL 和 Token 数量
    len_stats = df.groupby('Token_Len')['PPL'].agg(['mean', 'count']).reset_index()

    # 我们只看样本量足够的长度（比如超过 10 个 Token 的长度段）
    len_stats = len_stats[len_stats['count'] > 5]

    sns.barplot(data=len_stats, x='Token_Len', y='mean', palette='coolwarm', ax=ax2)

    # 在柱状图上方标注该长度下的 Token 数量
    for i, row in len_stats.iterrows():
        ax2.text(i, row['mean'] + 0.001, f"n={int(row['count'])}",
                 ha='center', va='bottom', fontsize=9, rotation=45)

    ax2.set_title('不同长度 Token 的平均 PPL (样本量需 > 5)', fontsize=14)
    ax2.set_xlabel('Token 长度 (字符数)')
    ax2.set_ylabel('平均 PPL')
    ax2.set_ylim(1.0, len_stats['mean'].max() * 1.02) # 聚焦差异

    plt.tight_layout()
    plt.savefig('length_vs_ppl.png', dpi=300)
    plt.show()

def plot_token_audit(df):
    # 2. 定义分级逻辑 (基于你的“拍脑袋”目标)
    def classify(ppl):
        ppl = float(ppl)
        if ppl <= 1.0001: return 'A (<= 1.0001)'
        elif ppl <= 1.001: return 'B (<= 1.001)'
        elif ppl <= 1.01:  return 'C (<= 1.01)'
        elif ppl <= 1.1:   return 'D (<= 1.1)'
        else:              return 'E (> 1.1)'

    df['Grade'] = df['PPL'].apply(classify)

    # 创建画布
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2)

    # --- 图表 1: 饼图 (各等级占比) ---
    ax1 = fig.add_subplot(gs[0, 0])
    grade_counts = df['Grade'].value_counts().sort_index()
    colors = sns.color_palette("viridis", len(grade_counts))
    ax1.pie(grade_counts, labels=grade_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=140, explode=[0.05]*len(grade_counts))
    ax1.set_title('Qwen3.5 35B-A3B MXFP4 长Token语义理解等级分布', fontsize=14)

    # --- 图表 2: PPL 分布直方图 (聚焦核心区域) ---
    ax2 = fig.add_subplot(gs[0, 1])
    # 过滤掉极端的离群值便于观察分布情况
    sns.histplot(df[df['PPL'] < 1.02]['PPL'], bins=50, kde=True, ax=ax2, color='teal')
    ax2.axvline(1.001, color='green', linestyle='--', label='B/C 边界')
    ax2.axvline(1.01, color='orange', linestyle='--', label='C/D 边界')
    ax2.set_title('Token PPL 密度分布 (聚焦 1.0 - 1.02)', fontsize=14)
    ax2.legend()

    # --- 图表 3: ID vs PPL 散点图 (寻找词表演化规律) ---
    ax3 = fig.add_subplot(gs[1, :])
    sns.scatterplot(data=df, x='ID', y='PPL', hue='Grade', alpha=0.5, ax=ax3, palette='viridis')
    ax3.set_ylim(1.0, 1.2) # 限制Y轴范围观察细节
    ax3.set_title('Token ID 与 PPL 的关系 (检测3.5新词的质量)', fontsize=14)
    ax3.set_xlabel('Token ID')
    ax3.set_ylabel('Last Token Perplexity (Low = Better)')

    plt.tight_layout()
    plt.savefig('qwen3.5_token_audit.png', dpi=300)
    plt.show()

    # 输出统计报告
    print("--- 词表审计简报 ---")
    print(df['Grade'].value_counts())
    print(f"\n平均 PPL: {df['PPL'].mean():.4f}")
    print(f"中位数 PPL: {df['PPL'].median():.4f}")

# 调用函数 (将 'data.csv' 换成你的文件名)
df = pd.read_csv('black_hole.csv')
df = df[df['PPL'] > 0]
plot_token_audit(df.copy())
plot_length_vs_ppl(df.copy())