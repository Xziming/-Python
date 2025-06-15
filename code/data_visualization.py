import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import seaborn as sns


def overview_data(df):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 今日阅读数量分布
    plt.figure(figsize=(8, 6))
    plt.hist(df['reading_count'], bins=50, alpha=0.7, color='skyblue',
             range=(df['reading_count'].min(), df['reading_count'].quantile(0.95)))
    plt.title('今日阅读数量分布')
    plt.xlabel('阅读数量')
    plt.ylabel('频次')
    plt.tight_layout()
    plt.show()

    # 好评率分布
    plt.figure(figsize=(8, 6))
    recommendation_data = df[df['recommendation'] != -1]['recommendation']
    plt.hist(recommendation_data, bins=30, alpha=0.7, color='lightgreen')
    plt.title('好评率分布')
    plt.xlabel('好评率')
    plt.ylabel('频次')
    plt.tight_layout()
    plt.show()

    # 字数分级分布
    plt.figure(figsize=(8, 6))
    word_level_counts = df['word_level'].value_counts()
    colors = sns.color_palette('pastel')[0:len(word_level_counts)]
    plt.pie(word_level_counts.values, labels=word_level_counts.index, autopct='%1.1f%%', startangle=140, colors=colors,
            textprops={'fontsize': 12})
    plt.title('字数分级分布', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 评价等级分布
    plt.figure(figsize=(9, 6))
    eval_counts = df['evaluation'].value_counts()
    if '无评价' in eval_counts.index:
        eval_counts = eval_counts.drop('无评价')
    bar_colors = sns.color_palette('Set2')[0:len(eval_counts)]
    plt.bar(eval_counts.index, eval_counts.values, color=bar_colors, edgecolor='black')
    plt.title('评价等级分布', fontsize=16, fontweight='bold')
    plt.xlabel('评价等级', fontsize=13)
    plt.ylabel('数量', fontsize=13)
    plt.xticks(rotation=30, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

    # 出版年份趋势
    plt.figure(figsize=(10, 6))
    # 过滤掉 publish_year 为 -1 的数据
    valid_years = df[df['publish_year'] != -1]
    year_counts = valid_years['publish_year'].value_counts().sort_index()
    plt.plot(year_counts.index, year_counts.values, marker='o', color='#A259F7', linewidth=2)
    plt.title('出版年份趋势', fontsize=16, fontweight='bold')
    plt.xlabel('年份', fontsize=13)
    plt.ylabel('书籍数量', fontsize=13)
    plt.grid(linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

    # 简介情感分布
    plt.figure(figsize=(7, 5))
    sentiment_counts = df['intro_sentiment'].value_counts()
    plt.bar(sentiment_counts.index, sentiment_counts.values, color=['#6CC24A', '#F7B32B', '#E94F37'])
    plt.title('简介情感分布')
    plt.xlabel('情感')
    plt.ylabel('数量')
    plt.tight_layout()
    plt.show()

    # 简介长度分布
    plt.figure(figsize=(8, 6))

    # 计算直方图数据但不绘制
    counts, bins, _ = plt.hist(df['intro_length'], bins=30,
                              range=(df['intro_length'].min(), df['intro_length'].quantile(0.95)),
                              alpha=0)
    # 获取每个bin的中心点
    bin_centers = (bins[:-1] + bins[1:]) / 2
    # 绘制折线图
    plt.plot(bin_centers, counts, color='green', linewidth=2, marker='o', markersize=4)

    plt.title('简介长度分布')
    plt.xlabel('简介长度')
    plt.ylabel('频次')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def category_analysis(df):
    """分类分析"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 原始分类TOP10
    plt.figure(figsize=(8, 6))
    top_categories = df['category'].value_counts().head(10)
    plt.barh(range(len(top_categories)), top_categories.values)
    plt.yticks(range(len(top_categories)), top_categories.index)
    plt.title('原始分类TOP10')
    plt.xlabel('书籍数量')
    plt.tight_layout()
    plt.show()

    # 详细分类TOP15
    plt.figure(figsize=(8, 6))
    top_detailed = df['detailed_category'].value_counts().head(15)
    plt.barh(range(len(top_detailed)), top_detailed.values)
    plt.yticks(range(len(top_detailed)), top_detailed.index)
    plt.title('详细分类TOP15')
    plt.xlabel('书籍数量')
    plt.tight_layout()
    plt.show()

    # 不同分类的平均好评率
    plt.figure(figsize=(8, 6))
    category_rating = df.groupby('category')['recommendation'].mean().sort_values(ascending=False).head(10)
    plt.bar(range(len(category_rating)), category_rating.values, color='lightcoral')
    plt.xticks(range(len(category_rating)), category_rating.index, rotation=45, ha='right')
    plt.title('不同分类平均好评率TOP10')
    plt.ylabel('平均好评率')
    plt.tight_layout()
    plt.show()

    # 不同分类的平均阅读量
    plt.figure(figsize=(8, 6))
    category_reading = df.groupby('category')['reading_count'].mean().sort_values(ascending=False).head(10)
    plt.bar(range(len(category_reading)), category_reading.values, color='lightblue')
    plt.xticks(range(len(category_reading)), category_reading.index, rotation=45, ha='right')
    plt.title('不同分类平均阅读量TOP10')
    plt.ylabel('平均阅读量')
    plt.tight_layout()
    plt.show()


def bar_of_pie(df):
    """
    分类分布的 bar of pie 图
    """

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.style.use('ggplot')

    category_counts = df['category'].value_counts()
    threshold = 0.025
    mask = category_counts / category_counts.sum() >= threshold
    other_count = category_counts[~mask].sum()

    # 获取大于阈值的类别，并添加"其他"类别
    filtered_counts = category_counts[mask]
    if other_count > 0:
        filtered_counts['其他'] = other_count

    ratios = filtered_counts.values / filtered_counts.values.sum()
    labels = filtered_counts.index

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(wspace=0.2)

    # 饼图参数
    explode = [0.08] + [0] * (len(ratios) - 1)
    angle = -180 * ratios[0]
    wedges, *_ = ax1.pie(ratios, autopct='%1.1f%%', startangle=angle,
                         labels=labels, explode=explode, textprops={'fontsize': 12})
    ax1.set_title('分类分布（饼图）', fontsize=16, fontweight='bold')

    # 条形图参数
    width = .3
    bottom = 1

    # 找出占比最大的主分类
    main_category = df['category'].value_counts().index[0]
    # 筛选该主分类下的所有子分类
    detailed_counts = df[df['category'] == main_category]['detailed_category'].value_counts()
    detailed_ratios = detailed_counts.values / len(df[df['category'] == main_category])
    detailed_labels = detailed_counts.index

    for j, (height, label) in enumerate(reversed([*zip(detailed_ratios, detailed_labels)])):
        bottom -= height
        bc = ax2.bar(0, height, width, bottom=bottom, color=f'C{j}', label=label, alpha=0.7)
        ax2.bar_label(bc, labels=[f"{height:.0%}"], label_type='center', fontsize=11)
    ax2.set_title('分类分布（条形图）', fontsize=16, fontweight='bold')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.axis('off')
    ax2.set_xlim(-2.5 * width, 2.5 * width)

    # 连接线
    theta1, theta2 = wedges[0].theta1, wedges[0].theta2
    center, r = wedges[0].center, wedges[0].r
    bar_height = sum(ratios)

    # 上连接线
    x = r * np.cos(np.pi / 180 * theta2) + center[0]
    y = r * np.sin(np.pi / 180 * theta2) + center[1]
    con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData,
                          xyB=(x, y), coordsB=ax1.transData, color='black', linewidth=2)
    ax2.add_artist(con)

    # 下连接线
    x = r * np.cos(np.pi / 180 * theta1) + center[0]
    y = r * np.sin(np.pi / 180 * theta1) + center[1]
    con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData,
                          xyB=(x, y), coordsB=ax1.transData, color='black', linewidth=2)
    ax2.add_artist(con)

    plt.tight_layout()
    plt.show()


def main():
    # 读取数据
    df = pd.read_csv('../data/weread_books_cleaned.csv')

    overview_data(df)
    category_analysis(df)
    bar_of_pie(df)


if __name__ == "__main__":
    main()
