import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 创建数值型特征列表
numeric_features = [
    'reading_count', 'total_readers', 'recommendation', 'word_count',
    'rating_count', 'recommend_count', 'general_count', 'bad_count',
    'publish_year', 'shelf_duration', 'intro_length']

# 分类特征统计
categorical_features = ['category', 'detailed_category', 'evaluation',
                        'word_level', 'intro_sentiment']


def basic_statistic(df):
    # 数值型特征描述性统计
    desc_stats = df[numeric_features].describe()
    print("\n数值型特征描述性统计:")
    print(desc_stats.round(3))

    print("\n分类特征统计:")
    for feature in categorical_features:
        if feature in df.columns:
            print(df[feature].value_counts().head(10))


def correlation_analysis(df):
    """相关性分析"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 计算相关系数矩阵
    corr_matrix = df[numeric_features].corr()

    # 可视化相关性热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, square=True, fmt='.2f',
                cbar_kws={'label': '相关系数'})
    plt.title('变量相关性热力图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 找出高相关性特征对
    threshold = 0.7
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_value
                })

    if high_corr_pairs:
        print("\n高相关性特征对 (|r| > 0.7):")
        for pair in high_corr_pairs:
            print(f"{pair['feature1']} - {pair['feature2']}: {pair['correlation']:.3f}")

    # 强相关性对散点图
    strong_corr = corr_matrix.abs() > threshold
    strong_pairs = []
    for i in range(len(strong_corr.columns)):
        for j in range(i + 1, len(strong_corr.columns)):
            if strong_corr.iloc[i, j]:
                strong_pairs.append((strong_corr.columns[i], strong_corr.columns[j]))

    if strong_pairs:
        x_col, y_col = strong_pairs[0]  # 取第一个强相关对
        plt.figure(figsize=(8, 6))
        plt.scatter(df[x_col], df[y_col], alpha=0.6, color='red')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'{x_col} vs {y_col} 散点图')
        plt.tight_layout()
        plt.show()


def category_analysis(df):
    """分类分析"""
    print("分类维度分析")

    # 分类阅读量分析
    category_stats = df.groupby('category').agg({
        'reading_count': 'mean',
        'total_readers': 'mean',
        'recommendation': 'mean',
        'rating_count': 'mean'
    }).round(2)

    print(category_stats)


    # 评价等级分析
    if 'evaluation' in df.columns:
        evaluation_stats = df.groupby('evaluation').agg({
            'reading_count': 'mean',
            'total_readers': 'mean',
            'recommendation': 'mean'
        }).round(2)

        print("\n评价等级统计:")
        print(evaluation_stats)

    return category_stats


def main():
    df = pd.read_csv('../data/weread_books_cleaned.csv')

    basic_statistic(df)
    correlation_analysis(df)
    category_analysis(df)


if __name__ == "__main__":
    main()
