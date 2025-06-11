import pandas as pd
import numpy as np
from snownlp import SnowNLP
from sklearn.preprocessing import StandardScaler


# 数据清洗
## 缺失值处理
def handle_missing_data(df):
    print("=== 数据清洗前缺失值统计 ===")
    print(df.isnull().sum())
    print(f"数据清洗前数据量: {len(df)} ")

    # 去重
    df.drop_duplicates(subset=['title', 'author'], inplace=True)

    # 填充缺失值（推荐写法，避免 FutureWarning）
    df.fillna({
        'author': '未知作者',
        'isbn': '未知',
        'category': '未知',
        'detailed_category': '未知',
        'publisher': '未知出版社',
        'publish_time': '未知时间',
        'introduction': '无简介'
    }, inplace=True)

    rate = df['total_readers'].mean() / df['reading_count'].mean()
    df['total_readers'] = df['total_readers'].fillna(df['reading_count'] * rate)

    # 只删除 word_count 为空的行
    df.dropna(subset=['word_count'], inplace=True)

    print("=== 数据清洗后缺失值统计 ===")
    print(df.isnull().sum())
    print(f"数据清洗前数据量: {len(df)} ")


## 数据类型转换
def convert_data_types(df):
    print("=== 数据清洗前数据类型统计 ===")
    print(df.dtypes)

    df['total_readers'] = df['total_readers'].astype(int)

    # 百分号字符串转小数
    def percent_to_float(x):
        if pd.isnull(x):
            return np.nan
        x = str(x).strip()
        if x.endswith('%'):
            try:
                return round(float(x.rstrip('%')) / 100, 3)
            except:
                return np.nan
        try:
            return float(x)
        except:
            return np.nan
    df['recommendation'] = df['recommendation'].apply(percent_to_float)

    # # 统一字数列单位
    def parse_word_count(x):
        if pd.isnull(x):
            return np.nan
        x = str(x).replace(',', '').replace(' ', '')
        try:
            return int(float(x))
        except:
            return np.nan

    df['word_count'] = df['word_count'].apply(parse_word_count)

    print("=== 数据清洗后数据类型统计 ===")
    print(df.dtypes)

## 数据标准化
def standardize_data(df):
    cols_to_standardize = [
        'reading_count', 'total_readers', 'word_count',
        'rating_count', 'recommend_count', 'general_count', 'bad_count'
    ]
    scaler = StandardScaler()
    for col in cols_to_standardize:
        if col in df.columns:
            df[col + '_zscore'] = scaler.fit_transform(df[[col]].astype(float))


## 特征工程
def feature_engineering(df):
    #  处理出版时间
    df['publish_year'] = pd.to_datetime(df['publish_time'], format='%Y年%m月', errors='coerce').dt.year.astype('Int64')
    df['publish_month'] = pd.to_datetime(df['publish_time'], format='%Y年%m月', errors='coerce').dt.month.astype('Int64')

    #上架时长
    df['shelf_duration'] = (pd.to_datetime('now') - pd.to_datetime(df['publish_time'], format='%Y年%m月', errors='coerce')).dt.days

    # 字数分级
    df['word_level'] = pd.cut(df['word_count'], bins=[0, 20000, 50000, 100000, np.inf],
                              labels=['短篇', '中篇', '长篇', '超长篇'], right=False)

    df['intro_length'] = df['introduction'].apply(len)
    # 使用SnowNLP进行中文情感分析

    df['intro_sentiment'] = df['introduction'].apply(lambda x: SnowNLP(x).sentiments if isinstance(x, str) else np.nan)
    print(df.dtypes)


def main():
    csv_path = '../data/weread_books_detailed.csv'
    save_path = '../data/weread_books_cleaned.csv'

    df = pd.read_csv(csv_path)

    handle_missing_data(df)
    convert_data_types(df)
    standardize_data(df)
    feature_engineering(df)

    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()

    """
    title：书名
    author：作者
    isbn：ISBN号码
    reading_count：阅读数量
    total_readers：总阅读人数
    recommendation：推荐信息
    evaluation：评价等级
    category：原始分类
    detailed_category：详细分类
    publisher：出版社
    publish_time：出版时间
    word_count：字数
    rating_count：点评人数
    recommend_count：推荐数
    general_count：一般数
    bad_count：不行数
    introduction：简介
    url：书籍链接
    """
