import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# 数据清洗
## 缺失值处理
def handle_missing_data(df):
    print("=== 数据清洗前缺失值统计 ===")
    print(df.isnull().sum())
    print(f"数据清洗前数据量: {len(df)} ")

    # 去重
    df.drop_duplicates(subset=['title', 'author'], inplace=True)

    # 填充缺失值
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

    print("=== 数据清洗后数据类型统计 ===")
    print(df.dtypes)


## 数据标准化
def standardize_data(df):
    # 百分号字符串转小数
    def percent_to_float(x):
        if pd.isnull(x):
            return np.nan
        if x == '无推荐':
            return -1  # 用 1 标记
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

    # 数值型特征标准化
    numeric_cols = [
        'reading_count', 'total_readers', 'word_count',
        'rating_count', 'recommend_count', 'general_count', 'bad_count'
    ]
    scaler = StandardScaler()
    for col in numeric_cols:
        if col in df.columns:
            df[col + '_zscore'] = scaler.fit_transform(df[[col]].astype(float))

    # 文本数据标准化
    text_cols = ['title', 'author', 'publisher']
    for col in text_cols:
        if col in df.columns:
            # 去除多余空格，统一为半角字符
            df[col] = df[col].str.strip().str.normalize('NFKC')
            # 处理title列中的括号、引号和中文标点
            if col == 'title':
                # 删除小括号及其内容
                df[col] = df[col].str.replace(r'\([^)]*\)', '', regex=True)
                # 删除方括号及其内容
                df[col] = df[col].str.replace(r'【[^】]*】', '', regex=True)

    # 编码
    encoded_cols = ['category', 'detailed_category', 'evaluation', 'title', 'author', 'publisher']
    for col in encoded_cols:
        if col in df.columns:
            try:
                # 获取唯一值并创建编码映射
                unique_values = df[col].unique()
                encoding_map = {val: idx for idx, val in enumerate(unique_values)}

                # 创建新的编码列
                encoded_col_name = f'{col}_code'
                df[encoded_col_name] = df[col].map(encoding_map)

                # 保存编码映射
                mapping_df = pd.DataFrame({
                    'original': list(encoding_map.keys()),
                    'encoded': list(encoding_map.values())
                })
                print(f"成功对 {col} 列进行编码，映射关系：\n{mapping_df.head()}")

            except Exception as e:
                print(f"对 {col} 列进行编码时出错：{str(e)}")


## 特征工程
def feature_engineering(df):
    # 处理出版时间
    unknown_time_mask = df['publish_time'] == '未知时间'

    df.loc[unknown_time_mask, 'publish_year'] = -1
    df.loc[unknown_time_mask, 'publish_month'] = -1
    df.loc[unknown_time_mask, 'shelf_duration'] = -1

    # 处理已知时间的数据
    df.loc[~unknown_time_mask, 'publish_year'] = pd.to_datetime(
        df.loc[~unknown_time_mask, 'publish_time'],
        format='%Y年%m月',
        errors='coerce'
    ).dt.year

    df.loc[~unknown_time_mask, 'publish_month'] = pd.to_datetime(
        df.loc[~unknown_time_mask, 'publish_time'],
        format='%Y年%m月',
        errors='coerce'
    ).dt.month

    df.loc[~unknown_time_mask, 'shelf_duration'] = (
            pd.to_datetime('now') - pd.to_datetime(
        df.loc[~unknown_time_mask, 'publish_time'],
        format='%Y年%m月',
        errors='coerce'
    )
    ).dt.days

    # 字数分级
    bins = [0, 20000, 50000, 100000, np.inf]
    labels = ['短篇', '中篇', '长篇', '超长篇']
    df['word_level'] = pd.cut(df['word_count'], bins=bins,
                              labels=labels, right=False)

    df['intro_length'] = df['introduction'].apply(len)

    # 简单情感分析
    positive_words = ['好', '优秀', '精彩', '推荐', '喜欢', '感人',
                      '有趣', '值得', '精彩', '深刻', '震撼']
    negative_words = ['差', '糟糕', '失望', '无聊', '一般', '枯燥',
                      '平庸', '尴尬', '烂', '垃圾']

    def sentiment(x):
        x = str(x)
        if any(word in x for word in positive_words):
            return '正面'
        elif any(word in x for word in negative_words):
            return '负面'
        else:
            return '中性'

    df['intro_sentiment'] = df['introduction'].apply(sentiment)

    print("=== 特征工程后数据类型统计 ===")
    print(df.dtypes)


def main():
    csv_path = '../data/weread_books_detailed.csv'
    save_path = '../data/weread_books_cleaned.csv'

    df = pd.read_csv(csv_path)

    handle_missing_data(df)
    convert_data_types(df)
    standardize_data(df)
    feature_engineering(df)
    print(df.dtypes)

    #df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()


