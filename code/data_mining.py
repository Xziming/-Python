import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

"""
def clustering_analysis(df, n_clusters=5):
    # 选择用于聚类的特征
    # 选择用于建模的特征
    clustering_features = [ 'shelf_duration','intro_length', 'publish_year', 'publish_month']
    # 添加zscore特征, 和编码特征
    clustering_features.extend([col for col in df.columns if 'zscore' in col])
    clustering_features.extend([col for col in df.columns if '_code' in col])

    print(f"聚类特征: {clustering_features}")

    # 准备聚类数据，处理缺失值和非数值型数据
    cluster_data = df[clustering_features].copy()
    for col in clustering_features:
        cluster_data[col] = pd.to_numeric(cluster_data[col], errors='coerce')
    cluster_data = cluster_data.fillna(cluster_data.mean())

    # 标准化数据
    scaler = StandardScaler()
    cluster_data_scaled = scaler.fit_transform(cluster_data)

    # 确定最佳聚类数
    silhouette_scores = []
    k_range = range(2, min(11, len(cluster_data) // 10))  # 避免聚类数过多

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(cluster_data_scaled)
        score = silhouette_score(cluster_data_scaled, labels)
        silhouette_scores.append(score)

    # 选择最佳k值
    if silhouette_scores:
        best_k = k_range[np.argmax(silhouette_scores)]
        print(f"最佳聚类数: {best_k} (轮廓系数: {max(silhouette_scores):.3f})")
    else:
        best_k = n_clusters
        print(f"使用默认聚类数: {best_k}")

    # K-means聚类
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(cluster_data_scaled)

    # 添加聚类标签到数据框
    df['cluster'] = cluster_labels

    # 分析各聚类的特征
    analysis_cols = ['reading_count', 'total_readers', 'recommendation',
                     'word_count', 'rating_count']
    available_cols = [col for col in analysis_cols if col in df.columns]

    if available_cols:
        cluster_stats = df.groupby('cluster')[available_cols].mean().round(2)
        print(f"\n聚类结果统计:")
        print(cluster_stats)

    # PCA降维可视化
    if len(clustering_features) > 1:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(cluster_data_scaled)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title(f'K-means聚类结果 (PCA降维可视化, K={best_k})')
        plt.xlabel(f'第一主成分 (解释方差: {pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'第二主成分 (解释方差: {pca.explained_variance_ratio_[1]:.2%})')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # 各聚类的分类分布
    if 'category' in df.columns:
        cluster_category = pd.crosstab(df['cluster'], df['category'])
        print("\n各聚类的分类分布:")
        print(cluster_category)

    return df
"""

def predict_analysis(df, target='recommendation', cv_folds=5):
    """预测分析"""

    # 选择用于建模的特征
    feature_cols = [ 'shelf_duration','intro_length', 'publish_year', 'publish_month']
    # 添加zscore特征, 和编码特征
    feature_cols.extend([col for col in df.columns if 'zscore' in col])
    feature_cols.extend([col for col in df.columns if '_code' in col])

    # 只保留存在的特征
    available_features = [col for col in feature_cols if col in df.columns]
    print(f"使用特征: {available_features}")

    # 准备特征数据
    df_filtered = df[df[target] != -1].copy()

    # 再提取特征和目标变量

    X = df_filtered[feature_cols]
    y = df_filtered[target]

    print(f"有效样本数: {len(y)}")
    print(f"目标变量统计: 最小值={y.min():.3f}, 最大值={y.max():.3f}, 均值={y.mean():.3f}")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # 定义模型列表
    models = {
        '随机森林': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        '梯度提升': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        'SVR': SVR(C=100, epsilon=0.1, kernel='rbf')
    }

    results = {}
    best_model_name = None
    best_score = -np.inf

    # 训练和评估模型
    for name, model in models.items():
        print(f"\n=== {name} ===")

        # 训练模型
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 计算评估指标
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"RMSE: {rmse:.4f}, R²: {r2:.4f}")

        # 交叉验证
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2', n_jobs=-1)
        print(f"交叉验证R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'rmse': rmse,
            'r2': r2,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }

        # 记录最佳模型
        if cv_scores.mean() > best_score:
            best_score = cv_scores.mean()
            best_model_name = name


    if not results:
        print("所有模型训练失败!")
        return

    print(f"\n最佳模型: {best_model_name} (CV R²: {best_score:.4f})")

    # 可视化分析
    visualize_results(results, X, y, X_test, y_test, best_model_name)

    # 超参数调优 (针对最佳模型)
    if best_model_name and len(X) > 100:  # 数据足够多才进行调优
        hyperparameter_tuning(X, y, best_model_name)

    return results


def visualize_results(results, X, y, X_test, y_test, best_model_name):
    """结果可视化"""
    if not results:
        return

    # 1. 模型 R² 比较
    model_names = list(results.keys())
    r2_scores = [results[name]['r2'] for name in model_names]

    plt.figure(figsize=(8, 6))
    plt.bar(model_names, r2_scores, color='skyblue', alpha=0.7)
    plt.title('模型 $R^2$ 比较')
    plt.ylabel('$R^2$ 分数')
    plt.xlabel('模型')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # 2. 模型 RMSE 比较
    rmse_scores = [results[name]['rmse'] for name in model_names]

    plt.figure(figsize=(8, 6))
    plt.bar(model_names, rmse_scores, color='lightcoral', alpha=0.7)
    plt.title('模型 RMSE 比较')
    plt.ylabel('RMSE')
    plt.xlabel('模型')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # 3. 最佳模型的真实值 vs 预测值
    if best_model_name and best_model_name in results:
        y_pred = results[best_model_name]['y_pred']

        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.6, color='green')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title(f'{best_model_name}: 真实值 vs 预测值')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 4. 残差分析图
        residuals = y_test - y_pred

        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.6, color='purple')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('预测值')
        plt.ylabel('残差')
        plt.title(f'{best_model_name}: 残差分析')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # 学习曲线 (仅针对最佳模型)
    if best_model_name and best_model_name in results:
        plot_learning_curve(results[best_model_name]['model'], X, y, best_model_name)


def plot_learning_curve(model, X, y, title):
    """绘制学习曲线"""
    try:
        train_sizes = np.linspace(0.1, 1.0, 5)
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=3, scoring='r2', n_jobs=-1,
            train_sizes=train_sizes, random_state=42
        )

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label="训练得分")
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')

        plt.plot(train_sizes, val_mean, 'o-', color='red', label="验证得分")
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')

        plt.title(f"{title} 的学习曲线")
        plt.xlabel("训练样本数")
        plt.ylabel(f"$R^2$ 分数")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"学习曲线绘制失败: {str(e)}")


def hyperparameter_tuning(X, y, model_name):
    """超参数调优"""
    print(f"\n=== {model_name} 超参数调优 ===")

    try:
        if model_name == '随机森林':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
            model = RandomForestRegressor(random_state=42, n_jobs=-1)

        elif model_name == '梯度提升':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            model = GradientBoostingRegressor(random_state=42)
        else:
            return

        # 网格搜索
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1
        )
        grid_search.fit(X, y)

        print(f"最优参数: {grid_search.best_params_}")
        print(f"最佳交叉验证R²: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    except Exception as e:
        print(f"超参数调优失败: {str(e)}")
        return None


def main():
    """主函数"""

    # 读取数据
    df = pd.read_csv('../data/weread_books_cleaned.csv')

    # clustering_analysis(df)

    # 预测分析
    results = predict_analysis(df, target='recommendation')
    print(results)


if __name__ == "__main__":
    main()
