import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from tqdm import tqdm
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 系统已有的中文字体列表
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """数据预处理类"""

    def __init__(self):
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_cols = None

    def fit_transform(self, data, feature_cols):
        """拟合并转换数据"""
        self.feature_cols = feature_cols
        data = np.array(data, dtype=float)

        # 处理无穷值
        data = np.where(np.isinf(data), np.nan, data)

        # 填充缺失值
        data_imputed = self.imputer.fit_transform(data)

        # 标准化
        data_scaled = self.scaler.fit_transform(data_imputed)

        self.is_fitted = True
        return data_scaled

    def transform(self, data):
        """转换新数据"""
        if not self.is_fitted:
            raise ValueError("预处理模型尚未拟合，请先调用fit_transform")

        data = np.array(data, dtype=float)
        data = np.where(np.isinf(data), np.nan, data)
        data_imputed = self.imputer.transform(data)
        data_scaled = self.scaler.transform(data_imputed)
        return data_scaled

    def save(self, path):
        """保存预处理模型"""
        joblib.dump({
            'imputer': self.imputer,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'feature_cols': self.feature_cols  # 确保这一行存在
        }, path)

    @classmethod
    def load(cls, path):
        """加载预处理模型"""
        obj = cls()
        data = joblib.load(path)
        obj.imputer = data['imputer']
        obj.scaler = data['scaler']
        obj.is_fitted = data['is_fitted']
        obj.feature_cols = data['feature_cols']
        return obj


class ModelTrainer:
    """模型训练类"""

    def __init__(self):
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.fig_dir = "short_train_predict_figures"
        os.makedirs(self.fig_dir, exist_ok=True)
        self.feature_cols = None
        self.train_history = {
            'iterations': [],
            'train_rmse': [],
            'val_rmse': []
        }

    def prepare_features(self, data):
        """准备时序特征"""
        print("正在准备时序特征...")

        df = data.copy()

        # 确保目标列是数值类型
        if 'total_power' in df.columns:
            df['total_power'] = pd.to_numeric(df['total_power'], errors='coerce')

        # 提取时间特征
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['month'] = df.index.month
        df['dayofyear'] = df.index.dayofyear
        df['weekofyear'] = df.index.isocalendar().week
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

        # 添加滞后特征
        if 'total_power' in df.columns:
            lags = [1, 2, 3, 4, 24, 48, 96]  # 15分钟, 30分钟, 45分钟, 1小时, 6小时, 12小时, 24小时
            for lag in tqdm(lags, desc="生成滞后特征"):
                df[f'load_lag_{lag}'] = df['total_power'].shift(lag)

            # 滚动统计特征
            windows = [4, 24, 96]  # 1小时, 6小时, 24小时
            for window in tqdm(windows, desc="生成滚动特征"):
                df[f'load_rolling_mean_{window}'] = df['total_power'].rolling(
                    window=window, min_periods=1).mean()
                df[f'load_rolling_std_{window}'] = df['total_power'].rolling(
                    window=window, min_periods=1).std()

        # 填充缺失值
        df = df.ffill().bfill().fillna(0)

        # 确保所有列都是数值类型
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.fillna(0)
        print(f"特征准备完成，总特征数: {len(df.columns)}")
        return df

    def train_model(self, data):
        """训练随机森林模型并可视化训练进度"""
        print("开始训练随机森林模型...")

        # 准备特征
        df_with_features = self.prepare_features(data)

        # 选择特征和目标变量
        self.feature_cols = [col for col in df_with_features.columns if col != 'total_power']
        X = df_with_features[self.feature_cols]
        y = df_with_features['total_power']

        # 划分训练集和验证集（最后7天作为验证集）
        split_date = df_with_features.index.max() - timedelta(days=7)
        train_mask = df_with_features.index <= split_date
        val_mask = df_with_features.index > split_date

        X_train, X_val = X[train_mask], X[val_mask]
        y_train, y_val = y[train_mask], y[val_mask]

        print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")

        # 预处理数据
        X_train_processed = self.preprocessor.fit_transform(X_train.values, self.feature_cols)
        X_val_processed = self.preprocessor.transform(X_val.values)

        # 可视化训练进度 - 逐步增加树的数量
        self._train_with_progress(X_train_processed, y_train, X_val_processed, y_val)

        # 评估最终模型
        y_pred = self.model.predict(X_val_processed)
        metrics = self.calculate_metrics(y_val, y_pred)

        print("✅ 模型训练完成")
        print(f"验证集评估指标:")
        print(f"  - R²: {metrics['R2']:.4f}")
        print(f"  - RMSE: {metrics['RMSE']:.2f}")
        print(f"  - MAE: {metrics['MAE']:.2f}")
        print(f"  - MAPE: {metrics['MAPE']:.2f}%")

        # 绘制训练相关图表
        self.plot_training_progress()
        self.plot_test_predictions(y_val, y_pred, X_val.index)
        self.plot_model_analysis(data)

        # 保存模型和预处理工具
        self.save_model()

        return metrics

    def _train_with_progress(self, X_train, y_train, X_val, y_val):
        """带进度可视化的训练过程"""
        total_estimators = 150
        self.model = RandomForestRegressor(
            n_estimators=1,  # 初始化为1棵树
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        # 使用tqdm显示训练进度
        for i in tqdm(range(total_estimators), desc="训练随机森林"):
            self.model.n_estimators = i + 1
            self.model.fit(X_train, y_train)

            # 计算训练集和验证集分数
            y_train_pred = self.model.predict(X_train)
            y_val_pred = self.model.predict(X_val)

            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

            # 记录训练历史
            self.train_history['iterations'].append(i + 1)
            self.train_history['train_rmse'].append(train_rmse)
            self.train_history['val_rmse'].append(val_rmse)

    def calculate_metrics(self, y_true, y_pred):
        """计算评估指标"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        # 避免除零错误
        y_true_safe = np.clip(np.abs(y_true), 1e-10, None)
        mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

        r2 = r2_score(y_true, y_pred)

        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }

    def plot_training_progress(self):
        """绘制训练进度曲线"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_history['iterations'], self.train_history['train_rmse'],
                 label='训练集RMSE', color='blue', alpha=0.7)
        plt.plot(self.train_history['iterations'], self.train_history['val_rmse'],
                 label='验证集RMSE', color='red', alpha=0.7)
        plt.title('模型训练进度', fontsize=14, fontweight='bold')
        plt.xlabel('决策树数量')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 标记最低验证误差点
        min_val_idx = np.argmin(self.train_history['val_rmse'])
        min_val_rmse = self.train_history['val_rmse'][min_val_idx]
        min_val_iter = self.train_history['iterations'][min_val_idx]
        plt.scatter(min_val_iter, min_val_rmse, color='green', s=100, zorder=5)
        plt.annotate(f'最低: {min_val_rmse:.2f}',
                     (min_val_iter, min_val_rmse),
                     xytext=(10, 10), textcoords='offset points',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green"))

        plt.tight_layout()
        plt.savefig(f'{self.fig_dir}/training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存训练进度图: {self.fig_dir}/training_progress.png")

    def plot_test_predictions(self, y_true, y_pred, dates):
        """绘制验证集预测效果"""
        plt.figure(figsize=(15, 10))

        # 绘制整体对比
        plt.subplot(2, 1, 1)
        plt.plot(dates, y_true, label='真实值', alpha=0.7, linewidth=1)
        plt.plot(dates, y_pred, label='预测值', alpha=0.7, linewidth=1)
        plt.title('验证集负荷预测对比', fontsize=14, fontweight='bold')
        plt.ylabel('负荷值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # 绘制散点图
        plt.subplot(2, 1, 2)
        plt.scatter(y_true, y_pred, alpha=0.6, s=20)
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title('预测值 vs 真实值', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # 添加R²值
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.tight_layout()
        plt.savefig(f'{self.fig_dir}/validation_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存验证集预测效果图: {self.fig_dir}/validation_predictions.png")

    def plot_model_analysis(self, data):
        """绘制模型分析图表"""
        print("\n正在绘制模型分析图表...")

        # 特征重要性分析
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)

            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=feature_importance, palette='viridis')
            plt.title('随机森林特征重要性 (Top 15)', fontsize=14, fontweight='bold')
            plt.xlabel('特征重要性')
            plt.ylabel('特征名称')
            plt.tight_layout()
            plt.savefig(f'{self.fig_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ 已保存特征重要性图: {self.fig_dir}/feature_importance.png")

        # 负荷时间序列分析
        plt.figure(figsize=(15, 10))

        # 原始负荷数据
        plt.subplot(2, 2, 1)
        plt.plot(data.index, data['total_power'], linewidth=0.5, alpha=0.7)
        plt.title('历史负荷数据', fontsize=12)
        plt.ylabel('负荷值')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # 日负荷模式
        plt.subplot(2, 2, 2)
        daily_pattern = data.groupby(data.index.hour)['total_power'].mean()
        plt.plot(daily_pattern.index, daily_pattern.values, marker='o', color='C1')
        plt.title('典型日负荷曲线', fontsize=12)
        plt.xlabel('小时')
        plt.ylabel('平均负荷')
        plt.grid(True, alpha=0.3)

        # 周负荷模式
        plt.subplot(2, 2, 3)
        weekly_pattern = data.groupby(data.index.dayofweek)['total_power'].mean()
        days = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        sns.barplot(x=list(range(7)), y=weekly_pattern.values, palette='viridis')
        plt.title('周负荷模式', fontsize=12)
        plt.xlabel('星期')
        plt.ylabel('平均负荷')
        plt.xticks(range(7), days)
        plt.grid(True, alpha=0.3)

        # 月负荷模式
        plt.subplot(2, 2, 4)
        monthly_pattern = data.groupby(data.index.month)['total_power'].mean()
        sns.barplot(x=list(range(1, 13)), y=monthly_pattern.values, palette='viridis')
        plt.title('月负荷模式', fontsize=12)
        plt.xlabel('月份')
        plt.ylabel('平均负荷')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.fig_dir}/load_pattern_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存负荷模式分析图: {self.fig_dir}/load_pattern_analysis.png")

    def save_model(self):
        """保存模型和预处理工具"""
        joblib.dump(self.model, 'random_forest_model.pkl')
        self.preprocessor.save('data_preprocessor.pkl')
        print(f"✅ 模型已保存到: random_forest_model.pkl")
        print(f"✅ 预处理工具已保存到: data_preprocessor.pkl")


def main():
    """主函数：训练随机森林模型"""
    print("=" * 80)
    print("RandomForest - 负荷预测模型训练")
    print("=" * 80)

    try:
        # 加载数据
        print("正在加载数据...")
        data = pd.read_csv('load_weather_data_15min.csv', index_col=0, parse_dates=True)
        print(f"✅ 数据加载成功，形状: {data.shape}")

        # 检查目标列是否存在
        if 'total_power' not in data.columns:
            print("❌ 数据中未找到 'total_power' 列")
            return

        # 显示数据基本信息
        print(f"\n数据基本信息:")
        print(f"时间范围: {data.index.min()} 到 {data.index.max()}")
        print(f"数据频率: {pd.infer_freq(data.index)}")
        print(f"总记录数: {len(data)}")
        print(f"特征数量: {len(data.columns)}")

        # 初始化训练器
        trainer = ModelTrainer()

        # 训练模型
        metrics = trainer.train_model(data)

        # 输出总结
        print("\n" + "=" * 80)
        print("模型训练完成总结")
        print("=" * 80)
        print(f"📊 模型性能:")
        print(f"   - R²: {metrics['R2']:.4f}")
        print(f"   - RMSE: {metrics['RMSE']:.2f}")
        print(f"   - MAE: {metrics['MAE']:.2f}")
        print(f"   - MAPE: {metrics['MAPE']:.2f}%")

        print(f"\n📈 可视化文件保存目录: {trainer.fig_dir}")

        return metrics

    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()