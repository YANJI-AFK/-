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
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


class MidTermDataPreprocessor:
    """中期预测数据预处理类"""

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
            'feature_cols': self.feature_cols
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


class MidTermModelTrainer:
    """中期负荷预测模型训练类"""

    def __init__(self):
        self.model = None
        self.preprocessor = MidTermDataPreprocessor()
        self.fig_dir = "mid_term_train_figures"
        os.makedirs(self.fig_dir, exist_ok=True)
        self.feature_cols = None
        self.train_history = {
            'iterations': [],
            'train_rmse': [],
            'val_rmse': []
        }
        self.industries = ['商业', '大工业用电', '普通工业', '非普工业']
        self.target_types = ['max', 'min']

    def prepare_midterm_features(self, data):
        """准备中期预测特征"""
        print("正在准备中期预测特征...")

        df = data.copy()

        # 确保日期格式
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

        # 基础时间特征
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek
        df['day_of_year'] = df.index.dayofyear
        df['week_of_year'] = df.index.isocalendar().week
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)

        # 季节特征
        df['is_spring'] = ((df['month'] >= 3) & (df['month'] <= 5)).astype(int)
        df['is_summer'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)
        df['is_autumn'] = ((df['month'] >= 9) & (df['month'] <= 11)).astype(int)
        df['is_winter'] = ((df['month'] <= 2) | (df['month'] == 12)).astype(int)

        # 节假日特征（中国主要节假日，更详细）
        df['day_of_month'] = df.index.day
        df['is_holiday'] = (
                ((df['month'] == 1) & (df['day_of_month'] <= 3)) |  # 元旦
                ((df['month'] == 2) & (df['day_of_month'] >= 10) & (df['day_of_month'] <= 17)) |  # 春节
                ((df['month'] == 4) & (df['day_of_month'] >= 3) & (df['day_of_month'] <= 5)) |  # 清明节
                ((df['month'] == 5) & (df['day_of_month'] >= 1) & (df['day_of_month'] <= 3)) |  # 劳动节
                ((df['month'] == 6) & (df['day_of_month'] >= 12) & (df['day_of_month'] <= 14)) |  # 端午节
                ((df['month'] == 9) & (df['day_of_month'] >= 19) & (df['day_of_month'] <= 21)) |  # 中秋节
                ((df['month'] == 10) & (df['day_of_month'] >= 1) & (df['day_of_month'] <= 7))  # 国庆节
        ).astype(int)
        df['is_holiday_prev'] = df['is_holiday'].shift(1).fillna(0).astype(int)
        df['is_holiday_next'] = df['is_holiday'].shift(-1).fillna(0).astype(int)

        # 周期性特征
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # 为每个行业和负荷类型创建滞后特征
        for industry in self.industries:
            for target_type in self.target_types:
                target_col = f'{industry}_{target_type}_power'
                if target_col in df.columns:
                    # 扩展滞后特征
                    for lag in [7, 14, 21, 30, 60, 90]:  # 增加21天和60天滞后
                        if len(df) > lag:
                            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

                    # 扩展滚动统计特征
                    for window in [7, 14, 30, 60, 90]:  # 增加14天和60天窗口
                        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(
                            window=window, min_periods=1).mean()
                        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(
                            window=window, min_periods=1).std()
                        df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(
                            window=window, min_periods=1).min()
                        df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(
                            window=window, min_periods=1).max()
                        # 增加滚动中位数特征
                        df[f'{target_col}_rolling_median_{window}'] = df[target_col].rolling(
                            window=window, min_periods=1).median()

        # 增加更多统计特征
        for industry in self.industries:
            for target_type in self.target_types:
                target_col = f'{industry}_{target_type}_power'
                if target_col in df.columns:
                    # 年度同比和环比特征
                    df[f'{target_col}_year_growth'] = df[target_col].pct_change(periods=365)
                    df[f'{target_col}_month_growth'] = df[target_col].pct_change(periods=30)
                    df[f'{target_col}_week_growth'] = df[target_col].pct_change(periods=7)

        # 交叉特征：行业间的相关性特征
        if len(self.industries) > 1:
            for i, industry1 in enumerate(self.industries):
                for industry2 in self.industries[i+1:]:
                    for target_type in self.target_types:
                        col1 = f'{industry1}_{target_type}_power'
                        col2 = f'{industry2}_{target_type}_power'
                        if col1 in df.columns and col2 in df.columns:
                            df[f'{industry1}_{industry2}_{target_type}_ratio'] = df[col1] / (df[col2] + 1e-10)

        # 填充缺失值，使用更智能的填充策略
        df = df.fillna(method='ffill', limit=7)  # 向前填充最多7天
        df = df.fillna(method='bfill', limit=7)  # 向后填充最多7天
        df = df.fillna(0)  # 剩余缺失值填充0

        # 确保所有列都是数值类型
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.fillna(0)
        print(f"中期特征准备完成，总特征数: {len(df.columns)}")
        return df

    def prepare_targets(self, data):
        """准备多输出目标"""
        targets = []
        target_names = []

        for industry in self.industries:
            for target_type in self.target_types:
                target_col = f'{industry}_{target_type}_power'
                if target_col in data.columns:
                    targets.append(data[target_col])
                    target_names.append(target_col)

        if not targets:
            raise ValueError("没有找到任何目标列，请检查数据")

        y = pd.concat(targets, axis=1)
        y.columns = target_names
        return y, target_names

    def train_midterm_model(self, data):
        """训练中期负荷预测模型"""
        print("开始训练中期负荷预测模型...")

        # 准备特征
        df_with_features = self.prepare_midterm_features(data)

        # 准备目标变量（多输出）
        y, target_names = self.prepare_targets(df_with_features)

        # 选择特征列（排除目标列）
        self.feature_cols = [col for col in df_with_features.columns if col not in target_names]
        X = df_with_features[self.feature_cols]

        print(f"特征数量: {len(self.feature_cols)}")
        print(f"目标数量: {len(target_names)}")
        print(f"目标列: {target_names}")

        # 划分训练集和验证集（最后3个月作为验证集）
        split_date = df_with_features.index.max() - timedelta(days=90)
        train_mask = df_with_features.index <= split_date
        val_mask = df_with_features.index > split_date

        X_train, X_val = X[train_mask], X[val_mask]
        y_train, y_val = y[train_mask], y[val_mask]

        print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")
        print(f"训练时间范围: {X_train.index.min()} 到 {X_train.index.max()}")
        print(f"验证时间范围: {X_val.index.min()} 到 {X_val.index.max()}")

        # 预处理数据
        X_train_processed = self.preprocessor.fit_transform(X_train.values, self.feature_cols)
        X_val_processed = self.preprocessor.transform(X_val.values)

        # 为每个目标训练单独的模型（中期预测通常需要更精确的模型）
        self.models = {}
        all_metrics = {}

        for i, target_name in enumerate(tqdm(target_names, desc="训练各行业模型")):
            print(f"\n训练模型: {target_name}")

            # 优化后的随机森林参数
            model = RandomForestRegressor(
                n_estimators=300,  # 增加树的数量
                max_depth=25,       # 增加树的深度
                min_samples_split=2, # 减少分裂所需的最小样本数
                min_samples_leaf=1,  # 减少叶节点所需的最小样本数
                max_features='sqrt',  # 使用平方根特征数
                random_state=42,
                n_jobs=-1,
                verbose=0
            )

            model.fit(X_train_processed, y_train[target_name])
            self.models[target_name] = model

            # 评估模型
            y_pred = model.predict(X_val_processed)
            metrics = self.calculate_metrics(y_val[target_name], y_pred)
            all_metrics[target_name] = metrics

            print(f"  {target_name} - R²: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.2f}, MAPE: {metrics['MAPE']:.2f}%")

        # 计算总体指标
        overall_metrics = self.calculate_overall_metrics(all_metrics)

        print("\n✅ 中期模型训练完成")
        print(f"总体评估指标:")
        print(f"  - 平均R²: {overall_metrics['mean_R2']:.4f}")
        print(f"  - 平均RMSE: {overall_metrics['mean_RMSE']:.2f}")
        print(f"  - 平均MAE: {overall_metrics['mean_MAE']:.2f}")
        print(f"  - 平均MAPE: {overall_metrics['mean_MAPE']:.2f}%")

        # 绘制分析图表
        self.plot_midterm_analysis(data, y_val, target_names)
        self.plot_industry_comparison(all_metrics)
        self.plot_feature_importance(X_train, X.columns)

        # 保存模型
        self.save_models()

        return overall_metrics, all_metrics

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

    def calculate_overall_metrics(self, all_metrics):
        """计算总体指标"""
        mean_r2 = np.mean([metrics['R2'] for metrics in all_metrics.values()])
        mean_rmse = np.mean([metrics['RMSE'] for metrics in all_metrics.values()])
        mean_mae = np.mean([metrics['MAE'] for metrics in all_metrics.values()])
        mean_mape = np.mean([metrics['MAPE'] for metrics in all_metrics.values()])

        return {
            'mean_R2': mean_r2,
            'mean_RMSE': mean_rmse,
            'mean_MAE': mean_mae,
            'mean_MAPE': mean_mape
        }

    def plot_midterm_analysis(self, data, y_val, target_names):
        """绘制中期预测分析图表"""
        print("\n正在绘制中期预测分析图表...")

        # 1. 各行业负荷趋势图
        plt.figure(figsize=(15, 12))

        # 最大负荷趋势
        plt.subplot(2, 1, 1)
        for industry in self.industries:
            target_col = f'{industry}_max_power'
            if target_col in data.columns:
                plt.plot(data.index, data[target_col], label=f'{industry}最大负荷', alpha=0.7)
        plt.title('各行业最大负荷趋势', fontsize=14, fontweight='bold')
        plt.ylabel('负荷值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # 最小负荷趋势
        plt.subplot(2, 1, 2)
        for industry in self.industries:
            target_col = f'{industry}_min_power'
            if target_col in data.columns:
                plt.plot(data.index, data[target_col], label=f'{industry}最小负荷', alpha=0.7)
        plt.title('各行业最小负荷趋势', fontsize=14, fontweight='bold')
        plt.ylabel('负荷值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(f'{self.fig_dir}/industry_load_trends.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 季节性分析
        plt.figure(figsize=(15, 10))

        for i, industry in enumerate(self.industries[:4]):  # 最多显示4个行业
            plt.subplot(2, 2, i + 1)
            target_col = f'{industry}_max_power'
            if target_col in data.columns:
                monthly_avg = data.groupby(data.index.month)[target_col].mean()
                plt.plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2)
                plt.title(f'{industry} - 月平均负荷')
                plt.xlabel('月份')
                plt.ylabel('平均负荷')
                plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.fig_dir}/seasonal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 已保存中期分析图表到: {self.fig_dir}")

    def plot_industry_comparison(self, all_metrics):
        """绘制各行业性能比较图"""
        plt.figure(figsize=(15, 10))

        # R²比较
        plt.subplot(2, 2, 1)
        r2_scores = [metrics['R2'] for metrics in all_metrics.values()]
        targets = list(all_metrics.keys())
        bars = plt.bar(range(len(targets)), r2_scores, color='skyblue', alpha=0.7)
        plt.title('各目标R²比较', fontsize=12, fontweight='bold')
        plt.ylabel('R² Score')
        plt.xticks(range(len(targets)), targets, rotation=45)

        # 在柱子上添加数值
        for bar, score in zip(bars, r2_scores):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{score:.3f}', ha='center', va='bottom', fontsize=8)

        # RMSE比较
        plt.subplot(2, 2, 2)
        rmse_scores = [metrics['RMSE'] for metrics in all_metrics.values()]
        bars = plt.bar(range(len(targets)), rmse_scores, color='lightcoral', alpha=0.7)
        plt.title('各目标RMSE比较', fontsize=12, fontweight='bold')
        plt.ylabel('RMSE')
        plt.xticks(range(len(targets)), targets, rotation=45)

        for bar, score in zip(bars, rmse_scores):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(rmse_scores) * 0.01,
                     f'{score:.1f}', ha='center', va='bottom', fontsize=8)

        # MAE比较
        plt.subplot(2, 2, 3)
        mae_scores = [metrics['MAE'] for metrics in all_metrics.values()]
        bars = plt.bar(range(len(targets)), mae_scores, color='lightgreen', alpha=0.7)
        plt.title('各目标MAE比较', fontsize=12, fontweight='bold')
        plt.ylabel('MAE')
        plt.xticks(range(len(targets)), targets, rotation=45)

        for bar, score in zip(bars, mae_scores):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(mae_scores) * 0.01,
                     f'{score:.1f}', ha='center', va='bottom', fontsize=8)

        # MAPE比较
        plt.subplot(2, 2, 4)
        mape_scores = [metrics['MAPE'] for metrics in all_metrics.values()]
        bars = plt.bar(range(len(targets)), mape_scores, color='gold', alpha=0.7)
        plt.title('各目标MAPE比较', fontsize=12, fontweight='bold')
        plt.ylabel('MAPE (%)')
        plt.xticks(range(len(targets)), targets, rotation=45)

        for bar, score in zip(bars, mape_scores):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(mape_scores) * 0.01,
                     f'{score:.1f}%', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(f'{self.fig_dir}/industry_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存行业性能比较图: {self.fig_dir}/industry_performance_comparison.png")

    def plot_feature_importance(self, X_train, feature_names, top_n=20):
        """绘制特征重要性图"""
        print("绘制特征重要性图...")
        
        # 计算平均特征重要性（所有模型）
        feature_importances = np.zeros(X_train.shape[1])
        for model in self.models.values():
            feature_importances += model.feature_importances_
        feature_importances /= len(self.models)
        
        # 获取重要性排序
        indices = np.argsort(feature_importances)[::-1]
        top_indices = indices[:top_n]
        
        # 绘制前N个重要特征
        plt.figure(figsize=(15, 10))
        plt.title('特征重要性（Top {}）'.format(top_n), fontsize=14, fontweight='bold')
        plt.bar(range(top_n), feature_importances[top_indices], color='skyblue', alpha=0.8)
        plt.xticks(range(top_n), [feature_names[i] for i in top_indices], rotation=90, fontsize=10)
        plt.ylabel('重要性分数')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(f'{self.fig_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 已保存特征重要性图到: {self.fig_dir}/feature_importance.png")
        
        # 打印重要特征
        print("\nTop 10 重要特征:")
        for i in range(10):
            print(f"  {i+1}. {feature_names[indices[i]]}: {feature_importances[indices[i]]:.4f}")
    
    def save_models(self):
        """保存模型和预处理工具"""
        # 保存所有模型
        model_dict = {
            'models': self.models,
            'feature_cols': self.feature_cols,
            'industries': self.industries,
            'target_types': self.target_types,
            'save_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        joblib.dump(model_dict, 'mid_random_forest_models.pkl')
        self.preprocessor.save('mid_random_forest_preprocessor.pkl')
        print(f"✅ 中期模型已保存到: mid_random_forest_models.pkl")
        print(f"✅ 预处理工具已保存到: mid_random_forest_preprocessor.pkl")


def main():
    """主函数：训练中期负荷预测模型"""
    print("=" * 80)
    print("中期负荷预测 - 随机森林模型训练")
    print("=" * 80)

    try:
        # 加载数据
        print("正在加载数据...")
        data = pd.read_csv('industry_weather_data_daily.csv')
        print(f"✅ 数据加载成功，形状: {data.shape}")

        # 检查数据列
        print("数据列:", list(data.columns))

        # 显示数据基本信息
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
            print(f"\n数据基本信息:")
            print(f"时间范围: {data.index.min()} 到 {data.index.max()}")
            print(f"总记录数: {len(data)}")
            print(f"特征数量: {len(data.columns)}")

        # 初始化训练器
        trainer = MidTermModelTrainer()

        # 训练模型
        overall_metrics, detailed_metrics = trainer.train_midterm_model(data)

        # 输出总结
        print("\n" + "=" * 80)
        print("中期模型训练完成总结")
        print("=" * 80)
        print(f"📊 总体性能:")
        print(f"   - 平均R²: {overall_metrics['mean_R2']:.4f}")
        print(f"   - 平均RMSE: {overall_metrics['mean_RMSE']:.2f}")
        print(f"   - 平均MAE: {overall_metrics['mean_MAE']:.2f}")
        print(f"   - 平均MAPE: {overall_metrics['mean_MAPE']:.2f}%")

        print(f"\n📈 各行业详细性能:")
        for target, metrics in detailed_metrics.items():
            print(f"   {target}:")
            print(f"     - R²: {metrics['R2']:.4f}")
            print(f"     - RMSE: {metrics['RMSE']:.2f}")
            print(f"     - MAE: {metrics['MAE']:.2f}")
            print(f"     - MAPE: {metrics['MAPE']:.2f}%")

        print(f"\n📁 可视化文件保存目录: {trainer.fig_dir}")

        return overall_metrics, detailed_metrics

    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    main()