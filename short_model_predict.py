import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm
import os
from short_model_train import DataPreprocessor

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
warnings.filterwarnings('ignore')


class LoadPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_cols = None
        self.preprocessor_data = None
        # ========== 添加图表保存目录（关键） ==========
        self.fig_dir = "short_train_predict_figures"  # 与训练代码保持一致的目录名
        os.makedirs(self.fig_dir, exist_ok=True)  # 确保目录存在，不存在则创建

    def load_model(self, model_path='random_forest_model.pkl',
                   preprocessor_path='data_preprocessor.pkl'):
        """加载训练好的模型和预处理工具（修复版）"""
        print("正在加载模型...")

        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"预处理工具文件不存在: {preprocessor_path}")

        # 1. 加载模型和预处理数据（字典格式）
        self.model = joblib.load(model_path)
        preprocessor_data = joblib.load(
            preprocessor_path)  # 这是保存的字典：{'imputer': ..., 'scaler': ..., 'feature_cols': ...}

        # 2. 实例化 DataPreprocessor 类（关键：之前缺少这一步）
        self.preprocessor = DataPreprocessor()

        # 3. 将加载的预处理数据赋值给 self.preprocessor
        self.preprocessor.imputer = preprocessor_data['imputer']
        self.preprocessor.scaler = preprocessor_data['scaler']
        self.preprocessor.is_fitted = preprocessor_data['is_fitted']
        self.preprocessor.feature_cols = preprocessor_data.get('feature_cols')

        # 4. 检查特征列是否存在
        if not self.preprocessor.feature_cols:
            raise AttributeError("预处理工具中未找到特征列信息，请重新训练模型")

        self.feature_cols = self.preprocessor.feature_cols  # 同步特征列信息
        print(f"✅ 模型加载成功，特征列数量: {len(self.feature_cols)}")
        return self
    def prepare_features(self, data):
        """准备时序特征（与训练时保持一致）"""
        print("正在准备预测特征...")

        df = data.copy()

        # 提取时间特征
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['month'] = df.index.month
        df['dayofyear'] = df.index.dayofyear
        df['weekofyear'] = df.index.isocalendar().week
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

        # 添加滞后特征（使用历史数据）
        if 'total_power' in df.columns:
            lags = [1, 2, 3, 4, 24, 48, 96]
            for lag in lags:
                df[f'load_lag_{lag}'] = df['total_power'].shift(lag)

            # 滚动统计特征
            windows = [4, 24, 96]
            for window in windows:
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
        return df

    def predict_future_10days(self, data):
        """预测未来10天负荷"""
        print("\n开始预测未来10天负荷...")

        # 准备历史数据特征
        historical_data = self.prepare_features(data)

        # 生成未来10天的日期时间索引（15分钟间隔）
        last_date = historical_data.index.max()
        future_dates = pd.date_range(
            start=last_date + timedelta(minutes=15),
            periods=10 * 24 * 4,  # 10天 * 24小时 * 4(15分钟间隔)
            freq='15min'
        )

        print(f"预测时间范围: {future_dates.min()} 到 {future_dates.max()}")

        # 创建未来数据框
        future_df = pd.DataFrame(index=future_dates)

        # 复制历史数据的最后一行作为基础
        last_row = historical_data.iloc[-1:].copy()

        predictions = []

        # 逐步预测，带进度条
        for i, current_date in tqdm(enumerate(future_dates),
                                    total=len(future_dates),
                                    desc="预测进度"):
            # 更新当前时间特征
            current_data = last_row.copy()
            current_data.index = [current_date]

            current_data['hour'] = current_date.hour
            current_data['dayofweek'] = current_date.dayofweek
            current_data['month'] = current_date.month
            current_data['dayofyear'] = current_date.dayofyear
            current_data['weekofyear'] = current_date.isocalendar().week
            current_data['is_weekend'] = 1 if current_date.dayofweek >= 5 else 0

            # 使用最新的预测值更新滞后特征
            if i > 0:
                for lag in [1, 2, 3, 4, 24, 48, 96]:
                    if f'load_lag_{lag}' in current_data.columns and i >= lag:
                        current_data[f'load_lag_{lag}'] = predictions[i - lag]

            # 更新滚动特征
            if i > 0:
                window_sizes = [4, 24, 96]
                for window in window_sizes:
                    if i >= window:
                        # 使用最近window个预测值计算均值和标准差
                        recent_preds = predictions[i - window:i]
                        current_data[f'load_rolling_mean_{window}'] = np.mean(recent_preds)
                        current_data[f'load_rolling_std_{window}'] = np.std(recent_preds)

            # 确保所有特征都存在
            for col in self.feature_cols:
                if col not in current_data.columns:
                    current_data[col] = 0

            # 确保所有列都是数值类型
            for col in self.feature_cols:
                current_data[col] = pd.to_numeric(current_data[col], errors='coerce')
            current_data = current_data.fillna(0)

            # 选择特征并预处理
            X_future = current_data[self.feature_cols]
            X_future_processed = self.preprocessor.transform(X_future.values.reshape(1, -1))

            # 预测
            pred = self.model.predict(X_future_processed)[0]
            predictions.append(pred)

            # 更新最后一行用于下一次预测
            last_row = current_data.copy()
            last_row['total_power'] = pred

        # 创建预测结果DataFrame
        future_predictions = pd.DataFrame({
            'timestamp': future_dates,
            'predicted_load': predictions
        })
        future_predictions.set_index('timestamp', inplace=True)

        print(f"✅ 未来10天负荷预测完成，共 {len(predictions)} 个预测点")
        return future_predictions

    def analyze_prediction_results(self, future_predictions):
        """分析预测结果"""
        print("\n正在分析预测结果...")

        # 基本统计
        print("预测结果统计:")
        print(f"  预测点数: {len(future_predictions)}")
        print(f"  平均负荷: {future_predictions['predicted_load'].mean():.2f}")
        print(f"  最大负荷: {future_predictions['predicted_load'].max():.2f}")
        print(f"  最小负荷: {future_predictions['predicted_load'].min():.2f}")
        print(f"  负荷标准差: {future_predictions['predicted_load'].std():.2f}")

        # 按天分析
        daily_stats = future_predictions.groupby(future_predictions.index.date).agg({
            'predicted_load': ['mean', 'max', 'min', 'std']
        })
        daily_stats.columns = ['日均负荷', '日最大负荷', '日最小负荷', '日负荷标准差']

        print("\n每日负荷统计:")
        print(daily_stats.round(2))

        # 绘制预测结果
        self.plot_future_predictions(future_predictions)

        return daily_stats

    def plot_future_predictions(self, future_predictions):
        """绘制未来10天预测结果（美化增强版）"""
        # 设置整体风格
        plt.style.use('seaborn-v0_8-whitegrid')

        # 创建画布和子图，增加hspace调整间距
        fig, axes = plt.subplots(4, 1, figsize=(16, 20))
        fig.subplots_adjust(hspace=0.4)
        fig.suptitle('电力负荷10天预测分析报告', fontsize=20, fontweight='bold', y=0.99)

        # 1. 整体预测趋势图
        ax1 = axes[0]
        future_predictions['predicted_load'].plot(
            ax=ax1, linewidth=2, alpha=0.8, color='#2c7fb8'
        )

        # 标记每天的最大值和最小值
        daily_max = future_predictions.groupby(future_predictions.index.date)['predicted_load'].idxmax()
        daily_min = future_predictions.groupby(future_predictions.index.date)['predicted_load'].idxmin()

        max_points = future_predictions.loc[daily_max]
        min_points = future_predictions.loc[daily_min]

        ax1.scatter(max_points.index, max_points['predicted_load'],
                    color='#e41a1c', s=60, label='日最大值', zorder=5, edgecolors='black')
        ax1.scatter(min_points.index, min_points['predicted_load'],
                    color='#4daf4a', s=60, label='日最小值', zorder=5, edgecolors='black')

        # 添加网格和标题
        ax1.set_title('未来10天负荷预测趋势', fontsize=16, fontweight='bold', pad=15)
        ax1.set_ylabel('负荷值 (MW)', fontsize=12)
        ax1.legend(fontsize=11, loc='upper left')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_xlim(future_predictions.index.min(), future_predictions.index.max())

        # 2. 每日负荷曲线对比
        ax2 = axes[1]
        days = future_predictions.index.normalize().unique()
        # 使用更美观的渐变色
        colors = plt.cm.Set3(np.linspace(0, 1, len(days)))

        for i, day in enumerate(days):
            day_data = future_predictions[future_predictions.index.normalize() == day]
            hours = day_data.index.hour + day_data.index.minute / 60
            ax2.plot(hours, day_data['predicted_load'],
                     color=colors[i], alpha=0.8, linewidth=2,
                     label=day.strftime('%m-%d (%a)'))  # 显示星期几

        ax2.set_title('每日负荷曲线对比', fontsize=16, fontweight='bold', pad=15)
        ax2.set_xlabel('一天中的小时', fontsize=12)
        ax2.set_ylabel('负荷值 (MW)', fontsize=12)
        ax2.set_xlim(0, 24)
        ax2.set_xticks(range(0, 25, 3))  # 每3小时一个刻度
        ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10,
                   title='日期(星期)', title_fontsize=11)
        ax2.grid(True, linestyle='--', alpha=0.7)

        # 3. 负荷分布直方图和核密度图
        ax3 = axes[2]
        sns.histplot(future_predictions['predicted_load'], bins=30,
                     kde=True, alpha=0.7, edgecolor='black',
                     color='#fdae61', ax=ax3)

        # 添加统计线
        mean_val = future_predictions['predicted_load'].mean()
        median_val = future_predictions['predicted_load'].median()
        max_val = future_predictions['predicted_load'].max()

        ax3.axvline(mean_val, color='#e41a1c', linestyle='--',
                    label=f'平均值: {mean_val:.2f}', linewidth=2)
        ax3.axvline(median_val, color='#984ea3', linestyle='-.',
                    label=f'中位数: {median_val:.2f}', linewidth=2)

        ax3.set_title('预测负荷分布特征', fontsize=16, fontweight='bold', pad=15)
        ax3.set_xlabel('负荷值 (MW)', fontsize=12)
        ax3.set_ylabel('频次', fontsize=12)
        ax3.legend(fontsize=11)
        ax3.grid(True, linestyle='--', alpha=0.7, axis='y')

        # 4. 每日统计指标对比
        ax4 = axes[3]
        daily_stats = future_predictions.groupby(future_predictions.index.date)['predicted_load'].agg(
            ['mean', 'max', 'min'])
        daily_stats.columns = ['日均负荷', '日最大负荷', '日最小负荷']

        # 使用堆叠柱状图展示
        daily_stats.plot(kind='bar', ax=ax4, width=0.8, alpha=0.8,
                         color=['#377eb8', '#e41a1c', '#4daf4a'])

        ax4.set_title('每日负荷统计指标对比', fontsize=16, fontweight='bold', pad=15)
        ax4.set_xlabel('日期', fontsize=12)
        ax4.set_ylabel('负荷值 (MW)', fontsize=12)
        ax4.set_xticklabels([idx.strftime('%m-%d') for idx in daily_stats.index], rotation=45)
        ax4.legend(fontsize=11, loc='upper left')
        ax4.grid(True, linestyle='--', alpha=0.7, axis='y')

        # 调整布局并保存
        plt.tight_layout()
        # 确保中文显示正常
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        # 保存高分辨率图片
        save_path = f'{self.fig_dir}/future_10days_predictions_enhanced.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存美化版预测图: {save_path}")


def main():
    """主函数：使用训练好的模型预测未来10天负荷"""
    print("=" * 80)
    print("RandomForest - 未来10天负荷预测")
    print("=" * 80)

    try:
        # 加载历史数据
        print("正在加载历史数据...")
        data = pd.read_csv('load_weather_data_15min.csv', index_col=0, parse_dates=True)
        print(f"✅ 数据加载成功，形状: {data.shape}")

        # 检查目标列是否存在
        if 'total_power' not in data.columns:
            print("❌ 数据中未找到 'total_power' 列")
            return

        # 初始化预测器并加载模型
        predictor = LoadPredictor()
        predictor.load_model()

        # 预测未来10天
        future_predictions = predictor.predict_future_10days(data)

        # 分析预测结果
        daily_stats = predictor.analyze_prediction_results(future_predictions)

        # 保存预测结果
        future_predictions.to_csv('future_10days_load_predictions.csv')
        print(f"✅ 预测结果已保存到: future_10days_load_predictions.csv")

        # 输出总结
        print("\n" + "=" * 80)
        print("预测任务完成总结")
        print("=" * 80)
        print(f"🔮 预测结果:")
        print(f"   - 预测时长: 10天")
        print(f"   - 时间间隔: 15分钟")
        print(f"   - 总预测点: {len(future_predictions)}")
        print(f"   - 平均负荷: {future_predictions['predicted_load'].mean():.2f}")
        print(f"   - 最大负荷: {future_predictions['predicted_load'].max():.2f}")
        print(f"   - 最小负荷: {future_predictions['predicted_load'].min():.2f}")

        print(f"\n📈 可视化文件:")
        print(f"   - 未来10天预测: {predictor.fig_dir}/future_10days_predictions.png")

        return future_predictions

    except Exception as e:
        print(f"❌ 预测过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()