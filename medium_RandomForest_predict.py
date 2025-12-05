import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class MidTermDataPreprocessor:
    """中期预测数据预处理类 - 用于预测时加载"""

    def __init__(self):
        self.imputer = None
        self.scaler = None
        self.is_fitted = False
        self.feature_cols = None

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

    def transform(self, data):
        """转换新数据"""
        if not self.is_fitted:
            raise ValueError("预处理模型尚未拟合，请先调用fit_transform")

        data = np.array(data, dtype=float)
        data = np.where(np.isinf(data), np.nan, data)
        data_imputed = self.imputer.transform(data)
        data_scaled = self.scaler.transform(data_imputed)
        return data_scaled


class MidTermPredictor:
    """中期负荷预测类"""

    def __init__(self):
        self.models = None
        self.preprocessor = None
        self.feature_cols = None
        self.industries = ['商业', '大工业用电', '普通工业', '非普工业']
        self.target_types = ['max', 'min']
        self.fig_dir = "mid_term_prediction_figures"
        import os
        os.makedirs(self.fig_dir, exist_ok=True)

    def load_models(self, model_path='mid_random_forest_models.pkl',
                    preprocessor_path='mid_random_forest_preprocessor.pkl'):
        """加载训练好的模型和预处理工具"""
        print("正在加载模型...")
        try:
            # 加载模型
            model_dict = joblib.load(model_path)
            self.models = model_dict['models']
            self.feature_cols = model_dict['feature_cols']
            self.industries = model_dict.get('industries', self.industries)
            self.target_types = model_dict.get('target_types', self.target_types)

            # 加载预处理工具 - 使用正确的方式
            self.preprocessor = MidTermDataPreprocessor.load(preprocessor_path)

            print(f"✅ 模型加载成功")
            print(f"   目标变量数量: {len(self.models)}")
            print(f"   特征数量: {len(self.feature_cols)}")
            return True

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def prepare_future_features(self, historical_data, future_dates):
        """为未来日期准备特征"""
        print("正在准备未来日期特征...")

        # 复制历史数据用于特征工程
        df_historical = historical_data.copy()
        if 'date' in df_historical.columns:
            df_historical['date'] = pd.to_datetime(df_historical['date'])
            df_historical.set_index('date', inplace=True)

        # 创建未来日期的DataFrame
        future_df = pd.DataFrame(index=future_dates)
        future_df['month'] = future_df.index.month
        future_df['day_of_week'] = future_df.index.dayofweek
        future_df['day_of_year'] = future_df.index.dayofyear
        future_df['week_of_year'] = future_df.index.isocalendar().week
        future_df['quarter'] = future_df.index.quarter
        future_df['year'] = future_df.index.year
        future_df['is_weekend'] = (future_df.index.dayofweek >= 5).astype(int)

        # 季节特征
        future_df['is_spring'] = ((future_df['month'] >= 3) & (future_df['month'] <= 5)).astype(int)
        future_df['is_summer'] = ((future_df['month'] >= 6) & (future_df['month'] <= 8)).astype(int)
        future_df['is_autumn'] = ((future_df['month'] >= 9) & (future_df['month'] <= 11)).astype(int)
        future_df['is_winter'] = ((future_df['month'] <= 2) | (future_df['month'] == 12)).astype(int)

        # 节假日特征
        future_df['day_of_month'] = future_df.index.day
        future_df['is_holiday'] = (
                ((future_df['month'] == 1) & (future_df['day_of_month'] <= 3)) |
                ((future_df['month'] == 5) & (future_df['day_of_month'] >= 1) & (future_df['day_of_month'] <= 3)) |
                ((future_df['month'] == 10) & (future_df['day_of_month'] >= 1) & (future_df['day_of_month'] <= 7))
        ).astype(int)

        # 周期性特征
        future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
        future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)
        future_df['day_of_year_sin'] = np.sin(2 * np.pi * future_df['day_of_year'] / 365)
        future_df['day_of_year_cos'] = np.cos(2 * np.pi * future_df['day_of_year'] / 365)
        future_df['day_of_week_sin'] = np.sin(2 * np.pi * future_df['day_of_week'] / 7)
        future_df['day_of_week_cos'] = np.cos(2 * np.pi * future_df['day_of_week'] / 7)

        # 使用历史数据计算滞后特征
        last_date = df_historical.index.max()

        for industry in self.industries:
            for target_type in self.target_types:
                target_col = f'{industry}_{target_type}_power'
                if target_col in df_historical.columns:
                    # 滞后特征 - 使用历史数据的最后值
                    for lag in [7, 14, 30, 90]:
                        if len(df_historical) > lag:
                            # 获取滞后值
                            lag_values = {}
                            for future_date in future_dates:
                                lag_date = future_date - timedelta(days=lag)
                                if lag_date in df_historical.index:
                                    lag_values[future_date] = df_historical.loc[lag_date, target_col]
                                else:
                                    # 如果滞后日期不在历史数据中，使用最近的值
                                    available_dates = df_historical[df_historical.index <= future_date].index
                                    if len(available_dates) > 0:
                                        nearest_date = available_dates[-1]
                                        lag_values[future_date] = df_historical.loc[nearest_date, target_col]
                                    else:
                                        lag_values[future_date] = df_historical[target_col].mean()

                            future_df[f'{target_col}_lag_{lag}'] = future_df.index.map(lag_values)

                    # 滚动统计特征 - 使用历史数据的滚动统计
                    for window in [7, 30, 90]:
                        if len(df_historical) >= window:
                            # 计算历史数据的滚动统计
                            rolling_mean = df_historical[target_col].rolling(window=window).mean().iloc[-1]
                            rolling_std = df_historical[target_col].rolling(window=window).std().iloc[-1]
                            rolling_min = df_historical[target_col].rolling(window=window).min().iloc[-1]
                            rolling_max = df_historical[target_col].rolling(window=window).max().iloc[-1]

                            # 为所有未来日期使用相同的滚动统计值
                            future_df[f'{target_col}_rolling_mean_{window}'] = rolling_mean
                            future_df[f'{target_col}_rolling_std_{window}'] = rolling_std
                            future_df[f'{target_col}_rolling_min_{window}'] = rolling_min
                            future_df[f'{target_col}_rolling_max_{window}'] = rolling_max

        # 年度同比特征 - 使用历史数据
        for industry in self.industries:
            for target_type in self.target_types:
                target_col = f'{industry}_{target_type}_power'
                if target_col in df_historical.columns:
                    # 计算去年的增长率
                    if len(df_historical) > 365:
                        current_year_avg = df_historical[target_col].iloc[-90:].mean()  # 最近3个月平均
                        last_year_avg = df_historical[target_col].iloc[-455:-365].mean()  # 一年前的3个月平均
                        if last_year_avg > 0:
                            growth_rate = (current_year_avg - last_year_avg) / last_year_avg
                        else:
                            growth_rate = 0
                        future_df[f'{target_col}_year_growth'] = growth_rate

        # 交互特征
        max_cols = [f'{industry}_max_power' for industry in self.industries
                    if f'{industry}_max_power' in df_historical.columns]
        if len(max_cols) > 1:
            # 使用历史数据的平均值
            total_max = df_historical[max_cols].sum(axis=1).mean()
            avg_max = df_historical[max_cols].mean(axis=1).mean()
            future_df['total_max_power'] = total_max
            future_df['avg_max_power'] = avg_max

        # 确保所有特征列都存在
        for col in self.feature_cols:
            if col not in future_df.columns:
                # 如果特征不存在，使用历史数据的平均值
                if col in df_historical.columns:
                    future_df[col] = df_historical[col].mean()
                else:
                    future_df[col] = 0

        # 只保留需要的特征列
        future_features = future_df[self.feature_cols]

        # 填充缺失值
        future_features = future_features.fillna(0)

        print(f"✅ 未来特征准备完成，特征数: {len(future_features.columns)}")
        return future_features

    def predict_future(self, historical_data, days=90):
        """预测未来负荷"""
        if not self.models or not self.preprocessor:
            print("❌ 请先加载模型")
            return None

        print(f"开始预测未来 {days} 天负荷...")

        # 生成未来日期
        last_date = historical_data.index.max() if hasattr(historical_data, 'index') else pd.to_datetime(
            historical_data['date']).max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='D')

        # 准备未来特征
        future_features = self.prepare_future_features(historical_data, future_dates)

        # 预处理特征
        try:
            future_processed = self.preprocessor.transform(future_features.values)
        except Exception as e:
            print(f"❌ 特征预处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None

        # 进行预测
        predictions = {}
        for target_name, model in self.models.items():
            try:
                pred = model.predict(future_processed)
                predictions[target_name] = pred
                print(f"✅ {target_name} 预测完成")
            except Exception as e:
                print(f"❌ {target_name} 预测失败: {e}")
                predictions[target_name] = np.zeros(len(future_dates))

        # 创建预测结果DataFrame
        result_df = pd.DataFrame(index=future_dates)
        for target_name, pred_values in predictions.items():
            result_df[target_name] = pred_values

        print(f"✅ 未来 {days} 天负荷预测完成")
        return result_df

    def evaluate_prediction(self, actual_data, predicted_data, last_n_days=90):
        """评估预测效果（如果有实际数据）"""
        print("\n正在评估预测效果...")

        # 获取最近的实际数据用于评估
        evaluation_period = actual_data.index.max() - timedelta(days=last_n_days)
        actual_recent = actual_data[actual_data.index >= evaluation_period]

        # 对齐实际数据和预测数据的时间范围
        common_dates = actual_recent.index.intersection(predicted_data.index)

        if len(common_dates) == 0:
            print("❌ 没有共同的时间范围用于评估")
            return None

        actual_common = actual_recent.loc[common_dates]
        predicted_common = predicted_data.loc[common_dates]

        evaluation_results = {}

        for target_col in self.models.keys():
            if target_col in actual_common.columns and target_col in predicted_common.columns:
                y_true = actual_common[target_col]
                y_pred = predicted_common[target_col]

                # 计算评估指标
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)

                # 安全的MAPE计算
                y_true_safe = np.abs(y_true)
                y_true_safe = np.where(y_true_safe < 1e-10, 1e-10, y_true_safe)
                mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

                evaluation_results[target_col] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE': mape,
                    'R2': r2
                }

        return evaluation_results

    def plot_predictions(self, historical_data, predicted_data, evaluation_results=None):
        """绘制预测结果"""
        print("正在生成预测图表...")

        # 1. 各行业预测趋势图
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        axes = axes.flatten()

        for i, industry in enumerate(self.industries):
            if i >= len(axes):
                break

            ax = axes[i]

            # 最大负荷
            max_col = f'{industry}_max_power'
            if max_col in historical_data.columns and max_col in predicted_data.columns:
                # 历史数据（最近180天）
                hist_start = historical_data.index.max() - timedelta(days=180)
                hist_recent = historical_data[historical_data.index >= hist_start]

                ax.plot(hist_recent.index, hist_recent[max_col],
                        label='历史最大负荷', color='blue', alpha=0.7, linewidth=1)
                ax.plot(predicted_data.index, predicted_data[max_col],
                        label='预测最大负荷', color='red', alpha=0.8, linewidth=2, linestyle='--')

            # 最小负荷
            min_col = f'{industry}_min_power'
            if min_col in historical_data.columns and min_col in predicted_data.columns:
                ax.plot(hist_recent.index, hist_recent[min_col],
                        label='历史最小负荷', color='green', alpha=0.7, linewidth=1)
                ax.plot(predicted_data.index, predicted_data[min_col],
                        label='预测最小负荷', color='orange', alpha=0.8, linewidth=2, linestyle='--')

            ax.set_title(f'{industry}负荷预测', fontweight='bold', fontsize=14)
            ax.set_ylabel('负荷值')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(f'{self.fig_dir}/industry_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 预测效果评估图（如果有评估结果）
        if evaluation_results:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            targets = list(evaluation_results.keys())
            short_names = [f"{t.split('_')[0]}-{t.split('_')[1]}" for t in targets]

            # R²比较
            r2_scores = [results['R2'] for results in evaluation_results.values()]
            bars1 = ax1.bar(range(len(targets)), r2_scores, color='skyblue', alpha=0.7)
            ax1.set_title('各目标预测R²比较', fontweight='bold')
            ax1.set_ylabel('R² Score')
            ax1.set_xticks(range(len(targets)))
            ax1.set_xticklabels(short_names, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3, axis='y')

            for bar, score in zip(bars1, r2_scores):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{score:.3f}', ha='center', va='bottom', fontsize=8)

            # RMSE比较
            rmse_scores = [results['RMSE'] for results in evaluation_results.values()]
            bars2 = ax2.bar(range(len(targets)), rmse_scores, color='lightcoral', alpha=0.7)
            ax2.set_title('各目标预测RMSE比较', fontweight='bold')
            ax2.set_ylabel('RMSE')
            ax2.set_xticks(range(len(targets)))
            ax2.set_xticklabels(short_names, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3, axis='y')

            for bar, score in zip(bars2, rmse_scores):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(rmse_scores) * 0.01,
                         f'{score:.1f}', ha='center', va='bottom', fontsize=8)

            # MAE比较
            mae_scores = [results['MAE'] for results in evaluation_results.values()]
            bars3 = ax3.bar(range(len(targets)), mae_scores, color='lightgreen', alpha=0.7)
            ax3.set_title('各目标预测MAE比较', fontweight='bold')
            ax3.set_ylabel('MAE')
            ax3.set_xticks(range(len(targets)))
            ax3.set_xticklabels(short_names, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3, axis='y')

            for bar, score in zip(bars3, mae_scores):
                ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(mae_scores) * 0.01,
                         f'{score:.1f}', ha='center', va='bottom', fontsize=8)

            # MAPE比较
            mape_scores = [results['MAPE'] for results in evaluation_results.values()]
            bars4 = ax4.bar(range(len(targets)), mape_scores, color='gold', alpha=0.7)
            ax4.set_title('各目标预测MAPE比较', fontweight='bold')
            ax4.set_ylabel('MAPE (%)')
            ax4.set_xticks(range(len(targets)))
            ax4.set_xticklabels(short_names, rotation=45, ha='right')
            ax4.grid(True, alpha=0.3, axis='y')

            for bar, score in zip(bars4, mape_scores):
                ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(mape_scores) * 0.01,
                         f'{score:.1f}%', ha='center', va='bottom', fontsize=8)

            plt.tight_layout()
            plt.savefig(f'{self.fig_dir}/prediction_evaluation.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 3. 预测汇总表格
        summary_data = []
        for industry in self.industries:
            for target_type in self.target_types:
                target_col = f'{industry}_{target_type}_power'
                if target_col in predicted_data.columns:
                    pred_values = predicted_data[target_col]
                    summary_data.append({
                        '行业': industry,
                        '负荷类型': '最大值' if target_type == 'max' else '最小值',
                        '预测平均值': pred_values.mean(),
                        '预测最大值': pred_values.max(),
                        '预测最小值': pred_values.min(),
                        '预测标准差': pred_values.std()
                    })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f'{self.fig_dir}/prediction_summary.csv', index=False, encoding='utf-8-sig')

        print(f"✅ 预测图表已保存到: {self.fig_dir}")
        return summary_df


def main():
    """主函数：进行未来3个月负荷预测"""
    print("=" * 80)
    print("中期负荷预测 - 未来3个月预测")
    print("=" * 80)

    try:
        # 加载历史数据
        print("正在加载历史数据...")
        historical_data = pd.read_csv('industry_weather_data_daily.csv')
        if 'date' in historical_data.columns:
            historical_data['date'] = pd.to_datetime(historical_data['date'])
            historical_data.set_index('date', inplace=True)
        print(f"✅ 历史数据加载成功，形状: {historical_data.shape}")

        # 初始化预测器
        predictor = MidTermPredictor()

        # 加载模型
        if not predictor.load_models():
            return None, None

        # 进行未来3个月预测
        prediction_days = 90  # 3个月
        predictions = predictor.predict_future(historical_data, days=prediction_days)

        if predictions is None:
            print("❌ 预测失败")
            return None, None

        # 保存预测结果
        predictions.to_csv('future_3month_predictions.csv', encoding='utf-8-sig')
        print("✅ 预测结果已保存到: future_3month_predictions.csv")

        # 评估预测效果（如果有最近的实际数据）
        evaluation_results = None
        # 这里可以取消注释来评估预测效果
        # evaluation_results = predictor.evaluate_prediction(historical_data, predictions)

        # 生成预测图表和汇总
        summary_df = predictor.plot_predictions(historical_data, predictions, evaluation_results)

        # 输出预测总结
        print("\n" + "=" * 80)
        print("未来3个月负荷预测总结")
        print("=" * 80)

        print(f"\n📅 预测时间范围:")
        print(f"   开始日期: {predictions.index.min()}")
        print(f"   结束日期: {predictions.index.max()}")
        print(f"   总预测天数: {len(predictions)}")

        print(f"\n📊 各行业预测汇总:")
        for industry in predictor.industries:
            print(f"\n   {industry}:")
            max_col = f'{industry}_max_power'
            min_col = f'{industry}_min_power'

            if max_col in predictions.columns:
                max_vals = predictions[max_col]
                print(f"     最大负荷 - 平均: {max_vals.mean():.2f}, 范围: {max_vals.min():.2f} ~ {max_vals.max():.2f}")

            if min_col in predictions.columns:
                min_vals = predictions[min_col]
                print(f"     最小负荷 - 平均: {min_vals.mean():.2f}, 范围: {min_vals.min():.2f} ~ {min_vals.max():.2f}")

        print(f"\n📈 预测趋势分析:")
        # 分析季节性趋势
        predictions['month'] = predictions.index.month
        monthly_trend = predictions.groupby('month').mean()

        for industry in predictor.industries:
            max_col = f'{industry}_max_power'
            if max_col in monthly_trend.columns:
                peak_month = monthly_trend[max_col].idxmax()
                low_month = monthly_trend[max_col].idxmin()
                print(f"   {industry}最大负荷: 峰值在{peak_month}月, 谷值在{low_month}月")

        print(f"\n💾 输出文件:")
        print(f"   - 预测数据: future_3month_predictions.csv")
        print(f"   - 预测图表: {predictor.fig_dir}/")
        print(f"   - 预测汇总: {predictor.fig_dir}/prediction_summary.csv")

        return predictions, summary_df

    except Exception as e:
        print(f"❌ 预测过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    predictions, summary = main()