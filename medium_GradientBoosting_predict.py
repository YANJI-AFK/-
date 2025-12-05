import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from tqdm import tqdm


class GradientBoostingPredictor:
    def __init__(self, model_path='gradient_boosting_deep_trained.pkl'):
        self.model_info = self.load_model(model_path)
        self.model = self.model_info['model']
        self.scaler = self.model_info['scaler']
        self.feature_columns = self.model_info['feature_columns']
        self.target_columns = self.model_info['target_columns']

        print("✅ 已加载模型信息:")
        print(f"   - 模型类型: GradientBoosting")
        print(f"   - 特征数量: {len(self.feature_columns)}")
        print(f"   - 目标数量: {len(self.target_columns)}")
        print(f"   - 训练时间: {self.model_info.get('train_time', '未知')}")

    def load_model(self, model_path):
        """加载完整模型信息"""
        print(f"🔄 正在加载GradientBoosting模型...")
        with open(model_path, 'rb') as f:
            model_info = pickle.load(f)
        return model_info

    def load_historical_data(self, data_path='电力数据.csv'):
        """加载历史数据（确保与训练时格式一致）"""
        print("📂 正在加载历史数据...")
        data = pd.read_csv(data_path)
        data['date'] = pd.to_datetime(data['date'])
        print(f"✅ 历史数据加载成功，形状: {data.shape}")
        return data

    def prepare_future_features(self, historical_data, future_dates):
        """准备未来特征（严格遵循训练时的特征工程逻辑）"""
        print("🔄 准备未来特征...")

        # 创建未来数据框架
        future_data = pd.DataFrame({'date': future_dates})

        # 时间特征（与训练时完全一致）
        future_data['day_of_week'] = future_data['date'].dt.dayofweek
        future_data['day_of_month'] = future_data['date'].dt.day
        future_data['month'] = future_data['date'].dt.month
        future_data['is_weekend'] = future_data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        future_data['is_workday'] = future_data['day_of_week'].apply(lambda x: 1 if x < 5 else 0)

        # 周期性特征（正弦/余弦编码）
        future_data['day_of_week_sin'] = np.sin(2 * np.pi * future_data['day_of_week'] / 7)
        future_data['day_of_week_cos'] = np.cos(2 * np.pi * future_data['day_of_week'] / 7)
        future_data['month_sin'] = np.sin(2 * np.pi * future_data['month'] / 12)
        future_data['month_cos'] = np.cos(2 * np.pi * future_data['month'] / 12)

        # 天气特征（这里用历史同期平均值填充，实际应用中应使用天气预报数据）
        # 关键：确保特征名称和数量与训练时一致
        weather_features = ['max_temp', 'min_temp', 'avg_temp', 'temp_diff',
                            'day_wind_level', 'night_wind_level', 'avg_wind_level',
                            'is_rainy', 'is_extreme_weather', 'weather_encoded']

        for feat in weather_features:
            if feat in historical_data.columns:
                # 用历史同期数据的统计值填充
                future_data[feat] = historical_data.groupby(['month', 'day_of_month'])[feat].transform('median').mean()
            else:
                # 如果训练时存在该特征，填充默认值
                future_data[feat] = 0

        # 确保所有训练时的特征都存在
        for col in self.feature_columns:
            if col not in future_data.columns:
                future_data[col] = 0  # 缺失特征填充默认值

        # 只保留训练时使用的特征，顺序严格一致
        future_data = future_data[self.feature_columns]
        print(f"✅ 未来特征准备完成，形状: {future_data.shape}")

        # 检查特征完整性
        missing_features = set(self.feature_columns) - set(future_data.columns)
        extra_features = set(future_data.columns) - set(self.feature_columns)

        if missing_features:
            print(f"⚠️ 缺失特征: {missing_features}")
            raise ValueError(f"缺失训练时的关键特征: {missing_features}")
        if extra_features:
            print(f"⚠️ 有 {len(extra_features)} 个额外特征，将被忽略")
            future_data = future_data[self.feature_columns]

        print(f"✅ 最终特征数量: {len(future_data.columns)}")
        return future_data

    def predict_future_3months(self, historical_data):
        """预测未来3个月负荷（修复特征对齐问题）"""
        print("\n🚀 开始预测未来3个月负荷...")

        # 生成未来3个月日期（排除重复日期）
        last_date = historical_data['date'].max()
        future_dates = []
        current_date = last_date + timedelta(days=1)
        while len(future_dates) < 90:  # 约3个月
            if current_date not in future_dates:
                future_dates.append(current_date)
            current_date += timedelta(days=1)

        future_dates = pd.to_datetime(future_dates)
        print(f"📅 预测时间范围: {future_dates.min()} 到 {future_dates.max()}")

        # 准备未来特征
        future_features = self.prepare_future_features(historical_data, future_dates)

        # 特征标准化（严格使用训练时的scaler）
        future_features_scaled = self.scaler.transform(future_features)

        # 预测（带进度条）
        print("🔄 正在进行多输出预测...")
        all_predictions = []
        with tqdm(total=len(future_features_scaled), desc="预测进度") as pbar:
            for i in range(len(future_features_scaled)):
                pred = self.model.predict(future_features_scaled[i:i + 1])[0]
                all_predictions.append(pred)
                pbar.update(1)

        # 处理预测结果（确保维度匹配）
        predictions = np.vstack(all_predictions)
        print(f"🔍 预测结果形状: {predictions.shape}")

        # 确保预测结果维度与目标列一致
        if predictions.shape[1] != len(self.target_columns):
            predictions = predictions[:, :len(self.target_columns)]  # 截断多余列

        # 创建预测结果DataFrame
        predictions_df = pd.DataFrame(predictions, columns=self.target_columns)
        predictions_df['date'] = future_dates

        # 保存预测结果
        predictions_df.to_csv('未来3个月电力负荷预测结果.csv', index=False, encoding='utf-8')
        print("✅ 预测结果已保存到: 未来3个月电力负荷预测结果.csv")

        # 可视化预测结果
        self.visualize_predictions(predictions_df)

        return predictions_df

    def visualize_predictions(self, predictions_df):
        """可视化预测结果"""
        print("📊 正在生成预测可视化...")

        # 创建子图（4个行业×2个指标=8个子图）
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        axes = axes.flatten()

        industry_names = ['商业', '大工业用电', '普通工业', '非普工业']
        metrics = ['max_power', 'min_power']

        for idx, industry in enumerate(industry_names):
            for metric_idx, metric in enumerate(metrics):
                col_name = f'{industry}_{metric}'
                ax = axes[idx * 2 + metric_idx]

                ax.plot(predictions_df['date'], predictions_df[col_name],
                        linewidth=2, color='blue', marker='o', markersize=3)
                ax.set_title(f'{industry} - {metric.replace("_power", "负荷")}', fontsize=12)
                ax.set_xlabel('日期')
                ax.set_ylabel('负荷值')
                ax.grid(alpha=0.3)
                ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('未来3个月电力负荷预测可视化.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 预测可视化已保存到: 未来3个月电力负荷预测可视化.png")

    def analyze_model_performance(self, historical_data):
        """分析模型在历史数据上的性能（修复特征对齐问题）"""
        print("\n📊 正在分析模型预测效果...")

        historical_data = historical_data.copy()
        historical_data['date'] = pd.to_datetime(historical_data['date'])
        historical_data = historical_data.sort_values('date')

        # 选择最近90天作为验证期
        validation_start = historical_data['date'].max() - timedelta(days=90)
        validation_data = historical_data[historical_data['date'] >= validation_start]

        if len(validation_data) < 30:
            print("⚠️ 验证数据不足，跳过模型性能分析")
            return

        print(f"🔍 使用验证期: {validation_data['date'].min()} 到 {validation_data['date'].max()}")

        # 准备验证集特征
        feature_data = self.prepare_future_features(
            historical_data[historical_data['date'] < validation_start],
            validation_data['date']
        )

        # 严格对齐特征（关键修复）
        X_val = feature_data.reindex(columns=self.feature_columns, fill_value=0)
        X_val_scaled = self.scaler.transform(X_val)

        # 预测验证集
        print("🔄 验证集预测中...")
        y_pred = self.model.predict(X_val_scaled)
        y_true = validation_data[self.target_columns].values

        # 计算性能指标
        avg_mae = np.mean([mean_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(len(self.target_columns))])
        avg_r2 = np.mean([r2_score(y_true[:, i], y_pred[:, i]) for i in range(len(self.target_columns))])

        print(f"📈 验证集性能:")
        print(f"   - 平均MAE: {avg_mae:.4f}")
        print(f"   - 平均R²: {avg_r2:.4f}")

        # 可视化验证结果
        self.visualize_validation(y_true, y_pred, validation_data['date'])

    def visualize_validation(self, y_true, y_pred, dates):
        """可视化验证结果"""
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        axes = axes.flatten()

        for idx, target in enumerate(self.target_columns):
            ax = axes[idx]
            ax.plot(dates, y_true[:, idx], label='实际值', linewidth=2)
            ax.plot(dates, y_pred[:, idx], label='预测值', linewidth=2, alpha=0.8)
            ax.set_title(f'{target}', fontsize=12)
            ax.set_xlabel('日期')
            ax.set_ylabel('负荷值')
            ax.legend()
            ax.grid(alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('模型验证结果可视化.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 验证结果可视化已保存到: 模型验证结果可视化.png")


def main():
    """主函数"""
    print("=================================================================================")
    print("GradientBoosting模型 - 未来3个月负荷预测系统")
    print("=================================================================================")

    # 初始化预测器
    predictor = GradientBoostingPredictor()

    # 加载历史数据
    historical_data = predictor.load_historical_data()

    # 分析模型性能（验证）
    predictor.analyze_model_performance(historical_data)

    # 预测未来3个月
    predictions = predictor.predict_future_3months(historical_data)

    print("\n🎉 预测完成！")
    print("📋 输出文件:")
    print("   - 未来3个月电力负荷预测结果.csv")
    print("   - 未来3个月电力负荷预测可视化.png")
    print("   - 模型验证结果可视化.png")


if __name__ == "__main__":
    main()