import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
import joblib
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


class UnifiedModelSelector:
    """统一模型选择器 - 为所有行业选择最优的多输出预测模型"""

    def __init__(self):
        # 定义多输出模型
        self.models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoosting': MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42)),
            'XGBoost': MultiOutputRegressor(XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0)
        }

        self.model_performance = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_columns = []
        self.trained_models = {}

        # 定义所有要预测的目标
        self.industries = ['商业', '大工业用电', '普通工业', '非普工业']
        self.target_types = ['max', 'min']

    def prepare_features(self, data):
        """为所有行业准备统一的特征集"""
        df = data.copy()

        # 确保日期格式
        df['date'] = pd.to_datetime(df['date'])

        # 基础时间特征
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year

        # 季节特征
        df['is_spring'] = ((df['month'] >= 3) & (df['month'] <= 5)).astype(int)
        df['is_summer'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)
        df['is_autumn'] = ((df['month'] >= 9) & (df['month'] <= 11)).astype(int)
        df['is_winter'] = ((df['month'] <= 2) | (df['month'] == 12)).astype(int)

        # 节假日特征（简化版）
        if 'day_of_month' in df.columns:
            df['is_holiday'] = ((df['month'] == 1) & (df['day_of_month'] <= 3)) | \
                               ((df['month'] == 5) & (df['day_of_month'] >= 1) & (df['day_of_month'] <= 3)) | \
                               ((df['month'] == 10) & (df['day_of_month'] >= 1) & (df['day_of_month'] <= 7))
        else:
            df['day_of_month'] = df['date'].dt.day
            df['is_holiday'] = ((df['month'] == 1) & (df['day_of_month'] <= 3)) | \
                               ((df['month'] == 5) & (df['day_of_month'] >= 1) & (df['day_of_month'] <= 3)) | \
                               ((df['month'] == 10) & (df['day_of_month'] >= 1) & (df['day_of_month'] <= 7))

        # 历史负荷统计特征（所有行业的汇总）
        max_cols = [f'{industry}_max_power' for industry in self.industries]
        min_cols = [f'{industry}_min_power' for industry in self.industries]

        # 只选择存在的列
        existing_max_cols = [col for col in max_cols if col in df.columns]
        existing_min_cols = [col for col in min_cols if col in df.columns]

        if existing_max_cols:
            df['total_max_power'] = df[existing_max_cols].sum(axis=1)
            df['avg_max_power'] = df[existing_max_cols].mean(axis=1)

        if existing_min_cols:
            df['total_min_power'] = df[existing_min_cols].sum(axis=1)
            df['avg_min_power'] = df[existing_min_cols].mean(axis=1)

        # 滞后特征（使用总负荷）
        for lag in [1, 3, 7]:
            if 'total_max_power' in df.columns:
                df[f'total_max_lag_{lag}'] = df['total_max_power'].shift(lag)
            if 'total_min_power' in df.columns:
                df[f'total_min_lag_{lag}'] = df['total_min_power'].shift(lag)

        # 滚动统计特征
        for window in [7, 14]:
            if 'total_max_power' in df.columns:
                df[f'total_max_rolling_mean_{window}'] = df['total_max_power'].rolling(window=window,
                                                                                       min_periods=1).mean()
                df[f'total_max_rolling_std_{window}'] = df['total_max_power'].rolling(window=window,
                                                                                      min_periods=1).std()
            if 'total_min_power' in df.columns:
                df[f'total_min_rolling_mean_{window}'] = df['total_min_power'].rolling(window=window,
                                                                                       min_periods=1).mean()
                df[f'total_min_rolling_std_{window}'] = df['total_min_power'].rolling(window=window,
                                                                                      min_periods=1).std()

        # 周期性特征
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

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
        self.target_columns = target_names

        return y

    def clean_data(self, X, y):
        """清理数据，处理NaN值"""
        # 合并特征和目标
        data = pd.concat([X, y], axis=1)

        # 多重填充策略
        data = data.fillna(method='ffill').fillna(method='bfill').fillna(data.median()).fillna(0)

        # 分离特征和目标
        X_clean = data[X.columns]
        y_clean = data[y.columns]

        return X_clean, y_clean

    def evaluate_models(self, data):
        """评估所有模型的多输出预测性能"""
        print("正在评估统一模型的多输出预测性能...")

        try:
            # 准备特征
            feature_data = self.prepare_features(data)

            # 选择特征列（排除日期和目标列）
            exclude_cols = ['date'] + [f'{industry}_{target_type}_power'
                                       for industry in self.industries
                                       for target_type in self.target_types]

            feature_cols = [col for col in feature_data.columns if col not in exclude_cols]
            self.feature_columns = feature_cols

            # 准备特征和目标
            X = feature_data[feature_cols]
            y = self.prepare_targets(feature_data)

            print(f"特征数量: {len(feature_cols)}")
            print(f"目标数量: {len(self.target_columns)}")
            print(f"目标列: {self.target_columns}")

            # 清理数据
            X, y = self.clean_data(X, y)

            # 确保数据足够
            if len(X) < 30:
                print("⚠️ 警告: 数据量过少")
                return

            # 划分训练测试集
            split_idx = max(int(len(X) * 0.8), 1)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

            # 标准化特征
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # 确保没有NaN或inf
            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            # 时间序列交叉验证
            n_splits = min(3, len(X_train_scaled) - 1)
            if n_splits >= 2:
                tscv = TimeSeriesSplit(n_splits=n_splits)
            else:
                tscv = None
                print("⚠️ 数据量太少，无法进行交叉验证")

            # 评估每个模型
            for model_name, model in self.models.items():
                print(f"正在训练 {model_name}...")

                cv_scores = []
                final_model = None

                try:
                    # 交叉验证
                    if tscv:
                        for train_idx, val_idx in tscv.split(X_train_scaled):
                            X_cv_train, X_cv_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                            try:
                                model_clone = self._clone_model(model, model_name)
                                model_clone.fit(X_cv_train, y_cv_train)
                                y_pred = model_clone.predict(X_cv_val)

                                # 计算多输出的平均MAE
                                mae_scores = []
                                for i in range(y_cv_val.shape[1]):
                                    mae = mean_absolute_error(y_cv_val.iloc[:, i], y_pred[:, i])
                                    mae_scores.append(mae)

                                cv_scores.append(np.mean(mae_scores))
                            except Exception as e:
                                print(f"  {model_name} 交叉验证失败: {e}")
                                cv_scores.append(np.inf)

                    # 最终评估
                    final_model = self._clone_model(model, model_name)
                    final_model.fit(X_train_scaled, y_train)
                    y_test_pred = final_model.predict(X_test_scaled)

                    # 计算总体性能指标
                    overall_mae = 0
                    overall_rmse = 0
                    overall_r2 = 0

                    # 计算每个目标的性能指标
                    target_performance = {}
                    for i, target_name in enumerate(self.target_columns):
                        mae = mean_absolute_error(y_test.iloc[:, i], y_test_pred[:, i])
                        rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_test_pred[:, i]))
                        r2 = r2_score(y_test.iloc[:, i], y_test_pred[:, i])

                        target_performance[target_name] = {
                            'MAE': mae,
                            'RMSE': rmse,
                            'R2': r2
                        }

                        overall_mae += mae
                        overall_rmse += rmse
                        overall_r2 += r2

                    # 计算平均指标
                    n_targets = len(self.target_columns)
                    overall_mae /= n_targets
                    overall_rmse /= n_targets
                    overall_r2 /= n_targets

                    self.model_performance[model_name] = {
                        'Overall_MAE': overall_mae,
                        'Overall_RMSE': overall_rmse,
                        'Overall_R2': overall_r2,
                        'CV_MAE': np.mean(cv_scores) if cv_scores else overall_mae,
                        'Target_Performance': target_performance
                    }

                    # 存储训练好的模型
                    self.trained_models[model_name] = final_model

                    print(f"  {model_name}: 总体MAE={overall_mae:.2f}, 总体R²={overall_r2:.4f}")

                except Exception as e:
                    print(f"  {model_name} 训练失败: {e}")
                    self.model_performance[model_name] = {
                        'Overall_MAE': np.inf,
                        'Overall_RMSE': np.inf,
                        'Overall_R2': -np.inf,
                        'CV_MAE': np.inf,
                        'Target_Performance': {}
                    }

        except Exception as e:
            print(f"❌ 模型评估失败: {e}")
            import traceback
            traceback.print_exc()

    def _clone_model(self, model, model_name):
        """创建模型的新实例"""
        if model_name == 'RandomForest':
            return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_name == 'GradientBoosting':
            return MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42))
        elif model_name == 'XGBoost':
            return MultiOutputRegressor(XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        elif model_name == 'Ridge':
            return Ridge(alpha=1.0)
        elif model_name == 'Lasso':
            return Lasso(alpha=1.0)
        else:
            return model

    def select_best_model(self):
        """选择最优的统一模型"""
        if not self.model_performance:
            raise ValueError("没有可用的模型性能数据，请先运行 evaluate_models")

        # 过滤有效模型
        valid_models = {name: metrics for name, metrics in self.model_performance.items()
                        if metrics['CV_MAE'] < np.inf and metrics['Overall_R2'] > -np.inf}

        if not valid_models:
            print("⚠️ 所有模型训练失败，无法选择最优模型")
            return None, None

        # 基于交叉验证的MAE选择最佳模型
        best_score = np.inf
        for model_name, metrics in valid_models.items():
            if metrics['CV_MAE'] < best_score:
                best_score = metrics['CV_MAE']
                self.best_model_name = model_name
                self.best_model = self.trained_models.get(model_name, self.models[model_name])

        print(f"\n🎯 最优统一模型: {self.best_model_name}")
        print(f"📊 总体性能指标:")
        print(f"   - 平均MAE: {self.model_performance[self.best_model_name]['Overall_MAE']:.2f}")
        print(f"   - 平均RMSE: {self.model_performance[self.best_model_name]['Overall_RMSE']:.2f}")
        print(f"   - 平均R²: {self.model_performance[self.best_model_name]['Overall_R2']:.4f}")

        return self.best_model_name, self.best_model

    def analyze_model_performance(self):
        """分析模型在各行业各目标上的预测效果"""
        if not self.model_performance or self.best_model_name not in self.model_performance:
            return

        print("\n" + "=" * 80)
        print("各行业各目标预测效果分析")
        print("=" * 80)

        best_performance = self.model_performance[self.best_model_name]['Target_Performance']

        # 按行业分析
        for industry in self.industries:
            print(f"\n📈 {industry}行业:")
            industry_mae = []
            industry_r2 = []

            for target_type in self.target_types:
                target_name = f'{industry}_{target_type}_power'
                if target_name in best_performance:
                    perf = best_performance[target_name]
                    industry_mae.append(perf['MAE'])
                    industry_r2.append(perf['R2'])
                    print(f"   {target_type}负荷: MAE={perf['MAE']:.2f}, R²={perf['R2']:.4f}")

            if industry_mae:
                print(f"   行业平均: MAE={np.mean(industry_mae):.2f}, R²={np.mean(industry_r2):.4f}")

        # 按目标类型分析
        print(f"\n📊 按目标类型分析:")
        for target_type in self.target_types:
            type_mae = []
            type_r2 = []

            for industry in self.industries:
                target_name = f'{industry}_{target_type}_power'
                if target_name in best_performance:
                    perf = best_performance[target_name]
                    type_mae.append(perf['MAE'])
                    type_r2.append(perf['R2'])

            if type_mae:
                print(f"   {target_type}负荷: 平均MAE={np.mean(type_mae):.2f}, 平均R²={np.mean(type_r2):.4f}")

    def plot_performance_comparison(self):
        """绘制模型性能比较图"""
        if not self.model_performance:
            return

        # 过滤有效模型
        valid_models = {name: metrics for name, metrics in self.model_performance.items()
                        if metrics['Overall_MAE'] < np.inf and metrics['Overall_R2'] > -np.inf}

        if not valid_models:
            return

        models = list(valid_models.keys())
        mae_scores = [valid_models[m]['Overall_MAE'] for m in models]
        r2_scores = [valid_models[m]['Overall_R2'] for m in models]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # MAE比较
        bars1 = ax1.bar(models, mae_scores, color='skyblue', alpha=0.7)
        ax1.set_title('各模型总体MAE比较', fontsize=14, fontweight='bold')
        ax1.set_ylabel('平均MAE')
        ax1.tick_params(axis='x', rotation=45)

        for bar, score in zip(bars1, mae_scores):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(mae_scores) * 0.01,
                     f'{score:.1f}', ha='center', va='bottom', fontsize=10)

        # R²比较
        bars2 = ax2.bar(models, r2_scores, color='lightcoral', alpha=0.7)
        ax2.set_title('各模型总体R²比较', fontsize=14, fontweight='bold')
        ax2.set_ylabel('平均R² Score')
        ax2.tick_params(axis='x', rotation=45)

        for bar, score in zip(bars2, r2_scores):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{score:.4f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        os.makedirs('model_comparison', exist_ok=True)
        plt.savefig('model_comparison/unified_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 统一模型比较图已保存: model_comparison/unified_model_comparison.png")


def main():
    """主函数：统一模型选择流程"""
    print("=" * 80)
    print("中期负荷预测 - 统一模型选择系统")
    print("=" * 80)

    try:
        # 加载数据
        print("正在加载数据...")
        data = pd.read_csv('industry_weather_data_daily.csv', encoding='utf-8')
        print(f"✅ 数据加载成功，形状: {data.shape}")

        # 检查数据列
        print("数据列:", list(data.columns))

        # 初始化统一模型选择器
        selector = UnifiedModelSelector()

        # 评估所有模型
        selector.evaluate_models(data)

        # 选择最优模型
        best_name, best_model = selector.select_best_model()

        if best_name and best_model:
            # 分析模型性能
            selector.analyze_model_performance()

            # 绘制比较图
            selector.plot_performance_comparison()

            # 不保存最优模型，只输出总结报告
            print("\n" + "=" * 80)
            print("模型选择总结报告")
            print("=" * 80)
            print(f"最优模型: {best_name}")
            print(f"预测目标: {len(selector.target_columns)}个负荷指标")
            print(f"覆盖行业: {', '.join(selector.industries)}")
            print(f"模型用途: 预测各行业未来3个月日负荷最大值和最小值")
            print(f"总体性能: MAE={selector.model_performance[best_name]['Overall_MAE']:.2f}, "
                  f"R²={selector.model_performance[best_name]['Overall_R2']:.4f}")

            # 输出所有模型性能排名
            print(f"\n📊 模型性能排名:")
            valid_models = {name: metrics for name, metrics in selector.model_performance.items()
                            if metrics['Overall_MAE'] < np.inf and metrics['Overall_R2'] > -np.inf}

            sorted_models = sorted(valid_models.items(), key=lambda x: x[1]['Overall_MAE'])
            for i, (model_name, metrics) in enumerate(sorted_models, 1):
                print(f"  {i}. {model_name}: MAE={metrics['Overall_MAE']:.2f}, R²={metrics['Overall_R2']:.4f}")

            # 返回模型信息但不保存到文件
            model_info = {
                'model_name': best_name,
                'model': best_model,
                'feature_columns': selector.feature_columns,
                'target_columns': selector.target_columns,
                'scaler': selector.scaler,
                'performance': selector.model_performance[best_name],
                'industries': selector.industries,
                'target_types': selector.target_types
            }

            return model_info
        else:
            print("❌ 未能选择出有效的统一模型")
            return None

    except Exception as e:
        print(f"❌ 统一模型选择过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()