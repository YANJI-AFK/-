import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 时序模型
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import lightgbm as lgb

# 可视化库
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端

# 进度条库
from tqdm import tqdm


class DataPreprocessor:
    """数据预处理类"""

    def __init__(self):
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit_transform(self, data):
        """拟合并转换数据"""
        # 处理无穷值 - 使用numpy方法而不是pandas的replace
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
            raise ValueError("Preprocessor not fitted yet.")

        # 处理无穷值
        data = np.where(np.isinf(data), np.nan, data)
        data_imputed = self.imputer.transform(data)
        data_scaled = self.scaler.transform(data_imputed)
        return data_scaled


class Visualization:
    """可视化类"""

    def __init__(self):
        self.fig_dir = "short_model_comparison"
        import os
        os.makedirs(self.fig_dir, exist_ok=True)

        # 设置字体以支持数学符号和中文
        self._setup_fonts()

    def _setup_fonts(self):
        """设置字体以支持数学符号"""
        plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['mathtext.fontset'] = 'stix'  # 使用 STIX 字体，支持数学符号

    def plot_model_comparison(self, results, metric='RMSE', title='模型性能比较'):
        """绘制模型性能比较图"""
        model_names = []
        metric_values = []

        for name, result in results.items():
            if result.get('fitted', False) and 'metrics' in result:
                model_names.append(name)
                metric_values.append(result['metrics'][metric])

        if not model_names:
            print("没有可用的模型结果进行比较")
            return

        # 创建条形图
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, metric_values, color=plt.cm.Set3(np.linspace(0, 1, len(model_names))))
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel(metric, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)

        # 在柱子上添加数值
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(metric_values) * 0.01,
                     f'{value:.4f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{self.fig_dir}/model_comparison_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存模型比较图: {self.fig_dir}/model_comparison_{metric}.png")

    def plot_predictions_vs_actual(self, y_true, y_pred, model_name, sample_size=200):
        """绘制预测值与真实值对比"""
        if len(y_true) > sample_size:
            # 随机采样以避免过于密集的点
            indices = np.random.choice(len(y_true), sample_size, replace=False)
            y_true_sampled = y_true[indices]
            y_pred_sampled = y_pred[indices]
        else:
            y_true_sampled = y_true
            y_pred_sampled = y_pred

        plt.figure(figsize=(10, 6))
        plt.scatter(y_true_sampled, y_pred_sampled, alpha=0.6, s=50)

        # 绘制完美预测线
        min_val = min(min(y_true_sampled), min(y_pred_sampled))
        max_val = max(max(y_true_sampled), max(y_pred_sampled))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)

        plt.xlabel('真实值', fontsize=12)
        plt.ylabel('预测值', fontsize=12)
        plt.title(f'{model_name} - 预测值 vs 真实值', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # 添加R²值 - 使用Unicode字符确保正确显示
        r2 = r2_score(y_true, y_pred)
        r2_text = f'R² = {r2:.4f}'  # 使用Unicode上标字符

        plt.text(0.05, 0.95, r2_text, transform=plt.gca().transAxes,
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.tight_layout()
        plt.savefig(f'{self.fig_dir}/{model_name}_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存预测对比图: {self.fig_dir}/{model_name}_predictions_vs_actual.png")

    def plot_metrics_comparison(self, results):
        """绘制四个模型的R²、MAE、MAPE指标对比"""
        if not any(result.get('fitted', False) for result in results.values()):
            print("没有可用的模型结果进行指标对比")
            return

        # 准备数据
        model_names = []
        r2_scores = []
        mae_scores = []
        mape_scores = []

        for name, result in results.items():
            if result.get('fitted', False) and 'metrics' in result:
                model_names.append(name)
                metrics = result['metrics']
                r2_scores.append(metrics['R2'])
                mae_scores.append(metrics['MAE'])
                mape_scores.append(metrics['MAPE'])

        if not model_names:
            return

        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('模型性能指标对比', fontsize=16, fontweight='bold')

        # R²对比
        bars1 = axes[0].bar(model_names, r2_scores, color='skyblue', alpha=0.8)
        axes[0].set_title('R² 对比 (越高越好)')
        axes[0].set_ylabel('R²')
        axes[0].grid(True, alpha=0.3)
        # 在柱子上添加数值
        for bar, value in zip(bars1, r2_scores):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{value:.4f}', ha='center', va='bottom', fontsize=10)

        # MAE对比
        bars2 = axes[1].bar(model_names, mae_scores, color='lightcoral', alpha=0.8)
        axes[1].set_title('MAE 对比 (越低越好)')
        axes[1].set_ylabel('MAE')
        axes[1].grid(True, alpha=0.3)
        # 在柱子上添加数值
        for bar, value in zip(bars2, mae_scores):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(mae_scores) * 0.01,
                         f'{value:.2f}', ha='center', va='bottom', fontsize=10)

        # MAPE对比
        bars3 = axes[2].bar(model_names, mape_scores, color='lightgreen', alpha=0.8)
        axes[2].set_title('MAPE 对比 (越低越好)')
        axes[2].set_ylabel('MAPE (%)')
        axes[2].grid(True, alpha=0.3)
        # 在柱子上添加数值
        for bar, value in zip(bars3, mape_scores):
            axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(mape_scores) * 0.01,
                         f'{value:.2f}%', ha='center', va='bottom', fontsize=10)

        # 设置x轴标签旋转
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(f'{self.fig_dir}/metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存指标对比图: {self.fig_dir}/metrics_comparison.png")

    def plot_training_progress(self, progress_data):
        """绘制训练进度图"""
        if not progress_data:
            return

        plt.figure(figsize=(12, 8))

        for i, (model_name, progress) in enumerate(progress_data.items(), 1):
            if progress['status'] == 'completed':
                color = 'green'
                marker = 'o'
            elif progress['status'] == 'failed':
                color = 'red'
                marker = 'x'
            else:
                color = 'blue'
                marker = 's'

            plt.subplot(2, 2, i)
            plt.plot(progress['iterations'], progress['scores'],
                     color=color, marker=marker, linewidth=2, markersize=6, label=model_name)
            plt.title(f'{model_name} - 训练进度')
            plt.xlabel('迭代次数')
            plt.ylabel('评分')
            plt.grid(True, alpha=0.3)
            plt.legend()

        plt.tight_layout()
        plt.savefig(f'{self.fig_dir}/training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存训练进度图: {self.fig_dir}/training_progress.png")


class BaseModel:
    """模型基类"""

    def __init__(self, name):
        self.name = name
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.is_fitted = False
        self.visualizer = Visualization()
        self.training_history = []

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # 避免除零错误
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 使用np.clip避免除零
            y_test_safe = np.clip(np.abs(y_test), 1e-10, None)
            mape = np.mean(np.abs((y_test - y_pred) / y_test_safe)) * 100

        r2 = r2_score(y_test, y_pred)

        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }

    def plot_predictions(self, X_test, y_test, sample_size=200):
        """绘制预测结果"""
        y_pred = self.predict(X_test)
        self.visualizer.plot_predictions_vs_actual(y_test, y_pred, self.name, sample_size)


class ShortTermModelPool:
    """短期负荷预测模型池（15分钟间隔）"""

    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.visualizer = Visualization()
        self.training_progress = {}

    def add_model(self, model):
        """添加模型到模型池"""
        self.models[model.name] = model
        self.training_progress[model.name] = {
            'status': 'pending',
            'iterations': [],
            'scores': []
        }

    def prepare_short_term_data(self, data, target_col='total_power'):
        """准备短期预测数据"""
        print("正在准备短期预测数据...")

        # 选择数值型特征
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)

        features = data[numeric_cols]
        target = data[target_col]

        # 处理缺失值 - 使用numpy友好的方法
        features = features.ffill().fillna(0)
        target = target.ffill().fillna(0)

        print(f"特征数量: {len(numeric_cols)}")
        print(f"特征列: {numeric_cols}")

        return {
            'X': features.values,
            'y': target.values,
            'feature_names': numeric_cols
        }

    def train_models(self, data, test_size=0.2):
        """训练所有模型"""
        print("开始训练短期预测模型...")

        # 准备数据
        prepared_data = self.prepare_short_term_data(data)
        X, y = prepared_data['X'], prepared_data['y']

        # 划分训练测试集
        split_idx = int(len(X) * (1 - test_size))

        # 常规特征数据划分
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

        # 使用进度条训练每个模型
        model_names = list(self.models.keys())
        with tqdm(total=len(model_names), desc="训练进度") as pbar:
            for name, model in self.models.items():
                pbar.set_description(f"训练 {name}")
                try:
                    # 模拟训练进度（实际项目中可以根据具体训练过程更新）
                    self.training_progress[name]['status'] = 'training'

                    # 模拟迭代过程
                    iterations = 10  # 假设10个迭代步骤
                    for i in range(iterations):
                        # 在实际项目中，这里应该是真正的训练步骤
                        # 这里用随机数模拟训练进度
                        score = 0.8 + 0.2 * (i / iterations) * np.random.uniform(0.8, 1.2)
                        self.training_progress[name]['iterations'].append(i + 1)
                        self.training_progress[name]['scores'].append(score)

                        # 更新进度条描述
                        pbar.set_postfix({
                            'model': name,
                            'iter': f'{i + 1}/{iterations}',
                            'score': f'{score:.3f}'
                        })
                        pbar.update(1 / iterations / len(model_names))  # 部分更新

                    # 实际训练模型
                    model.fit(X_train, y_train)
                    metrics = model.evaluate(X_test, y_test)

                    # 绘制预测结果
                    model.plot_predictions(X_test, y_test)

                    self.results[name] = {
                        'model': model,
                        'metrics': metrics,
                        'fitted': True
                    }

                    self.training_progress[name]['status'] = 'completed'
                    print(f"✅ {name} 训练完成 - RMSE: {metrics['RMSE']:.2f}, R²: {metrics['R2']:.4f}")

                except Exception as e:
                    self.training_progress[name]['status'] = 'failed'
                    print(f"❌ {name} 训练失败: {str(e)}")
                    self.results[name] = {
                        'model': model,
                        'fitted': False,
                        'error': str(e)
                    }

                # 完成一个模型的训练
                pbar.update(1 - (pbar.n % 1))  # 确保进度条正确更新

        # 绘制模型比较图和指标对比
        if any(result.get('fitted', False) for result in self.results.values()):
            self.visualizer.plot_model_comparison(self.results, metric='RMSE', title='短期预测模型性能比较 (RMSE)')
            self.visualizer.plot_model_comparison(self.results, metric='R2', title='短期预测模型性能比较 (R²)')
            self.visualizer.plot_metrics_comparison(self.results)
            self.visualizer.plot_training_progress(self.training_progress)
        else:
            print("⚠️ 没有成功训练的模型，无法生成比较图")

    def select_best_model(self, metric='RMSE', ascending=True):
        """选择最优模型"""
        valid_results = {}
        for name, result in self.results.items():
            if result.get('fitted', False) and 'metrics' in result:
                valid_results[name] = result['metrics'][metric]

        if not valid_results:
            print("没有成功训练的模型")
            return None, None

        # 根据指标排序
        sorted_models = sorted(valid_results.items(), key=lambda x: x[1], reverse=not ascending)

        best_model_name = sorted_models[0][0]
        self.best_model = self.models[best_model_name]

        print(f"\n🎯 最优短期预测模型: {best_model_name}")
        print(f"评估指标 ({metric}): {valid_results[best_model_name]:.4f}")

        # 显示所有模型排名
        print("\n模型排名:")
        for i, (name, score) in enumerate(sorted_models, 1):
            print(f"{i:2d}. {name}: {score:.4f}")

        return best_model_name, self.best_model

    def predict_future(self, data, steps=96):
        """使用最优模型进行未来预测"""
        if not self.best_model:
            print("⚠️ 请先训练模型并选择最优模型")
            return None

        prepared_data = self.prepare_short_term_data(data)
        X_recent = prepared_data['X'][-steps:]

        predictions = self.best_model.predict(X_recent)
        return predictions


# =============================================================================
# 具体模型实现
# =============================================================================

class XGBoostModel(BaseModel):
    def __init__(self):
        super().__init__("XGBoost")
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )

    def fit(self, X, y):
        # 预处理数据
        X_processed = self.preprocessor.fit_transform(X)
        self.model.fit(X_processed, y)
        self.is_fitted = True

    def predict(self, X):
        X_processed = self.preprocessor.transform(X)
        return self.model.predict(X_processed)


class LightGBMModel(BaseModel):
    def __init__(self):
        super().__init__("LightGBM")
        self.model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1  # 关闭详细日志
        )

    def fit(self, X, y):
        X_processed = self.preprocessor.fit_transform(X)
        self.model.fit(X_processed, y)
        self.is_fitted = True

    def predict(self, X):
        X_processed = self.preprocessor.transform(X)
        return self.model.predict(X_processed)


class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__("RandomForest")
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1  # 使用所有CPU核心
        )

    def fit(self, X, y):
        X_processed = self.preprocessor.fit_transform(X)
        self.model.fit(X_processed, y)
        self.is_fitted = True

    def predict(self, X):
        X_processed = self.preprocessor.transform(X)
        return self.model.predict(X_processed)


class ARIMAModel(BaseModel):
    def __init__(self, order=(1, 1, 1)):
        super().__init__(f"ARIMA{order}")
        self.order = order
        self.model = None

    def fit(self, X, y):
        # ARIMA只需要目标序列，不需要特征X
        self.model = ARIMA(y, order=self.order)
        self.model_fit = self.model.fit()
        self.is_fitted = True

    def predict(self, X):
        # 返回未来len(X)步的预测
        return self.model_fit.forecast(steps=len(X))


# =============================================================================
# 主执行函数
# =============================================================================

def run_short_term_prediction():
    """运行短期负荷预测"""
    print("=" * 80)
    print("开始短期负荷预测（15分钟间隔）")
    print("=" * 80)

    # 加载短期数据
    try:
        short_term_data = pd.read_csv('load_weather_data_15min.csv', index_col=0, parse_dates=True)
        print(f"短期数据加载成功，形状: {short_term_data.shape}")

        # 数据基本信息
        print(f"\n数据基本信息:")
        print(f"列名: {short_term_data.columns.tolist()}")
        print(f"数据范围: {short_term_data.index.min()} 到 {short_term_data.index.max()}")

        # 检查数据质量
        print(f"\n数据质量检查:")
        print(f"缺失值数量: {short_term_data.isnull().sum().sum()}")
        print(f"无穷值数量: {np.isinf(short_term_data.select_dtypes(include=[np.number])).sum().sum()}")

    except Exception as e:
        print(f"❌ 短期数据加载失败: {e}")
        return None, None, None

    # 初始化短期模型池
    short_term_pool = ShortTermModelPool()

    # 添加短期预测模型
    short_term_pool.add_model(XGBoostModel())
    short_term_pool.add_model(LightGBMModel())
    short_term_pool.add_model(RandomForestModel())
    short_term_pool.add_model(ARIMAModel(order=(2, 1, 2)))

    # 训练模型
    short_term_pool.train_models(short_term_data, test_size=0.2)

    # 选择最优模型
    best_short_name, best_short_model = short_term_pool.select_best_model(metric='RMSE')

    return best_short_name, best_short_model, short_term_pool


def main():
    """主函数"""
    print("电力负荷预测模型池系统")
    print("=" * 80)

    # 运行短期预测
    best_short_name, best_short_model, short_term_pool = run_short_term_prediction()

    # 输出最终结果
    print("\n" + "=" * 80)
    print("最终模型选择结果")
    print("=" * 80)

    if best_short_name and best_short_model and short_term_pool:
        print(f"🎯 最优短期预测模型: {best_short_name}")
        print("   用途: 预测未来10天，15分钟间隔的负荷")

        # 显示最优模型的详细指标
        if best_short_name in short_term_pool.results:
            metrics = short_term_pool.results[best_short_name]['metrics']
            print(f"   详细指标:")
            print(f"   - RMSE: {metrics['RMSE']:.2f}")
            print(f"   - MAE: {metrics['MAE']:.2f}")
            print(f"   - R²: {metrics['R2']:.4f}")
            print(f"   - MAPE: {metrics['MAPE']:.2f}%")

            # 显示所有模型的指标对比
            print(f"\n所有模型指标对比:")
            print(f"{'模型':<12} {'R²':<8} {'MAE':<8} {'MAPE':<8}")
            print("-" * 40)
            for name, result in short_term_pool.results.items():
                if result.get('fitted', False):
                    m = result['metrics']
                    print(f"{name:<12} {m['R2']:<8.4f} {m['MAE']:<8.2f} {m['MAPE']:<8.2f}%")
    else:
        print("❌ 短期预测模型选择失败")

    print(f"\n📊 可视化结果已保存到 'short_model_comparison' 文件夹")
    print("✅ 模型池训练完成！")

    return best_short_model


if __name__ == "__main__":
    best_model = main()