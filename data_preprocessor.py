import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')


class PowerDataPreprocessor:
    """电力数据预处理类"""

    def __init__(self):
        self.weather_encoder = LabelEncoder()
        self.wind_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        # 更新文件路径 - 根据图片中的文件名
        self.file_paths = {
            'load_data': '区域15分钟负荷数据.csv',
            'weather_data': '气象数据.csv',
            'industry_data': '行业日负荷数据.csv'
        }

        # 存储原始数据副本
        self.raw_load_data = None
        self.raw_weather_data = None
        self.raw_industry_data = None

    def load_data(self):
        """加载所有数据文件"""
        print("正在加载数据...")

        try:
            # 加载负荷数据
            self.raw_load_data = pd.read_csv(self.file_paths['load_data'])
            print(f"负荷数据形状: {self.raw_load_data.shape}")

            # 加载天气数据
            self.raw_weather_data = pd.read_csv(self.file_paths['weather_data'])
            print(f"天气数据形状: {self.raw_weather_data.shape}")

            # 加载行业数据
            self.raw_industry_data = pd.read_csv(self.file_paths['industry_data'])
            print(f"行业数据形状: {self.raw_industry_data.shape}")

            return self.raw_load_data, self.raw_weather_data, self.raw_industry_data

        except Exception as e:
            print(f"加载数据出错: {e}")
            print("尝试使用当前目录下的文件...")
            return self.try_alternative_paths()

    def try_alternative_paths(self):
        """尝试其他文件路径"""
        # 检查当前目录下的文件
        current_files = os.listdir('.')
        print("当前目录下的文件:", [f for f in current_files if f.endswith(('.xlsx', '.xls', '.csv'))])

        # 尝试直接加载CSV文件
        try:
            self.raw_load_data = pd.read_csv('区域15分钟负荷数据.csv')
            self.raw_weather_data = pd.read_csv('气象数据.csv')
            self.raw_industry_data = pd.read_csv('行业日负荷数据.csv')
            print("成功加载CSV文件")
            return self.raw_load_data, self.raw_weather_data, self.raw_industry_data
        except Exception as e:
            print(f"加载CSV文件失败: {e}")
            # 尝试Excel文件
            try:
                self.raw_load_data = pd.read_excel('区域15分钟负荷数据.xlsx')
                self.raw_weather_data = pd.read_excel('气象数据.xlsx')
                self.raw_industry_data = pd.read_excel('行业日负荷数据.xlsx')
                print("成功加载Excel文件")
                return self.raw_load_data, self.raw_weather_data, self.raw_industry_data
            except:
                # 如果文件都不存在，创建示例数据继续演示
                print("创建示例数据以继续演示...")
                return self.create_sample_data()

    def create_sample_data(self):
        """创建示例数据"""
        # 创建负荷数据示例
        dates = pd.date_range('2018-01-01', '2018-12-31', freq='15T')
        self.raw_load_data = pd.DataFrame({
            '数据时间': dates,
            '总有功功率（kw）': np.random.normal(250000, 50000, len(dates))
        })

        # 创建天气数据示例
        weather_dates = pd.date_range('2018-01-01', '2018-12-31', freq='D')
        self.raw_weather_data = pd.DataFrame({
            '日期': weather_dates,
            '天气状况': np.random.choice(['晴', '多云', '阴', '小雨'], len(weather_dates)),
            '最高温度': np.random.randint(10, 35, len(weather_dates)),
            '最低温度': np.random.randint(0, 25, len(weather_dates)),
            '白天风力风向': '无持续风向<3级',
            '夜晚风力风向': '无持续风向<3级'
        })

        # 创建行业数据示例
        industry_dates = pd.date_range('2018-01-01', '2018-12-31', freq='D')
        industries = ['大工业用电', '非普工业', '商业', '普通工业']
        industry_data = []
        for date in industry_dates:
            for industry in industries:
                industry_data.append({
                    '行业类型': industry,
                    '数据时间': date,
                    '有功功率最大值（kw）': np.random.uniform(1000, 150000),
                    '有功功率最小值（kw）': np.random.uniform(500, 100000)
                })
        self.raw_industry_data = pd.DataFrame(industry_data)

        return self.raw_load_data, self.raw_weather_data, self.raw_industry_data

    def preprocess_load_data(self, load_data=None):
        """预处理负荷数据"""
        print("\n正在预处理负荷数据...")

        if load_data is None:
            load_data = self.raw_load_data.copy()

        # 显示数据基本信息
        print("负荷数据列名:", load_data.columns.tolist())
        print("前3行数据:")
        print(load_data.head(3))

        # 重命名列
        column_mapping = {}
        if '数据时间' in load_data.columns:
            column_mapping['数据时间'] = 'timestamp'
        if '总有功功率（kw）' in load_data.columns:
            column_mapping['总有功功率（kw）'] = 'total_power'

        if column_mapping:
            load_data = load_data.rename(columns=column_mapping)
        else:
            # 如果列名不匹配，使用前两列
            load_data.columns = ['timestamp', 'total_power']

        # 转换时间格式
        load_data['timestamp'] = pd.to_datetime(load_data['timestamp'], errors='coerce')

        # 删除无效时间
        load_data = load_data.dropna(subset=['timestamp'])

        # 设置时间索引
        load_data = load_data.set_index('timestamp').sort_index()

        # 检查缺失值
        print(f"负荷数据缺失值数量: {load_data['total_power'].isnull().sum()}")

        # 处理缺失值
        if load_data['total_power'].isnull().sum() > 0:
            load_data['total_power'] = load_data['total_power'].fillna(method='ffill')
            load_data['total_power'] = load_data['total_power'].fillna(method='bfill')

        # 添加时间特征
        load_data['hour'] = load_data.index.hour
        load_data['day_of_week'] = load_data.index.dayofweek
        load_data['day_of_month'] = load_data.index.day
        load_data['month'] = load_data.index.month
        load_data['is_weekend'] = (load_data['day_of_week'] >= 5).astype(int)
        load_data['is_workday'] = 1 - load_data['is_weekend']

        print("负荷数据预处理完成")
        return load_data

    def preprocess_weather_data(self, weather_data=None):
        """预处理天气数据"""
        print("\n正在预处理天气数据...")

        if weather_data is None:
            weather_data = self.raw_weather_data.copy()

        print("天气数据列名:", weather_data.columns.tolist())
        print("前3行数据:")
        print(weather_data.head(3))

        # 重命名列
        column_mapping = {}
        if '日期' in weather_data.columns:
            column_mapping['日期'] = 'date'
        if '天气状况' in weather_data.columns:
            column_mapping['天气状况'] = 'weather'
        if '最高温度' in weather_data.columns:
            column_mapping['最高温度'] = 'max_temp'
        if '最低温度' in weather_data.columns:
            column_mapping['最低温度'] = 'min_temp'
        if '白天风力风向' in weather_data.columns:
            column_mapping['白天风力风向'] = 'day_wind'
        if '夜晚风力风向' in weather_data.columns:
            column_mapping['夜晚风力风向'] = 'night_wind'

        if column_mapping:
            weather_data = weather_data.rename(columns=column_mapping)
        else:
            # 如果列名已经是英文，跳过重命名
            if 'date' not in weather_data.columns:
                # 如果列名不匹配，使用前几列
                cols = ['date', 'weather', 'max_temp', 'min_temp', 'day_wind', 'night_wind']
                weather_data.columns = cols[:len(weather_data.columns)]

        # 处理日期列
        weather_data['date'] = pd.to_datetime(weather_data['date'], errors='coerce')
        weather_data = weather_data.dropna(subset=['date'])

        # 去除重复行
        weather_data = weather_data.drop_duplicates(subset=['date']).reset_index(drop=True)

        # 处理温度数据
        def extract_temperature(temp_str):
            if pd.isna(temp_str):
                return np.nan
            temp_str = str(temp_str)
            # 尝试提取温度数值
            match = re.search(r'(-?\d+)℃', temp_str)
            if match:
                return float(match.group(1))
            else:
                # 尝试直接提取数字
                numbers = re.findall(r'-?\d+', temp_str)
                return float(numbers[0]) if numbers else np.nan

        if 'max_temp' in weather_data.columns:
            weather_data['max_temp'] = weather_data['max_temp'].apply(extract_temperature)
        if 'min_temp' in weather_data.columns:
            weather_data['min_temp'] = weather_data['min_temp'].apply(extract_temperature)

        # 计算平均温度
        if 'max_temp' in weather_data.columns and 'min_temp' in weather_data.columns:
            weather_data['avg_temp'] = (weather_data['max_temp'] + weather_data['min_temp']) / 2

        # 处理天气状况
        def simplify_weather(weather_str):
            if pd.isna(weather_str):
                return '未知'
            weather_str = str(weather_str)
            if any(rain in weather_str for rain in ['雨', '降水']):
                if any(heavy in weather_str for heavy in ['大雨', '暴雨', '大暴雨']):
                    return '大雨'
                elif '中雨' in weather_str:
                    return '中雨'
                else:
                    return '小雨'
            elif '雪' in weather_str:
                return '雪'
            elif '晴' in weather_str:
                return '晴'
            elif '多云' in weather_str:
                return '多云'
            elif '阴' in weather_str:
                return '阴'
            else:
                return '其他'

        if 'weather' in weather_data.columns:
            weather_data['weather_simple'] = weather_data['weather'].apply(simplify_weather)

        # 编码天气类型
        if 'weather_simple' in weather_data.columns:
            weather_data['weather_encoded'] = self.weather_encoder.fit_transform(
                weather_data['weather_simple']
            )

        # 处理风力数据
        def extract_wind_level(wind_str):
            if pd.isna(wind_str):
                return 0
            wind_str = str(wind_str)
            # 提取风力等级
            if any(pattern in wind_str for pattern in ['4～5级', '4-5级']):
                return 4.5
            elif any(pattern in wind_str for pattern in ['3～4级', '3-4级']):
                return 3.5
            elif any(pattern in wind_str for pattern in ['4级', '4-']):
                return 4
            elif '3级' in wind_str:
                return 3
            elif '微风' in wind_str:
                return 1
            elif '无持续风向' in wind_str:
                return 0
            else:
                return 0

        if 'day_wind' in weather_data.columns:
            weather_data['day_wind_level'] = weather_data['day_wind'].apply(extract_wind_level)
        if 'night_wind' in weather_data.columns:
            weather_data['night_wind_level'] = weather_data['night_wind'].apply(extract_wind_level)

        if 'day_wind_level' in weather_data.columns and 'night_wind_level' in weather_data.columns:
            weather_data['avg_wind_level'] = (weather_data['day_wind_level'] + weather_data['night_wind_level']) / 2

        print("天气数据预处理完成")
        return weather_data

    def preprocess_industry_data(self, industry_data=None):
        """预处理行业数据"""
        print("\n正在预处理行业数据...")

        if industry_data is None:
            industry_data = self.raw_industry_data.copy()

        print("行业数据列名:", industry_data.columns.tolist())
        print("前3行数据:")
        print(industry_data.head(3))

        # 重命名列
        column_mapping = {}
        if '行业类型' in industry_data.columns:
            column_mapping['行业类型'] = 'industry_type'
        if '数据时间' in industry_data.columns:
            column_mapping['数据时间'] = 'timestamp'
        if '有功功率最大值（kw）' in industry_data.columns:
            column_mapping['有功功率最大值（kw）'] = 'max_power'
        if '有功功率最小值（kw）' in industry_data.columns:
            column_mapping['有功功率最小值（kw）'] = 'min_power'

        if column_mapping:
            industry_data = industry_data.rename(columns=column_mapping)
        else:
            # 如果列名不匹配，使用前几列
            cols = ['industry_type', 'timestamp', 'max_power', 'min_power']
            industry_data.columns = cols[:len(industry_data.columns)]

        # 转换时间格式
        if 'timestamp' in industry_data.columns:
            industry_data['timestamp'] = pd.to_datetime(industry_data['timestamp'], errors='coerce')
            industry_data = industry_data.dropna(subset=['timestamp'])

            # 添加日期列用于合并
            industry_data['date'] = industry_data['timestamp'].dt.date
            industry_data['date'] = pd.to_datetime(industry_data['date'])

        print("行业数据预处理完成")
        return industry_data

    def create_load_weather_dataset(self):
        """创建负荷数据与天气数据的合并数据集（15分钟间隔）"""
        print("\n" + "=" * 60)
        print("正在创建负荷-天气数据集（15分钟间隔）")
        print("=" * 60)

        # 预处理负荷数据
        load_processed = self.preprocess_load_data()

        # 预处理天气数据
        weather_processed = self.preprocess_weather_data()

        # 为负荷数据添加日期列
        load_processed_reset = load_processed.reset_index()
        load_processed_reset['date'] = load_processed_reset['timestamp'].dt.date
        load_processed_reset['date'] = pd.to_datetime(load_processed_reset['date'])

        # 合并负荷数据和天气数据
        merged_data = pd.merge(
            load_processed_reset,
            weather_processed,
            left_on='date',
            right_on='date',
            how='left'
        )

        # 设置时间索引
        merged_data = merged_data.set_index('timestamp').sort_index()

        # 创建时间序列特征
        final_data = self.create_time_features_for_load(merged_data)

        # 填充合并后的缺失值
        numeric_cols = final_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            final_data[col] = final_data[col].fillna(method='ffill')
            final_data[col] = final_data[col].fillna(method='bfill')
            if final_data[col].isnull().sum() > 0:
                final_data[col] = final_data[col].fillna(final_data[col].mean())

        # 选择最终特征 - 只保留数值型特征
        exclude_features = ['weather', 'day_wind', 'night_wind', 'date', 'weather_simple']
        numeric_columns = final_data.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in numeric_columns
                           if col not in exclude_features and not col.startswith('Unnamed')]

        final_dataset = final_data[feature_columns]

        print(f"\n负荷-天气数据集形状: {final_dataset.shape}")
        print(f"特征数量: {len(feature_columns)}")

        return final_dataset, feature_columns

    def create_industry_weather_dataset(self):
        """创建行业数据与天气数据的合并数据集（日级数据）"""
        print("\n" + "=" * 60)
        print("正在创建行业-天气数据集（日级数据）")
        print("=" * 60)

        # 预处理行业数据 - 使用原始数据
        industry_processed = self.preprocess_industry_data()

        # 预处理天气数据 - 使用原始数据
        weather_processed = self.preprocess_weather_data(self.raw_weather_data.copy())

        # 按行业类型和日期聚合行业数据
        if 'industry_type' in industry_processed.columns:
            industry_daily = industry_processed.groupby(['industry_type', 'date']).agg({
                'max_power': 'mean',
                'min_power': 'mean'
            }).reset_index()

            # 创建行业用电特征（按行业类型展开）
            industry_pivot = industry_daily.pivot_table(
                index='date',
                columns='industry_type',
                values=['max_power', 'min_power'],
                aggfunc='mean'
            )

            # 扁平化列名
            industry_pivot.columns = [f"{col[1]}_{col[0]}" for col in industry_pivot.columns]
            industry_pivot = industry_pivot.reset_index()
        else:
            industry_pivot = pd.DataFrame()

        # 合并行业数据和天气数据
        if not industry_pivot.empty:
            merged_data = pd.merge(
                industry_pivot,
                weather_processed,
                left_on='date',
                right_on='date',
                how='left'
            )
        else:
            merged_data = weather_processed.copy()

        # 设置日期索引
        merged_data = merged_data.set_index('date').sort_index()

        # 创建日级时间特征
        final_data = self.create_time_features_for_industry(merged_data)

        # 填充合并后的缺失值
        numeric_cols = final_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            final_data[col] = final_data[col].fillna(method='ffill')
            final_data[col] = final_data[col].fillna(method='bfill')
            if final_data[col].isnull().sum() > 0:
                final_data[col] = final_data[col].fillna(final_data[col].mean())

        print(f"\n行业-天气数据集形状: {final_data.shape}")

        return final_data

    def create_time_features_for_load(self, data):
        """为负荷数据创建时间序列特征（15分钟间隔）"""
        print("\n正在创建负荷数据时间序列特征...")

        # 周期性特征（使用正弦余弦编码）
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

        # 滞后特征（针对15分钟间隔数据）
        target_col = 'total_power'
        if target_col in data.columns:
            lags = [1, 2, 3, 4, 24, 48, 96]  # 15min, 30min, 45min, 1h, 6h, 12h, 24h
            for lag in lags:
                data[f'power_lag_{lag}'] = data[target_col].shift(lag)

        # 滚动统计特征
        if target_col in data.columns:
            data['power_rolling_mean_6h'] = data[target_col].rolling(24, min_periods=1).mean()
            data['power_rolling_std_6h'] = data[target_col].rolling(24, min_periods=1).std()
            data['power_rolling_mean_1d'] = data[target_col].rolling(96, min_periods=1).mean()

        # 温度相关特征
        if 'max_temp' in data.columns and 'min_temp' in data.columns:
            data['temp_diff'] = data['max_temp'] - data['min_temp']  # 温差

        # 天气影响特征
        if 'weather_encoded' in data.columns:
            data['is_rainy'] = (data['weather_simple'].isin(['小雨', '中雨', '大雨'])).astype(int)
            data['is_extreme_weather'] = (data['weather_simple'].isin(['大雨', '雪'])).astype(int)

        # 填充由滞后产生的缺失值
        data = data.fillna(method='bfill')
        data = data.fillna(method='ffill')

        # 最后用均值填充任何剩余的缺失值
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].mean())

        print("负荷数据时间序列特征创建完成")
        return data

    def create_time_features_for_industry(self, data):
        """为行业数据创建时间序列特征（日级数据）"""
        print("\n正在创建行业数据时间序列特征...")

        # 添加时间特征
        data['day_of_week'] = data.index.dayofweek
        data['day_of_month'] = data.index.day
        data['month'] = data.index.month
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        data['is_workday'] = 1 - data['is_weekend']

        # 周期性特征（使用正弦余弦编码）
        data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

        # 温度相关特征
        if 'max_temp' in data.columns and 'min_temp' in data.columns:
            data['temp_diff'] = data['max_temp'] - data['min_temp']  # 温差

        # 天气影响特征
        if 'weather_encoded' in data.columns:
            data['is_rainy'] = (data['weather_simple'].isin(['小雨', '中雨', '大雨'])).astype(int)
            data['is_extreme_weather'] = (data['weather_simple'].isin(['大雨', '雪'])).astype(int)

        # 填充缺失值
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            data[col] = data[col].fillna(method='ffill')
            data[col] = data[col].fillna(method='bfill')
            if data[col].isnull().sum() > 0:
                data[col] = data[col].fillna(data[col].mean())

        print("行业数据时间序列特征创建完成")
        return data


# 使用示例
if __name__ == "__main__":
    # 初始化预处理器
    preprocessor = PowerDataPreprocessor()

    # 加载数据
    load_data, weather_data, industry_data = preprocessor.load_data()

    # 创建两个独立的数据集
    print("\n" + "=" * 80)
    print("开始创建两个独立的数据集")
    print("=" * 80)

    # 1. 负荷-天气数据集（15分钟间隔）
    load_weather_data, load_features = preprocessor.create_load_weather_dataset()

    # 2. 行业-天气数据集（日级数据）
    industry_weather_data = preprocessor.create_industry_weather_dataset()

    # 显示数据基本信息
    print("\n" + "=" * 80)
    print("数据预处理完成!")
    print("=" * 80)

    # 负荷-天气数据集信息
    print(f"\n📊 负荷-天气数据集（15分钟间隔）:")
    print(f"   数据形状: {load_weather_data.shape}")
    print(f"   时间范围: {load_weather_data.index.min()} 到 {load_weather_data.index.max()}")
    print(f"   总记录数: {len(load_weather_data)}")
    print(f"   特征数量: {len(load_features)}")

    # 行业-天气数据集信息
    print(f"\n📊 行业-天气数据集（日级数据）:")
    print(f"   数据形状: {industry_weather_data.shape}")
    print(f"   时间范围: {industry_weather_data.index.min()} 到 {industry_weather_data.index.max()}")
    print(f"   总记录数: {len(industry_weather_data)}")
    print(f"   特征数量: {len(industry_weather_data.columns)}")

    # 显示前几行数据
    print(f"\n负荷-天气数据集前3行:")
    print(load_weather_data.head(3))

    print(f"\n行业-天气数据集前3行:")
    print(industry_weather_data.head(3))

    # 保存处理后的数据
    load_weather_data.to_csv('load_weather_data_15min.csv', encoding='utf-8-sig')
    industry_weather_data.to_csv('industry_weather_data_daily.csv', encoding='utf-8-sig')

    print(f"\n💾 数据保存完成:")
    print(f"   • 负荷-天气数据: 'load_weather_data_15min.csv'")
    print(f"   • 行业-天气数据: 'industry_weather_data_daily.csv'")

    # 保存特征列表
    feature_info_load = pd.DataFrame({
        'feature_name': load_features,
        'feature_type': [load_weather_data[col].dtype for col in load_features]
    })
    feature_info_load.to_csv('load_weather_features.csv', index=False, encoding='utf-8-sig')

    feature_info_industry = pd.DataFrame({
        'feature_name': industry_weather_data.columns.tolist(),
        'feature_type': [industry_weather_data[col].dtype for col in industry_weather_data.columns]
    })
    feature_info_industry.to_csv('industry_weather_features.csv', index=False, encoding='utf-8-sig')

    print(f"   • 特征列表: 'load_weather_features.csv', 'industry_weather_features.csv'")