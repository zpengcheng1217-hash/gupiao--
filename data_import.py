"""
CSV数据导入模块 - 用于导入股票数据并进行分析
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockDataImporter:
    """股票数据导入和处理类"""
    
    def __init__(self, csv_path):
        """
        初始化导入器
        
        Args:
            csv_path: CSV文件路径
        """
        self.csv_path = csv_path
        self.df = None
        
    def load_csv(self):
        """加载CSV文件"""
        try:
            self.df = pd.read_csv(self.csv_path)
            logger.info(f"成功加载CSV文件: {self.csv_path}")
            logger.info(f"数据行数: {len(self.df)}, 列数: {len(self.df.columns)}")
            logger.info(f"列名: {list(self.df.columns)}")
            return self.df
        except Exception as e:
            logger.error(f"加载CSV文件失败: {str(e)}")
            return None
    
    def prepare_data(self):
        """准备数据"""
        if self.df is None:
            logger.error("数据未加载")
            return None
        
        df = self.df.copy()
        
        # 确保日期列格式正确
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif '日期' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'])
            df.rename(columns={'日期': 'date'}, inplace=True)
        
        # 确保数值列为正确格式
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 按日期升序排列
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
        
        self.df = df
        logger.info("数据准备完成")
        return self.df
    
    def get_data_info(self):
        """获取数据信息"""
        if self.df is None:
            return None
        
        info = {
            '总行数': len(self.df),
            '列名': list(self.df.columns),
            '开始日期': self.df['date'].min() if 'date' in self.df.columns else 'N/A',
            '结束日期': self.df['date'].max() if 'date' in self.df.columns else 'N/A',
            '最低价': self.df['low'].min() if 'low' in self.df.columns else 'N/A',
            '最高价': self.df['high'].max() if 'high' in self.df.columns else 'N/A',
            '平均成交量': self.df['volume'].mean() if 'volume' in self.df.columns else 'N/A',
        }
        
        return info


class TradingSignalAnalyzer:
    """交易信号分析器"""
    
    def __init__(self, df):
        """
        初始化分析器
        
        Args:
            df: 股票数据框
        """
        self.df = df.copy()
        
    def calculate_indicators(self):
        """计算技术指标"""
        df = self.df
        
        # 1. 简单移动平均线 (SMA)
        df['SMA_5'] = df['close'].rolling(window=5).mean()
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        
        # 2. 指数移动平均线 (EMA)
        df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # 3. RSI (相对强弱指数)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 4. MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # 5. 布林线
        df['BB_Middle'] = df['close'].rolling(window=20).mean()
        df['BB_Std'] = df['close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
        
        # 6. 价格变化率
        df['Price_Change'] = df['close'].pct_change() * 100
        df['Price_Change_5d'] = df['close'].pct_change(5) * 100
        
        logger.info("技术指标计算完成")
        return df
    
    def generate_buy_signals(self, df):
        """
        生成买入信号
        
        策略：
        1. 价格突破20日均线向上
        2. RSI < 30 (超卖)
        3. MACD金叉
        4. 价格在布林线下方
        """
        signals = []
        
        for i in range(1, len(df)):
            buy_signal = False
            reasons = []
            
            # 条件1: 价格突破20日均线向上
            if df['close'].iloc[i] > df['SMA_20'].iloc[i] and \
               df['close'].iloc[i-1] <= df['SMA_20'].iloc[i-1]:
                buy_signal = True
                reasons.append("价格突破20日均线向上")
            
            # 条件2: RSI < 30 (超卖)
            if pd.notna(df['RSI'].iloc[i]) and df['RSI'].iloc[i] < 30:
                buy_signal = True
                reasons.append("RSI超卖 (<30)")
            
            # 条件3: MACD金叉
            if i > 1 and pd.notna(df['MACD'].iloc[i]) and pd.notna(df['MACD_Signal'].iloc[i]):
                if df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i] and \
                   df['MACD'].iloc[i-1] <= df['MACD_Signal'].iloc[i-1]:
                    buy_signal = True
                    reasons.append("MACD金叉")
            
            # 条件4: 价格在布林线下方
            if pd.notna(df['BB_Lower'].iloc[i]) and \
               df['close'].iloc[i] < df['BB_Lower'].iloc[i]:
                buy_signal = True
                reasons.append("价格触及布林线下轨")
            
            if buy_signal:
                signals.append({
                    'date': df['date'].iloc[i],
                    'price': df['close'].iloc[i],
                    'type': '买入',
                    'reasons': '；'.join(reasons),
                    'rsi': df['RSI'].iloc[i] if pd.notna(df['RSI'].iloc[i]) else None,
                    'macd': df['MACD'].iloc[i] if pd.notna(df['MACD'].iloc[i]) else None,
                })
        
        return signals
    
    def generate_sell_signals(self, df):
        """
        生成卖出信号
        
        策略：
        1. 价格跌破20日均线向下
        2. RSI > 70 (超买)
        3. MACD死叉
        4. 价格在布林线上方
        5. 跌幅超过2-3% (止损)
        """
        signals = []
        
        for i in range(1, len(df)):
            sell_signal = False
            reasons = []
            
            # ��件1: 价格跌破20日均线向下
            if df['close'].iloc[i] < df['SMA_20'].iloc[i] and \
               df['close'].iloc[i-1] >= df['SMA_20'].iloc[i-1]:
                sell_signal = True
                reasons.append("价格跌破20日均线向下")
            
            # 条件2: RSI > 70 (超买)
            if pd.notna(df['RSI'].iloc[i]) and df['RSI'].iloc[i] > 70:
                sell_signal = True
                reasons.append("RSI超买 (>70)")
            
            # 条件3: MACD死叉
            if i > 1 and pd.notna(df['MACD'].iloc[i]) and pd.notna(df['MACD_Signal'].iloc[i]):
                if df['MACD'].iloc[i] < df['MACD_Signal'].iloc[i] and \
                   df['MACD'].iloc[i-1] >= df['MACD_Signal'].iloc[i-1]:
                    sell_signal = True
                    reasons.append("MACD死叉")
            
            # 条件4: 价格在布林线上方
            if pd.notna(df['BB_Upper'].iloc[i]) and \
               df['close'].iloc[i] > df['BB_Upper'].iloc[i]:
                sell_signal = True
                reasons.append("价格触及布林线上轨")
            
            # 条件5: 止损 (跌幅超过2-3%)
            if pd.notna(df['Price_Change'].iloc[i]) and \
               df['Price_Change'].iloc[i] < -2.0:
                sell_signal = True
                reasons.append("触发止损 (跌幅>2%)")
            
            if sell_signal:
                signals.append({
                    'date': df['date'].iloc[i],
                    'price': df['close'].iloc[i],
                    'type': '卖出',
                    'reasons': '；'.join(reasons),
                    'rsi': df['RSI'].iloc[i] if pd.notna(df['RSI'].iloc[i]) else None,
                    'macd': df['MACD'].iloc[i] if pd.notna(df['MACD'].iloc[i]) else None,
                })
        
        return signals
    
    def generate_all_signals(self):
        """生成所有交易信号"""
        # 计算指标
        df = self.calculate_indicators()
        
        # 生成信号
        buy_signals = self.generate_buy_signals(df)
        sell_signals = self.generate_sell_signals(df)
        
        # 合并并排序
        all_signals = buy_signals + sell_signals
        all_signals = sorted(all_signals, key=lambda x: x['date'])
        
        logger.info(f"生成了 {len(buy_signals)} 个买入信号，{len(sell_signals)} 个卖出信号")
        
        return {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'all_signals': all_signals,
            'data_with_indicators': df
        }
    
    def calculate_strategy_performance(self, buy_signals, sell_signals):
        """计算策略表现"""
        if not buy_signals or not sell_signals:
            return None
        
        trades = []
        buy_idx = 0
        
        for sell in sell_signals:
            # 找到对应的买入信号
            matching_buys = [b for b in buy_signals if b['date'] < sell['date']]
            if matching_buys:
                buy = matching_buys[-1]  # 取最近的买入
                
                profit = sell['price'] - buy['price']
                profit_pct = (profit / buy['price']) * 100
                
                trades.append({
                    'buy_date': buy['date'],
                    'buy_price': buy['price'],
                    'sell_date': sell['date'],
                    'sell_price': sell['price'],
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'days_held': (sell['date'] - buy['date']).days
                })
        
        if trades:
            total_profit = sum([t['profit'] for t in trades])
            winning_trades = len([t for t in trades if t['profit'] > 0])
            losing_trades = len([t for t in trades if t['profit'] < 0])
            avg_profit_pct = np.mean([t['profit_pct'] for t in trades])
            win_rate = winning_trades / len(trades) * 100 if trades else 0
            
            performance = {
                'total_trades': len(trades),
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'avg_profit_pct': avg_profit_pct,
                'trades': trades
            }
            
            return performance
        
        return None


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("开始股票数据分析")
    logger.info("=" * 80)
    
    # 1. 导入数据
    logger.info("\n[步骤1] 导入CSV数据")
    csv_file = '000002.sz.csv'
    
    if not os.path.exists(csv_file):
        logger.error(f"文件不存在: {csv_file}")
        return
    
    importer = StockDataImporter(csv_file)
    importer.load_csv()
    importer.prepare_data()
    
    # 获取数据信息
    info = importer.get_data_info()
    logger.info("\n数据信息:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")
    
    df = importer.df
    
    # 2. 分析交易信号
    logger.info("\n[步骤2] 分析交易信号")
    analyzer = TradingSignalAnalyzer(df)
    results = analyzer.generate_all_signals()
    
    buy_signals = results['buy_signals']
    sell_signals = results['sell_signals']
    df_with_indicators = results['data_with_indicators']
    
    # 3. 显示最近的买入信号
    logger.info("\n[最近的买入信号]")
    if buy_signals:
        for signal in buy_signals[-5:]:  # 显示最后5个
            logger.info(f"  日期: {signal['date'].strftime('%Y-%m-%d')}, "
                       f"价格: {signal['price']:.2f}, "
                       f"原因: {signal['reasons']}")
    else:
        logger.info("  没有买入信号")
    
    # 4. 显示最近的卖出信号
    logger.info("\n[最近的卖出信号]")
    if sell_signals:
        for signal in sell_signals[-5:]:  # 显示最后5个
            logger.info(f"  日期: {signal['date'].strftime('%Y-%m-%d')}, "
                       f"价格: {signal['price']:.2f}, "
                       f"原因: {signal['reasons']}")
    else:
        logger.info("  没有卖出信号")
    
    # 5. 计算策略表现
    logger.info("\n[步骤3] 计算策略表现")
    performance = analyzer.calculate_strategy_performance(buy_signals, sell_signals)
    
    if performance:
        logger.info(f"  总交易数: {performance['total_trades']}")
        logger.info(f"  赢利交易: {performance['winning_trades']}")
        logger.info(f"  亏损交易: {performance['losing_trades']}")
        logger.info(f"  胜率: {performance['win_rate']:.2f}%")
        logger.info(f"  总利润: {performance['total_profit']:.2f}")
        logger.info(f"  平均收益率: {performance['avg_profit_pct']:.2f}%")
    
    # 6. 保存结果
    logger.info("\n[步骤4] 保存分析结果")
    
    # 保存所有信号
    signals_df = pd.DataFrame(results['all_signals'])
    signals_df.to_csv('trading_signals.csv', index=False, encoding='utf-8')
    logger.info("  已保存: trading_signals.csv")
    
    # 保存带指标的数据
    df_with_indicators.to_csv('data_with_indicators.csv', index=False, encoding='utf-8')
    logger.info("  已保存: data_with_indicators.csv")
    
    # 保存交易记录
    if performance:
        trades_df = pd.DataFrame(performance['trades'])
        trades_df.to_csv('trade_records.csv', index=False, encoding='utf-8')
        logger.info("  已保存: trade_records.csv")
    
    logger.info("\n" + "=" * 80)
    logger.info("分析完成！")
    logger.info("=" * 80)
    
    return results, performance


if __name__ == '__main__':
    results, performance = main()
