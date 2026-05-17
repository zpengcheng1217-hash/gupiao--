"""
股票数据分析 - 买卖点识别系统
分析000002.sz股票，识别最佳买进点和卖出时间
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class StockAnalyzer:
    """股票分析器 - 识别买卖点"""
    
    def __init__(self, csv_file):
        """初始化分析器"""
        self.df = pd.read_csv(csv_file)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        self.buy_signals = []
        self.sell_signals = []
        
    def calculate_indicators(self):
        """计算所有技术指标"""
        
        # 1. 移动平均线
        self.df['MA5'] = self.df['close'].rolling(window=5).mean()
        self.df['MA10'] = self.df['close'].rolling(window=10).mean()
        self.df['MA20'] = self.df['close'].rolling(window=20).mean()
        self.df['MA50'] = self.df['close'].rolling(window=50).mean()
        
        # 2. RSI (相对强弱指数)
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        # 3. MACD
        ema_12 = self.df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = self.df['close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = ema_12 - ema_26
        self.df['Signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        self.df['Histogram'] = self.df['MACD'] - self.df['Signal']
        
        # 4. 布林线
        self.df['BB_Middle'] = self.df['close'].rolling(window=20).mean()
        bb_std = self.df['close'].rolling(window=20).std()
        self.df['BB_Upper'] = self.df['BB_Middle'] + (bb_std * 2)
        self.df['BB_Lower'] = self.df['BB_Middle'] - (bb_std * 2)
        
        # 5. 换手率 (成交量与流通股本关系)
        self.df['Volume_MA'] = self.df['volume'].rolling(window=5).mean()
        self.df['Volume_Ratio'] = self.df['volume'] / self.df['Volume_MA']
        
        # 6. 涨跌幅
        self.df['pct_change'] = self.df['close'].pct_change() * 100
        
        # 7. 日期和价格变化
        self.df['price_change'] = self.df['close'].diff()
        
        return self.df
    
    def identify_buy_signals(self):
        """识别买进信号"""
        
        signals = []
        
        for i in range(50, len(self.df)):
            current = self.df.iloc[i]
            
            # 买进条件组合
            conditions = {
                'condition1_ma_crossover': False,  # MA5穿过MA10
                'condition2_rsi_oversold': False,  # RSI超卖
                'condition3_macd_bullish': False,  # MACD底部上升
                'condition4_support': False,  # 价格接近支撑位
                'condition5_volume': False,  # 成交量放大
            }
            
            try:
                # 条件1: MA5穿过MA10（金叉）
                if i > 0:
                    prev = self.df.iloc[i-1]
                    if prev['MA5'] <= prev['MA10'] and current['MA5'] > current['MA10']:
                        conditions['condition1_ma_crossover'] = True
                
                # 条件2: RSI < 30 (超卖)
                if current['RSI'] < 30:
                    conditions['condition2_rsi_oversold'] = True
                
                # 条件3: MACD底部上升（负值变为正值或低点上升）
                if i > 0:
                    prev = self.df.iloc[i-1]
                    if prev['Histogram'] < 0 and current['Histogram'] > 0:
                        conditions['condition3_macd_bullish'] = True
                    elif current['Histogram'] > 0 and current['Histogram'] > prev['Histogram']:
                        conditions['condition3_macd_bullish'] = True
                
                # 条件4: 价格接近布林线下轨（接近支撑位）
                if current['close'] < current['BB_Middle'] * 0.98:
                    conditions['condition4_support'] = True
                
                # 条件5: 成交量放大
                if current['Volume_Ratio'] > 1.2:
                    conditions['condition5_volume'] = True
                
                # 综合判断：满足至少3个条件
                score = sum(conditions.values())
                
                if score >= 3:
                    signal = {
                        'date': current['date'],
                        'price': current['close'],
                        'score': score,
                        'conditions': conditions,
                        'ma5': current['MA5'],
                        'ma20': current['MA20'],
                        'rsi': current['RSI'],
                        'macd': current['MACD'],
                        'volume': current['volume'],
                    }
                    signals.append(signal)
            
            except Exception as e:
                continue
        
        self.buy_signals = signals
        return signals
    
    def identify_sell_signals(self):
        """识别卖出信号"""
        
        signals = []
        
        for i in range(50, len(self.df)):
            current = self.df.iloc[i]
            
            # 卖出条件组合
            conditions = {
                'condition1_ma_death_cross': False,  # MA5穿过MA10（死叉）
                'condition2_rsi_overbought': False,  # RSI超买
                'condition3_macd_bearish': False,  # MACD顶部下降
                'condition4_resistance': False,  # 价格接近阻力位
                'condition5_high_volume': False,  # 成交量异常高
                'condition6_profit_taking': False,  # 涨幅过大
            }
            
            try:
                # 条件1: MA5穿过MA10（死叉）
                if i > 0:
                    prev = self.df.iloc[i-1]
                    if prev['MA5'] >= prev['MA10'] and current['MA5'] < current['MA10']:
                        conditions['condition1_ma_death_cross'] = True
                
                # 条件2: RSI > 70 (超买)
                if current['RSI'] > 70:
                    conditions['condition2_rsi_overbought'] = True
                
                # 条件3: MACD顶部下降（正值变为负值或高点下降）
                if i > 0:
                    prev = self.df.iloc[i-1]
                    if prev['Histogram'] > 0 and current['Histogram'] < 0:
                        conditions['condition3_macd_bearish'] = True
                    elif current['Histogram'] > 0 and current['Histogram'] < prev['Histogram']:
                        conditions['condition3_macd_bearish'] = True
                
                # 条件4: 价格接近布林线上轨（接近阻力位）
                if current['close'] > current['BB_Middle'] * 1.02:
                    conditions['condition4_resistance'] = True
                
                # 条件5: 成交量异常高
                if current['Volume_Ratio'] > 1.5:
                    conditions['condition5_high_volume'] = True
                
                # 条件6: 涨幅过大（3天内涨幅>5%）
                if i >= 3:
                    three_days_ago = self.df.iloc[i-3]['close']
                    pct = (current['close'] - three_days_ago) / three_days_ago * 100
                    if pct > 5:
                        conditions['condition6_profit_taking'] = True
                
                # 综合判断：满足至少2个条件
                score = sum(conditions.values())
                
                if score >= 2:
                    signal = {
                        'date': current['date'],
                        'price': current['close'],
                        'score': score,
                        'conditions': conditions,
                        'ma5': current['MA5'],
                        'ma20': current['MA20'],
                        'rsi': current['RSI'],
                        'macd': current['MACD'],
                        'volume': current['volume'],
                    }
                    signals.append(signal)
            
            except Exception as e:
                continue
        
        self.sell_signals = signals
        return signals
    
    def generate_report(self):
        """生成分析报告"""
        
        print("=" * 80)
        print("股票分析报告: 000002.sz (万科A)")
        print("=" * 80)
        
        # 基本信息
        print("\n【基本信息】")
        print(f"数据周期: {self.df['date'].min().date()} 至 {self.df['date'].max().date()}")
        print(f"交易天数: {len(self.df)} 天")
        print(f"当前价格: {self.df.iloc[-1]['close']:.2f} 元")
        print(f"最高价: {self.df['close'].max():.2f} 元")
        print(f"最低价: {self.df['close'].min():.2f} 元")
        print(f"平均价: {self.df['close'].mean():.2f} 元")
        
        # 技术指标
        print("\n【技术指标（最新）】")
        latest = self.df.iloc[-1]
        print(f"MA5: {latest['MA5']:.2f}")
        print(f"MA10: {latest['MA10']:.2f}")
        print(f"MA20: {latest['MA20']:.2f}")
        print(f"MA50: {latest['MA50']:.2f}")
        print(f"RSI(14): {latest['RSI']:.2f}")
        print(f"MACD: {latest['MACD']:.4f}")
        print(f"MACD Signal: {latest['Signal']:.4f}")
        print(f"MACD Histogram: {latest['Histogram']:.4f}")
        
        # 买进信号
        print("\n" + "=" * 80)
        print(f"【买进信号】共识别到 {len(self.buy_signals)} 个买进点")
        print("=" * 80)
        
        if self.buy_signals:
            for idx, signal in enumerate(self.buy_signals[-10:], 1):  # 显示最后10个
                print(f"\n买进信号 #{idx}")
                print(f"  日期: {signal['date'].strftime('%Y-%m-%d')}")
                print(f"  价格: {signal['price']:.2f} 元")
                print(f"  信号强度: {'★' * signal['score']}")
                print(f"  条件说明:")
                for cond, value in signal['conditions'].items():
                    status = "✓" if value else "✗"
                    print(f"    {status} {cond}")
        else:
            print("暂未发现明确的买进信号")
        
        # 卖出信号
        print("\n" + "=" * 80)
        print(f"【卖出信号】共识别到 {len(self.sell_signals)} 个卖出点")
        print("=" * 80)
        
        if self.sell_signals:
            for idx, signal in enumerate(self.sell_signals[-10:], 1):  # 显示最后10个
                print(f"\n卖出信号 #{idx}")
                print(f"  日期: {signal['date'].strftime('%Y-%m-%d')}")
                print(f"  价格: {signal['price']:.2f} 元")
                print(f"  信号强度: {'★' * signal['score']}")
                print(f"  条件说明:")
                for cond, value in signal['conditions'].items():
                    status = "✓" if value else "✗"
                    print(f"    {status} {cond}")
        else:
            print("暂未发现明确的卖出信号")
        
        # 建议
        print("\n" + "=" * 80)
        print("【投资建议】")
        print("=" * 80)
        
        if len(self.buy_signals) == 0 and len(self.sell_signals) == 0:
            print("⚠ 当前没有明确的买卖信号，建议继续观察")
        else:
            last_buy = self.buy_signals[-1] if self.buy_signals else None
            last_sell = self.sell_signals[-1] if self.sell_signals else None
            
            if last_buy and last_sell:
                if last_buy['date'] > last_sell['date']:
                    print(f"✓ 最近一个买进信号较新 ({last_buy['date'].strftime('%Y-%m-%d')})")
                    print(f"  建议考虑在 {last_buy['price']:.2f} 元附近建仓")
                else:
                    print(f"✗ 最近一个卖出信号较新 ({last_sell['date'].strftime('%Y-%m-%d')})")
                    print(f"  建议在 {last_sell['price']:.2f} 元附近减仓或离场")
            elif last_buy:
                print(f"✓ 最近一个买进信号: {last_buy['date'].strftime('%Y-%m-%d')}")
                print(f"  建议可以逐步建仓")
            elif last_sell:
                print(f"✗ 最近一个卖出信号: {last_sell['date'].strftime('%Y-%m-%d')}")
                print(f"  建议提高警惕，考虑减仓")
        
        print("\n【止损建议】")
        print("- 下跌2-3%时设置止损")
        print("- 保护已有利润")
        
        print("\n【获利建议】")
        print("- 当收益率达到5-10%时，可以考虑分批获利")
        print("- 不要贪心，见好就收")
        
        print("\n" + "=" * 80)
    
    def export_signals_to_csv(self, output_file='analysis_result.csv'):
        """导出信号到CSV文件"""
        
        all_signals = []
        
        # 添加买进信号
        for signal in self.buy_signals:
            all_signals.append({
                '日期': signal['date'].strftime('%Y-%m-%d'),
                '类型': '买进',
                '价格': signal['price'],
                '强度': signal['score'],
                'MA5': signal['ma5'],
                'MA20': signal['ma20'],
                'RSI': signal['rsi'],
                'MACD': signal['macd'],
                '成交量': signal['volume'],
            })
        
        # 添加卖出信号
        for signal in self.sell_signals:
            all_signals.append({
                '日期': signal['date'].strftime('%Y-%m-%d'),
                '类型': '卖出',
                '价格': signal['price'],
                '强度': signal['score'],
                'MA5': signal['ma5'],
                'MA20': signal['ma20'],
                'RSI': signal['rsi'],
                'MACD': signal['macd'],
                '成交量': signal['volume'],
            })
        
        # 按日期排序
        df_signals = pd.DataFrame(all_signals)
        df_signals['日期'] = pd.to_datetime(df_signals['日期'])
        df_signals = df_signals.sort_values('日期').reset_index(drop=True)
        df_signals['日期'] = df_signals['日期'].dt.strftime('%Y-%m-%d')
        
        # 保存
        df_signals.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✓ 分析结果已导出到: {output_file}")


def main():
    """主函数"""
    
    try:
        # 创建分析器
        analyzer = StockAnalyzer('000002.sz.csv')
        
        # 计算指标
        print("正在计算技术指标...")
        analyzer.calculate_indicators()
        
        # 识别买进点
        print("正在识别买进点...")
        buy_signals = analyzer.identify_buy_signals()
        
        # 识别卖出点
        print("正在识别卖出点...")
        sell_signals = analyzer.identify_sell_signals()
        
        # 生成报告
        analyzer.generate_report()
        
        # 导出结果
        analyzer.export_signals_to_csv()
        
    except FileNotFoundError:
        print("❌ 错误: 找不到 000002.sz.csv 文件")
        print("请确保 000002.sz.csv 文件在当前目录中")
    except Exception as e:
        print(f"❌ 错误: {str(e)}")


if __name__ == '__main__':
    main()
