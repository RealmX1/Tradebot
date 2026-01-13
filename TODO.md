# TODO List - TradeBot Development

## High Priority

### Data & Preprocessing
- [ ] **Improve State Normalization** - The state printed at data_loader "prepare_state" needs better normalization, especially for stock close prices
- [ ] **Verify FRED Data Timing** - Check whether FRED timestamps represent update time or observation time to avoid potential look-ahead bias
- [ ] **Add Industry Features** - Include stock industry sector as one-hot encoded input features

### Model Architecture
- [ ] **Scale Up Model** - Increase model capacity (more layers/neurons) and evaluate performance improvements
- [ ] **Implement LSTM/GRU** - Test recurrent architectures for better temporal pattern recognition
- [ ] **Multi-timeframe Analysis** - Incorporate multiple timeframes (1min, 5min, hourly) for better context

## Medium Priority

### Trading Simulation
- [ ] **Add Slippage Modeling** - Implement realistic slippage based on order size and market liquidity
- [ ] **Commission Structure** - Add configurable commission models (per-share, per-trade, tiered)
- [ ] **Market Impact** - Model price impact for large orders
- [ ] **Extended Trading Hours** - Support pre-market and after-hours trading

### Risk Management
- [ ] **Position Sizing** - Implement Kelly Criterion or other position sizing algorithms
- [ ] **Stop Loss/Take Profit** - Add configurable stop-loss and take-profit orders
- [ ] **Maximum Drawdown Limits** - Implement circuit breakers for excessive losses
- [ ] **Portfolio Diversification** - Add constraints for sector/stock concentration

## Low Priority

### Performance & Optimization
- [ ] **Batch Training Speed** - Optimize gradient updates to process more than 60 rows/second
- [ ] **Distributed Training** - Implement multi-GPU training for faster convergence
- [ ] **Hyperparameter Tuning** - Automated hyperparameter optimization using Optuna or similar

### Features & Analysis
- [ ] **Sentiment Analysis** - Integrate news/social media sentiment (partial code in sentiment_analysis/)
- [ ] **Options Trading** - Extend to options strategies
- [ ] **Pairs Trading** - Implement statistical arbitrage strategies
- [ ] **Market Regime Detection** - Adapt strategy based on market conditions (bull/bear/sideways)

### Infrastructure
- [ ] **Real-time Trading** - Production-ready live trading with proper error handling
- [ ] **Backtesting Framework** - More comprehensive backtesting with walk-forward analysis
- [ ] **Performance Dashboard** - Web-based dashboard for monitoring live performance
- [ ] **Cloud Deployment** - Containerization and cloud deployment guides

## Completed
- [x] Revamp indicator generation using `stock_indicators` library
- [x] Overhaul trade simulation process with realistic order execution
- [x] Implement DQN model with experience replay
- [x] Add visualization for training and testing
- [x] Integrate FRED macroeconomic data

## Notes
- Current training speed: ~60 rows/second (~1.6 seconds per 100 rows)
- Latest model shows promise but needs further optimization for production use
- Consider implementing ensemble methods combining multiple models