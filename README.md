# TradeBot - Reinforcement Learning Stock Trading System

## Overview
TradeBot is an advanced automated stock trading system that uses Deep Q-Network (DQN) reinforcement learning to make trading decisions. The system combines technical indicators, macroeconomic data from FRED (Federal Reserve Economic Data), and price action to train intelligent trading agents capable of learning optimal trading strategies.

## Features
- **Deep Q-Network (DQN) Agent**: Neural network-based trading agent with experience replay and target network for stable learning
- **Comprehensive Feature Engineering**: Integration of 50+ technical indicators using the `stock-indicators` library
- **Macroeconomic Data Integration**: Incorporates FRED data for broader market context
- **Realistic Trading Simulation**: Includes realistic constraints like failed orders when price limits aren't met
- **Multi-Stock Support**: Can train and test on multiple stocks from the S&P 500
- **Fine-Tuning Capability**: Supports fine-tuning pre-trained models on specific stocks
- **Visualization Tools**: Comprehensive plotting for training progress and test results

## Project Structure
```
Tradebot/
├── AI/                         # Core AI/ML components
│   ├── model.py               # DQN network architecture and agent
│   ├── trainer.py             # Training logic
│   ├── fine_tuner.py          # Fine-tuning for specific stocks
│   ├── tester.py              # Model testing and evaluation
│   └── models/                # Saved model checkpoints
├── data_util/                  # Data fetching and processing
│   ├── alpaca_api/            # Alpaca API integration
│   ├── data_loader.py        # Data loading and preprocessing
│   ├── yfinance_history.py   # Yahoo Finance data fetching
│   └── symbols.py             # Stock symbol management
├── feature_engineering/        # Technical indicator generation
│   ├── indicators.py          # Indicator calculation pipeline
│   └── stock_indicators_documentation.csv  # Indicator reference
├── fred_api/                   # Federal Reserve data integration
│   └── prepare_fred_data.py  # FRED data processing
├── sim/                        # Trading simulation environment
│   ├── simulator.py           # Main simulation engine
│   ├── account.py             # Account management and order execution
│   └── reward.py              # Reward calculation logic
├── visualization/              # Plotting and visualization
│   ├── train_plot.py          # Training progress visualization
│   └── test_plot.py           # Test results visualization
├── data/                       # Data storage
│   ├── train/                 # Training data
│   ├── test/                  # Test data
│   └── fred/                  # FRED economic data
└── results/                    # Output and results
    └── test_plot/             # Test visualization outputs
```

## Installation

### Prerequisites
- Python 3.12+
- CUDA-capable GPU (optional but recommended for faster training)
- Conda or virtualenv

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Tradebot.git
   cd Tradebot
   ```

2. **Create and activate a conda environment**
   ```bash
   conda create -n trading_env python=3.12
   conda activate trading_env
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Note: For GPU support with PyTorch, you may need to install the CUDA-specific version:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Usage

### 1. Data Preparation
First, prepare the training and test data by fetching historical stock prices, calculating technical indicators, and downloading FRED economic data:

```bash
python prepare_data.py
```

This will:
- Download historical stock data for S&P 500 companies
- Calculate technical indicators for each stock
- Fetch and process FRED macroeconomic data
- Save processed data to `data/train/` and `data/test/` directories

### 2. Training the Model
Train the DQN agent on the prepared data:

```bash
python train.py
```

Training configuration can be modified in `train.py`:
- `episodes`: Number of training episodes (default: 1000)
- `initial_cash`: Starting capital for each episode (default: $100,000)
- `target_update_freq`: Frequency of target network updates
- `save_freq`: Model checkpoint saving frequency

The training process will:
- Train on multiple stocks simultaneously
- Save model checkpoints every N episodes
- Display training progress and rewards
- Generate training visualization plots

### 3. Testing the Model
Evaluate the trained model on unseen test data:

```bash
python test.py
```

This will:
- Load the latest trained model
- Run backtests on test data (different time period)
- Generate performance visualizations
- Output results to `results/test_plot/` directory

### 4. Fine-Tuning (Optional)
Fine-tune a pre-trained model on specific stocks:

```python
# Uncomment in train.py
fine_tuner = FineTuner(config)
fine_tuner.fine_tune_all()
```

## Configuration

### Model Hyperparameters
Key hyperparameters can be adjusted in `AI/model.py`:
- `hidden_size`: Neural network hidden layer size (default: 64)
- `learning_rate`: Learning rate for Adam optimizer (default: 1e-5)
- `gamma`: Discount factor for future rewards (default: 0.99)
- `epsilon_start/end/decay`: Exploration parameters
- `memory_size`: Experience replay buffer size (default: 10,000)
- `batch_size`: Training batch size (default: 64)

### Trading Parameters
Trading simulation parameters in `sim/simulator.py`:
- Order types: Market orders with realistic execution
- Position sizing: Configurable based on account balance
- Risk management: Built-in position limits and cash management

## Key Components

### DQN Agent
The core trading agent uses a Deep Q-Network with:
- 3-layer fully connected neural network
- Experience replay for stable learning
- Target network for reducing overestimation
- Epsilon-greedy exploration strategy

### State Representation
Each state includes:
- Technical indicators (50+ features)
- Price action data (OHLCV)
- Account information (cash, positions, pending orders)
- Macroeconomic indicators from FRED

### Action Space
- **0**: Hold (no action)
- **1**: Buy (place buy order)
- **2**: Sell (place sell order)

### Reward Function
The reward function considers:
- Realized P&L from closed positions
- Unrealized P&L from open positions
- Transaction costs
- Risk-adjusted returns

## Performance Monitoring

Training progress is monitored through:
- Episode rewards
- Portfolio value over time
- Win rate and average trade metrics
- Sharpe ratio and other risk metrics

Test results include:
- Cumulative returns vs buy-and-hold
- Individual stock performance charts
- Trade execution visualization
- Statistical performance metrics

## Known Issues and TODOs

Current development priorities (from TODO.md):
1. Improve normalization of state features for better model convergence
2. Verify FRED data timing to avoid look-ahead bias
3. Add industry sector as one-hot encoded feature
4. Implement slippage modeling for more realistic simulations

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is for educational and research purposes. Use at your own risk for actual trading.

## Disclaimer

**IMPORTANT**: This software is for educational purposes only. Do not use for actual trading without thorough testing and validation. The authors are not responsible for any financial losses incurred through the use of this software. Always consult with financial professionals before making investment decisions.

## Acknowledgments

- Technical indicators powered by `stock-indicators` library
- Market data from Yahoo Finance and Alpaca Markets
- Economic data from Federal Reserve Economic Data (FRED)