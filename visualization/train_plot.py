import matplotlib.pyplot as plt
import numpy as np
from matplotlib.scale import SymmetricalLogTransform
from scipy.signal import savgol_filter

class TrainingPlotter:
    def __init__(self, window_size=1000, ema_alpha=0.05):
        """Initialize the training metrics plotter
        
        Args:
            window_size: Number of recent steps to display
            ema_alpha: EMA smoothing factor (1-decay)
        """
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.fig = None
        self.axes = None
        self.step_counter = 0
        
    def create_plot(self):
        """Create and initialize the subplot figure"""
        plt.ion()
        self.fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
        
        # Setup loss subplot
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Setup reward subplot
        ax2.set_title('Training Reward')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Reward')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('symlog', linthresh=1e-3)
        
        # Setup action subplot
        ax3.set_title('Agent Actions')
        ax3.set_xlabel('Training Steps')
        ax3.set_ylabel('Action')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-0.5, 0.5)  # Set fixed y-axis limits for actions
        
        # Setup account value subplot
        ax4.set_title('Account Value')
        ax4.set_xlabel('Training Steps')
        ax4.set_ylabel('Value ($)')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        # Create twin axis for stock price
        ax4_twin = ax4.twinx()
        
        return self.fig, (ax1, ax2, ax3, ax4, ax4_twin)
    
    def _calculate_ema(self, values):
        """Calculate exponential moving average"""
        if not values:
            return []
            
        emas = []
        ema = values[0]
        
        for value in values:
            ema = (1 - self.ema_alpha) * ema + self.ema_alpha * value
            emas.append(ema)
            
        return emas
    
    def update_plot(self, losses, rewards, actions, account_values, stock_prices=None):
        """Update all subplots with new data
        
        Args:
            losses: List of all loss values
            rewards: List of all reward values
            actions: List of all action values
            account_values: List of all account values
            stock_prices: List of stock prices
        """
        if self.fig is None:
            self.fig, (ax1, ax2, ax3, ax4, ax4_twin) = self.create_plot()
        else:
            ax1, ax2, ax3, ax4, ax4_twin = self.axes = self.fig.axes
        
        # Clear previous plots
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        ax4_twin.clear()
        
        # Get recent data
        record_length = len(losses)
        if record_length < self.window_size:
            recent_losses = losses
            recent_rewards = rewards
            recent_actions = actions
            recent_account_values = account_values
            stock_price_start = 0
            stock_price_end = record_length
        else:
            recent_losses = losses[-self.window_size:]
            recent_rewards = rewards[-self.window_size:]
            recent_actions = actions[-self.window_size:]
            recent_account_values = account_values[-self.window_size:]
            stock_price_start = record_length - self.window_size
            stock_price_end = record_length
        
        start_price = stock_prices[0]
        assert len(stock_prices) > 0, f"expected at least len {self.window_size} stock prices, but got {len(stock_prices)}"
        recent_stock_prices = stock_prices[stock_price_start:stock_price_end]
        
        # Calculate step numbers for x-axis
        current_step = len(losses)
        start_step = current_step - len(recent_losses)
        x_range = range(start_step, current_step)
        
        # Calculate EMAs
        loss_emas = self._calculate_ema(recent_losses)
        reward_emas = self._calculate_ema(recent_rewards)
        account_value_emas = self._calculate_ema(recent_account_values)
        stock_price_emas = self._calculate_ema(recent_stock_prices)

        # Plot loss
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        ax1.plot(x_range, recent_losses, 'b-', alpha=0.3, label='Raw Loss')
        ax1.plot(x_range, loss_emas, 'b-', label='EMA Loss')
        ax1.legend(loc='upper left')
        
        # Plot reward
        ax2.set_title('Training Reward')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Reward')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('symlog', linthresh=1e-3)
        ax2.axhline(y=0, color='r', linestyle='-', alpha=0.2)
        
        # Ensure the reward plot shows full range
        reward_min = min(min(recent_rewards), min(reward_emas))
        reward_max = max(max(recent_rewards), max(reward_emas))
        y_margin = (reward_max - reward_min) * 0.1
        ax2.set_ylim(reward_min - y_margin, reward_max + y_margin)
        
        ax2.plot(x_range, recent_rewards, 'g-', alpha=0.3, label='Raw Reward')
        ax2.plot(x_range, reward_emas, 'g-', label='EMA Reward')
        ax2.legend(loc='upper left')
        
        # Plot actions
        ax3.set_title('Agent Actions')
        ax3.set_xlabel('Training Steps')
        ax3.set_ylabel('Action')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-0.5, 0.5)
        
        # Plot actions as triangles
        for i, action in enumerate(recent_actions):
            x = start_step + i
            if action == 1:  # Buy action - upward triangle
                ax3.plot(x, 0.1, marker='^', markersize=8, color='limegreen', 
                        alpha=0.8, markeredgecolor='darkgreen', markeredgewidth=1)
            elif action == 2:  # Sell action - downward triangle (changed from -1 to 2)
                ax3.plot(x, -0.1, marker='v', markersize=8, color='red',
                        alpha=0.8, markeredgecolor='darkred', markeredgewidth=1)
            elif action == 0:  # Hold action - small dot
                ax3.plot(x, 0, marker='o', markersize=4, color='gray',
                        alpha=0.5, markeredgecolor='darkgray', markeredgewidth=1)
        
        # Plot account value vs stock price
        ax4.set_title('Account Value vs Stock Price')
        ax4.set_xlabel('Training Steps')
        ax4.set_ylabel('Account Value ($)', color='purple')
        ax4_twin.set_ylabel('Stock Price ($)', color='orange')
        
        # Plot the data
        ax4.plot(x_range, recent_account_values, 'purple', alpha=0.3, label='Raw Account Value')
        ax4.plot(x_range, account_value_emas, 'purple', label='EMA Account Value')            
        ax4_twin.plot(x_range, recent_stock_prices, 'orange', alpha=0.3, label='Raw Stock Price')
        ax4_twin.plot(x_range, stock_price_emas, 'orange', label='EMA Stock Price')

        # Calculate ranges and starting points
        account_start = recent_account_values[0]
        stock_start = recent_stock_prices[0]
        
        account_min = min(recent_account_values)
        account_max = max(recent_account_values)
        account_decrease_pct = account_min / account_start
        account_increase_pct = account_max / account_start
        
        stock_min = min(recent_stock_prices)
        stock_max = max(recent_stock_prices)
        stock_decrease_pct = stock_min / stock_start
        stock_increase_pct = stock_max / stock_start
        
        max_decrease_pct = max(account_decrease_pct, stock_decrease_pct)
        max_increase_pct = max(account_increase_pct, stock_increase_pct)
        
        # Calculate limits that maintain relative scale but align starting points
        account_y_min = account_start - max_decrease_pct * account_start
        account_y_max = account_start + max_increase_pct * account_start
        stock_y_min = stock_start - max_decrease_pct * stock_start
        stock_y_max = stock_start + max_increase_pct * stock_start
        
        # Set the limits
        ax4.set_ylim(account_y_min, account_y_max)
        ax4_twin.set_ylim(stock_y_min, stock_y_max)
        
        # Add legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Set the same x-axis limits for all plots
        ax1.set_xlim(start_step, current_step)
        ax2.set_xlim(start_step, current_step)
        ax3.set_xlim(start_step, current_step)
        ax4.set_xlim(start_step, current_step)
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Update display
        plt.draw()
        plt.pause(0.01)
    
    def reset(self):
        """Reset the plot"""
        self.axes = None
        self.step_counter = 0
    
    def close(self):
        """Close the plot"""
        plt.ioff()
        plt.show() 

class LossPlotter:
    def __init__(self, window_length=1001, poly_order=3):
        """Initialize the loss plotter with Savitzky-Golay filter parameters
        
        Args:
            window_length: Length of the smoothing window (must be odd)
            poly_order: Order of the polynomial used for smoothing
        """
        self.window_length = window_length
        self.poly_order = poly_order
        self.fig = None
        self.ax = None
        
    def update_plot(self, losses):
        """Update the loss plot with smoothed data"""
        if len(losses) < self.window_length:
            return
            
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(12, 6))
            
        self.ax.clear()
        
        # Apply Savitzky-Golay smoothing
        smoothed_losses = savgol_filter(losses, self.window_length, self.poly_order)
        
        # Plot both raw and smoothed data
        self.ax.plot(losses, 'b-', alpha=0.2, label='Raw Loss')
        self.ax.plot(smoothed_losses, 'r-', linewidth=2, label='Smoothed Loss')
        
        self.ax.set_title('Training Loss Over Time')
        self.ax.set_xlabel('Training Steps')
        self.ax.set_ylabel('Loss')
        self.ax.set_yscale('log')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
        
    def show_plot(self):
        """Show the final plot and keep it displayed"""
        if self.fig is not None:
            plt.ioff()
            plt.show() 