import matplotlib.pyplot as plt
import numpy as np
from matplotlib.scale import SymmetricalLogTransform

class TrainingPlotter:
    def __init__(self, window_size=100, ema_alpha=0.05):
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
        ax4.grid(True, alpha=0.3)
        
        return self.fig, (ax1, ax2, ax3, ax4)
    
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
    
    def update_plot(self, losses, rewards, actions, account_values):
        """Update all subplots with new data
        
        Args:
            losses: List of all loss values
            rewards: List of all reward values
            actions: List of all action values
            account_values: List of all account values
        """
        if self.fig is None:
            self.fig, (ax1, ax2, ax3, ax4) = self.create_plot()
        else:
            ax1, ax2, ax3, ax4 = self.axes = self.fig.axes
        
        # Clear previous plots
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        
        # Get recent data
        recent_losses = losses[-self.window_size:] if len(losses) > self.window_size else losses
        recent_rewards = rewards[-self.window_size:] if len(rewards) > self.window_size else rewards
        recent_actions = actions[-self.window_size:] if len(actions) > self.window_size else actions
        recent_account_values = account_values[-self.window_size:] if len(account_values) > self.window_size else account_values
        
        # Calculate step numbers for x-axis
        current_step = len(losses)
        start_step = current_step - len(recent_losses)
        x_range = range(start_step, current_step)
        
        # Calculate EMAs
        loss_emas = self._calculate_ema(recent_losses)
        reward_emas = self._calculate_ema(recent_rewards)
        account_value_emas = self._calculate_ema(recent_account_values)
        
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
        
        # Plot account value
        ax4.set_title('Account Value')
        ax4.set_xlabel('Training Steps')
        ax4.set_ylabel('Value ($)')
        ax4.grid(True, alpha=0.3)
        ax4.plot(x_range, recent_account_values, 'purple', alpha=0.3, label='Raw Value')
        ax4.plot(x_range, account_value_emas, 'purple', label='EMA Value')
        ax4.legend(loc='upper left')
        
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
    
    def close(self):
        """Close the plot"""
        plt.ioff()
        plt.show() 