from typing import Dict, Any
from pathlib import Path
from AI.trainer import Trainer

class FineTuner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def fine_tune(self, symbol: str):
        """Fine-tune the base model for a specific symbol."""
        print(f"\nFine-tuning model for {symbol}")
        
        # Create a new trainer instance in fine-tuning mode
        fine_tune_config = self.config.copy()
        # Typically use fewer episodes for fine-tuning
        fine_tune_config['episodes'] = fine_tune_config.get('fine_tune_episodes', 50)
        
        trainer = Trainer(fine_tune_config, fine_tune_symbol=symbol)
        trainer.train()
        
        print(f"Fine-tuning completed for {symbol}")
        
    def fine_tune_all(self):
        """Fine-tune the base model for all available symbols."""
        # Load one symbol to get available symbols
        temp_trainer = Trainer(self.config)
        available_symbols = list(temp_trainer.all_data.keys())
        
        for symbol in available_symbols:
            self.fine_tune(symbol)
