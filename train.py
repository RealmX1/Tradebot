from AI.trainer import Trainer
from AI.fine_tuner import FineTuner
def main():
    config = {
        'train_dir': "data/train",
        'fred_data_path': "data/fred/normalized_fred_data.csv",
        'initial_cash': 100000,
        'episodes': 1000,
        'fine_tune_episodes': 2,
        'target_update_freq': 10,
        'save_freq': 10,
        'model_dir': "AI/models",
    }
    
    trainer = Trainer(config)
    trainer.train()
    
    # fine_tuner = FineTuner(config)
    # fine_tuner.fine_tune_all()

if __name__ == "__main__":
    main() 