from AI.tester import Tester

def main():
    config = {
        'test_dir': 'data/test',
        'fred_data_path': 'data/fred/normalized_fred_data.csv',
        'initial_cash': 100000,
        'model_dir': 'AI/models'
    }
    
    tester = Tester(config)
    results = tester.test()

if __name__ == "__main__":
    main() 