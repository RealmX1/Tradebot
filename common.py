from datetime import datetime

all_action_log_pth = 'log/all_action_log.txt'

def print_n_log(log_pth, *args, **kwargs):
    # Print to the terminal
    print(*args, **kwargs)
    
    # Log to the file
    if isinstance(log_pth, str):
        with open(log_pth, "a") as log_file:
            print(*args, **kwargs, file=log_file)
    elif isinstance(log_pth, list):
        for pth in log_pth:
            with open(pth, "a") as log_file:
                print(*args, **kwargs, file=log_file)
    elif log_pth is None:
        with open(all_action_log_pth, "a") as log_file:
            print(*args, **kwargs, file=log_file)
    else: # log_pth is not None, nor str, nor list
        raise TypeError("log_pth must be a string, a list of strings, or None")

def log_purpose(log_pth, type_str):
    while True:
        tolog = input(f"Do you want to log the {type_str} result? (y/n) ")
        if tolog == 'y':
            print_n_log(log_pth, f'{datetime.now()}')
            purpose = input(f"What is the purpose/target of this {type_str}ing? ")
            print_n_log(log_pth, f'purpose: {purpose}')
            return True
        elif tolog == 'n':
            return False

def log_review(log_pth, type_str):
    
    pass


def main():
    from datetime import datetime
    # forever logger;

    while True:
        action = input("Enter action: ")
        print_n_log(None, datetime.now())
        print_n_log(None, action)


if __name__ == '__main__':
    main()