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

def main():
    from datetime import datetime
    # forever logger;

    while True:
        action = input("Enter action: ")
        print_n_log(None, datetime.now())
        print_n_log(None, action)


if __name__ == '__main__':
    main()