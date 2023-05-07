def print_n_log(log_pth, *args, **kwargs):
    # Print to the terminal
    print(*args, **kwargs)
    
    # Log to the file
    with open(log_pth, "a") as log_file:
        print(*args, **kwargs, file=log_file)

def main():
    log_pth = "log.txt"
    print_n_log(log_pth, "Hello World!")

if __name__ == '__main__':
    main()