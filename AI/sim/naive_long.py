from policy import Policy

def NaiveLong(Policy):
    def __init__(self, hist_window):
        super(NaiveLong, self).__init__(hist_window)
    
    def decide(self, hist, predict):
        return 1

long = NaiveLong(10)
print(long.hist_window)