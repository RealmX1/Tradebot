import time
from collections import defaultdict

class TimingStats:
    def __init__(self):
        self.times = defaultdict(float)  # Total accumulated time
        self.counts = defaultdict(int)   # Count of measurements
        self.last_print = time.time()
        self.print_interval = 2  # seconds between prints
        self.start_time = time.time()
        self.new_episode = True
        self.total_time = 0

    def update(self, section: str, duration: float):
        self.times[section] += duration
        self.counts[section] += 1
        self.total_time += duration

    def print_stats(self, force=False):
        current_time = time.time()
        if not force and (current_time - self.last_print) < self.print_interval:
            return

        self.last_print = current_time
        elapsed_time = current_time - self.start_time
        
        # Clear previous lines if not a new episode
        if not self.new_episode:
            print('\033[F' * (len(self.times) + 3))
        else:
            self.new_episode = False
        
        print(f"\nTiming Statistics (Total time: {elapsed_time:.1f}s):")
        for section, t in sorted(self.times.items()):
            percentage = (t / self.total_time * 100)
            avg_time = (t * 1000 / self.counts[section]) if self.counts[section] > 0 else 0
            print(f"{section:<20}: {t*1000:>8.2f}ms total, {avg_time:>8.2f}ms avg ({percentage:>6.2f}%)")
    
    def reset(self):
        self.new_episode = True
        self.total_time = 0
        self.times = defaultdict(float)
        self.counts = defaultdict(int)
        self.last_print = time.time()
        self.start_time = time.time() 