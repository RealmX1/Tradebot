import tkinter as tk
from tkinter import ttk
import time
import threading
from playsound import playsound

notification_sound_file = 'notification_sound.mp3' # replace with your own file

def countdown_timer():
    remaining_time = 3 # 30 minutes in seconds
    while remaining_time >= 0:
        minutes, seconds = divmod(remaining_time, 60)
        time_str = f'{minutes:02d}:{seconds:02d}'
        label.config(text=time_str)
        window.update()
        if remaining_time == 0:
            print("Time's up!")
            play_notification_sound()
            reset_button.pack()
        time.sleep(1)
        remaining_time -= 1

def play_notification_sound():
    playsound(notification_sound_file)

def reset_timer():
    reset_button.pack_forget()
    timer_thread = threading.Thread(target=countdown_timer)
    timer_thread.start()

# Create the main window
window = tk.Tk()
window.title("30 Minute Timer")
window.geometry("200x100")

# Add a label to display the remaining time
label = ttk.Label(window, text="00:00", font=("Helvetica", 24))
label.pack(pady=10)

# Add a button to reset the timer
reset_button = ttk.Button(window, text="Reset Timer", command=reset_timer)

# Start the timer on a separate thread to keep the GUI responsive
timer_thread = threading.Thread(target=countdown_timer)
timer_thread.start()

# Run the main loop
window.mainloop()