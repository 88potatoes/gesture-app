import pyautogui
import time # It's good practice to pause briefly after a move

# 1. Add a short delay (e.g., 2 seconds) so you have time 
#    to switch to the window where you run the script, or just to 
#    prepare. The script will wait before executing the move.
print("Starting mouse control in 2 seconds...")
time.sleep(2) 

pyautogui.moveTo(1000, 700, duration=0)
