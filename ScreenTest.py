import os
import pyautogui
import time
os.startfile("D:/SystemDesign.txt")
time.sleep(1)
myScreenshot = pyautogui.screenshot()
myScreenshot.save(r'D:/scr.png')