# main.py
import tkinter as tk
from MainWindow import MainWindow  # Import the MyApp class from ui.py

def run_app():
    root = tk.Tk()
    app = MainWindow(root)  # Pass root to MyApp class
    root.mainloop()

if __name__ == "__main__":
    run_app()