import tkinter as tk
from tkinter import ttk
from ttkbootstrap import Style
from PIL import Image, ImageTk
import webbrowser

def open_youtube_video(url):
    webbrowser.open(url)

def create_workspace_screen():
    root = tk.Tk()
    root.title("BladeSense")
    root.state('zoomed')

    style = Style(theme='darkly')

    # Left Column Style
    style.configure("LeftColumn.TFrame", background="#343a40")
    style.configure("Workspace.TFrame", background="white")

    # Left Column (section)
    left_column = ttk.Frame(root, width=200, height=600, style="LeftColumn.TFrame")
    left_column.place(x=0, y=0)

    # Workspace Header (section)
    header_label = ttk.Label(root, text="Workspace", style="Header.TLabel", font=("Arial", 17))
    header_label.place(x=200, y=10)

    # Main Workspace (section)
    main_workspace = ttk.Frame(root, width=800, height=600, style="Workspace.TFrame")
    main_workspace.place(x=200, y=0)

    # Buttons in Left Column
    workspace_button = ttk.Button(left_column, text="Workspace", style="DarkButton.TButton", width=20)
    workspace_button.grid(row=0, column=0, padx=10, pady=10, sticky="w")

    history_button = ttk.Button(left_column, text="Your History", style="DarkButton.TButton", width=20)
    history_button.grid(row=1, column=0, padx=10, pady=10, sticky="w")

    documentation_button = ttk.Button(left_column, text="Documentation", style="DarkButton.TButton", width=20)
    documentation_button.grid(row=2, column=0, padx=10, pady=10, sticky="w")

    settings_button = ttk.Button(left_column, text="Settings", style="DarkButton.TButton", width=20)
    settings_button.grid(row=3, column=0, padx=10, pady=10, sticky="w")

    help_button = ttk.Button(left_column, text="Get Help", style="DarkButton.TButton", width=20)
    help_button.grid(row=4, column=0, padx=10, pady=10, sticky="w")

    root.mainloop()

if __name__ == "__main__":
    create_workspace_screen()
