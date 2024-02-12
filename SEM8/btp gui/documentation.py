from tkinter import *
from tkPDFViewer import tkPDFViewer as pdf
import webbrowser
from tkinter import ttk
from ttkbootstrap import Style
from PIL import Image, ImageTk
import os
from tkinter import Tk
from pdfviewer import PDFViewer


def open_youtube_video(url):
    webbrowser.open(url)


def open_pdf_documentation(pdf_path):
    root = Tk()
    root.title("BladeSense Documentation")
    root.geometry("900x600")
    PDFViewer()
    root.mainloop()


def create_workspace_screen():
    root = Tk()
    root.title("BladeSense")
    root.geometry("900x600")

    style = Style(theme='darkly')

    # Left Column Style (Dark Color)
    style.configure("LeftColumn.TFrame", background="#343a40")  # Dark background color

    # Workspace Style (Light Color)
    style.configure("Workspace.TFrame", background="white")  # Light background color

    # Top Left Logo and Software Name
    logo_image = Image.open("logo.png")  # Replace "logo.png" with your logo file
    logo_image = logo_image.resize((100, 40), Image.Resampling.LANCZOS)
    logo_photo = ImageTk.PhotoImage(logo_image)
    logo_label = ttk.Label(root, image=logo_photo)
    logo_label.image = logo_photo
    logo_label.place(x=15, y=10)  # Position at x=10, y=10

    # Left Column with Buttons
    left_column = ttk.Frame(root, width=200, height=600, style="TFrame")
    left_column.place(x=0, y=60)  # Position at x=10, y=60

    # Calculate button width
    button_width = 19  # Adjust as needed

    # Create and place buttons
    workspace_button = ttk.Button(left_column, text="Workspace", style="DarkButton.TButton", width=button_width)
    workspace_button.grid(row=0, column=0, padx=10, pady=10, sticky="w")

    history_button = ttk.Button(left_column, text="Your History", style="DarkButton.TButton", width=button_width)
    history_button.grid(row=1, column=0, padx=10, pady=10, sticky="w")

    documentation_button = ttk.Button(left_column, text="Documentation", style="DarkButton.TButton", width=button_width,
                                      command=lambda: PDFViewer())
    documentation_button.grid(row=2, column=0, padx=10, pady=10, sticky="w")

    settings_button = ttk.Button(text="Settings", style="DarkButton.TButton", width=button_width)
    settings_button.place(x=10, y=520)  # Position at x=0, y=500

    help_button = ttk.Button(text="Get Help", style="DarkButton.TButton", width=button_width)
    help_button.place(x=10, y=560)  # Position at x=0, y=500

    # Draw a vertical line after the left column
    vertical_line = ttk.Separator(root, orient="vertical")
    vertical_line.place(x=160, y=0, relheight=600)  # Adjust x-coordinate as needed

    root.mainloop()


if __name__ == "__main__":
    create_workspace_screen()
