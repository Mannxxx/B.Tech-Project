import tkinter as tk
from tkinter import ttk
from ttkbootstrap import Style
from PIL import Image, ImageTk
from tkinter import OptionMenu
import tkinter.font as tkFont
from tkinter import PhotoImage
import webbrowser

def open_youtube_video(url):
    webbrowser.open(url)

# Set the height of the buttons
button_height = 30  # Adjust as needed

def create_workspace_screen():
    root = tk.Tk()
    root.title("BladeSense")
    root.geometry("900x600")
    
    style = Style(theme='darkly')

    # Left Column Style (Dark Color)
    style.configure("LeftColumn.TFrame", background="#343a40")  # Dark background color
    
    # Workspace Style (Light Color)
    style.configure("Workspace.TFrame", background="white")  # Light background color
    
    
    # Top Left Logo and Software Name
    logo_image = Image.open("small_logo.png")  # Replace "logo.png" with your logo file
    logo_image = logo_image.resize((20, 25), Image.Resampling.LANCZOS)
    logo_photo = ImageTk.PhotoImage(logo_image)
    logo_label = ttk.Label(root, image=logo_photo)
    logo_label.image = logo_photo
    # logo_label.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
    logo_label.place(x=15, y=10)  # Position at x=10, y=10
    
    # Left Column with Buttons
    left_column = ttk.Frame(root, width=50, height=600, style="TFrame")
    # left_column.grid(row=1, column=0, rowspan=2, padx=10, pady=10, sticky="nw")
    left_column.place(x=0, y=60)  # Position at x=10, y=60
    
    # SecondLeft Column with Buttons
    left_column = ttk.Frame(root, width=200, height=600, style="TFrame")
    # left_column.grid(row=1, column=0, rowspan=2, padx=10, pady=10, sticky="nw")
    left_column.place(x=50, y=60)  # Position at x=10, y=60

    # Calculate button width
    button_width = 2  # Adjust as needed
    
    # Create and place buttons
    workspace_button = ttk.Button(text="W", style="DarkButton.TButton", width=button_width)
    workspace_button.place(x=10, y=60)
    
    history_button = ttk.Button(text="H", style="DarkButton.TButton", width=button_width)
    history_button.place(x=10, y=100)
    
    documentation_button = ttk.Button(text="D", style="DarkButton.TButton", width=button_width)
    documentation_button.place(x=10, y=140)

    settings_button = ttk.Button(text="S", style="DarkButton.TButton", width=button_width)
    settings_button.place(x=10, y=520)

    help_button = ttk.Button(text="H", style="DarkButton.TButton", width=button_width)
    help_button.place(x=10, y=560)  # Position at x=0, y=500
    
    # Draw a vertical line after the left column
    vertical_line = ttk.Separator(root, orient="vertical")
    vertical_line.place(x=50, y=0, relheight=600)  # Adjust x-coordinate as needed

    # Draw a vertical line after the left column
    vertical_line = ttk.Separator(root, orient="vertical")
    vertical_line.place(x=250, y=50, relheight=600)  # Adjust x-coordinate as needed

    # Workspace Header
    header_label = ttk.Label(root, text="Workspace", style="Header.TLabel", font=tkFont.Font(size=17))
    header_label.place(x=70, y=10)  # Position at x=250, y=10

    headerbar_label = ttk.Label(root, text="Workspace", style="Header.TLabel", font=tkFont.Font(size=17))
    headerbar_label.place(x=70, y=10)  # Position at x=250, y=10 

    # header_label.grid(row=0, column=1, padx=10, pady=10, sticky="nw")
    # header_label = ttk.Label(root, text="\nAll workspace you have (0)", style="Header.TLabel", font=tkFont.Font(size=11))
    # header_label.grid(row=1, column=1, padx=10, pady=10, sticky="nw")

    # Define coordinates
    x_start = 630
    y_start = 40
    button_width = 120  
    spacing = 5

    # Load .ico icons
    # search_icon = Image.open("search.ico")
    notifications_icon = Image.open("notification.ico")
    userImage_icon=Image.open("user.ico")

    # Resize icons
    # search_icon = search_icon.resize((20, 20), Image.Resampling.LANCZOS)
    notifications_icon = notifications_icon.resize((15, 15), Image.Resampling.LANCZOS)
    userImage_icon=userImage_icon.resize((16,16),Image.Resampling.LANCZOS)

    # Convert to PhotoImage
    # search_icon = ImageTk.PhotoImage(search_icon)
    notifications_icon = ImageTk.PhotoImage(notifications_icon)
    userImage_icon=ImageTk.PhotoImage(userImage_icon)


    # Navbar
    search_button = ttk.Button(root, text="Search Workspaces", compound="left", style="NavBarButton.TButton")
    search_button.place(x=640, y=10)  # Position at x=600, y=20

    notifications_button = ttk.Button(root, image=notifications_icon, compound="left", style="NavbarButton.TButton")
    notifications_button.place(x=780, y=10)  # Position at x=800, y=20
    
    account_button = ttk.Button(root, image=userImage_icon, compound="left", style="NavbarButton.TButton")
    account_button.place(x=840, y=10)  # Position at x=800, y=20
    
    # account_dropdown = ttk.Combobox(root, values=["Profile Photo", "Account Info"], style="NavbarCombobox.TCombobox")
    # account_dropdown.grid(row=0, column=5, padx=10, pady=10, sticky="e")
    # account_dropdown.current(0)

    # account_image = Image.open("favicon.png")  # Replace "profile_photo.png" with your image file
    # account_image = account_image.resize((20, 20), Image.Resampling.LANCZOS)  # Resize the image
    # account_photo = ImageTk.PhotoImage(account_image)
    # account_button = ttk.Button(root, image=account_photo, style="CircularButton.TButton")
    # account_button.image = account_photo
    # account_button.place(x=840, y=20)  # Position at x=900, y=20
    # account_button.grid(row=0, column=5,padx=(10, 20), pady=(10, 20), sticky="e")

    # Draw horizontal line
    horizontal_line = ttk.Separator(root, orient="horizontal")
    horizontal_line.place(x=50, y=50, width=1200)

    # Calculate button width
    element_width = 5  # Adjust as needed
    
    # Create and place buttons
    workspace_button = ttk.Button(text="Text", style="DarkButton.Tc Button", width=element_width)
    workspace_button.place(x=60, y=60)
    
    history_button = ttk.Button(text="Line", style="DarkButton.TButton", width=element_width)
    history_button.place(x=130, y=60)
    
    documentation_button = ttk.Button(text="Shape", style="DarkButton.TButton", width=element_width)
    documentation_button.place(x=190, y=60)

    settings_button = ttk.Button(text="Point", style="DarkButton.TButton", width=element_width)
    settings_button.place(x=60, y=140)

    root.mainloop()


if __name__ == "__main__":
    create_workspace_screen()


