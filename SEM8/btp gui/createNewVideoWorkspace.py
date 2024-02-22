import tkinter as tk
from tkinter import ttk
from tkinter import *
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

# # Function to handle selection from the "Create New Video" dropdown
# def select_bluetooth_device(device):
#     print(f"Selected Bluetooth device: {device}")

class VideoPlayerWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Video Player")
        self.geometry("800x600")
        self.parent = parent

        self.create_widgets()

    def create_widgets(self):
        # Buttons for controlling video
        button_frame = ttk.Frame(self)
        button_frame.pack(side="top", padx=10, pady=10)

        upload_button = ttk.Button(button_frame, text="Upload Video")
        upload_button.pack(side="left", padx=5, pady=5)

        start_button = ttk.Button(button_frame, text="Start Video")
        start_button.pack(side="left", padx=5, pady=5)

        pause_button = ttk.Button(button_frame, text="Pause Video")
        pause_button.pack(side="left", padx=5, pady=5)

        close_button = ttk.Button(button_frame, text="Close Video", command=self.destroy)
        close_button.pack(side="left", padx=5, pady=5)

        # Frame to display video or thumbnail
        self.video_frame = ttk.Frame(self, width=640, height=480)
        self.video_frame.pack(side="top", padx=10, pady=10)

        # Placeholder image for video frame
        self.thumbnail_image = Image.open("play.ico")  # Replace with your thumbnail image
        self.thumbnail_image = self.thumbnail_image.resize((640, 480), Image.Resampling.LANCZOS)
        self.thumbnail_photo = ImageTk.PhotoImage(self.thumbnail_image)
        self.video_label = ttk.Label(self.video_frame, image=self.thumbnail_photo)
        self.video_label.pack(fill="both", expand=True)

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
    logo_image = Image.open("logo.png")  # Replace "logo.png" with your logo file
    logo_image = logo_image.resize((100, 40), Image.Resampling.LANCZOS)
    logo_photo = ImageTk.PhotoImage(logo_image)
    logo_label = ttk.Label(root, image=logo_photo)
    logo_label.image = logo_photo
    # logo_label.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
    logo_label.place(x=15, y=10)  # Position at x=10, y=10
    
    # Left Column with Buttons
    left_column = ttk.Frame(root, width=200, height=600, style="TFrame")
    # left_column.grid(row=1, column=0, rowspan=2, padx=10, pady=10, sticky="nw")
    left_column.place(x=0, y=60)  # Position at x=10, y=60
    
    # Calculate button width
    button_width = 19  # Adjust as needed
    
    # Create and place buttons
    workspace_button = ttk.Button(left_column, text="Workspace", style="DarkButton.TButton", width=button_width)
    workspace_button.grid(row=0, column=0, padx=10, pady=10, sticky="w")
    
    history_button = ttk.Button(left_column, text="Your History", style="DarkButton.TButton", width=button_width)
    history_button.grid(row=1, column=0, padx=10, pady=10, sticky="w")
    
    documentation_button = ttk.Button(left_column, text="Documentation", style="DarkButton.TButton", width=button_width)
    documentation_button.grid(row=2, column=0, padx=10, pady=10, sticky="w")

    settings_button = ttk.Button(text="Settings", style="DarkButton.TButton", width=button_width)
    settings_button.place(x=10, y=520)  # Position at x=0, y=500

    help_button = ttk.Button(text="Get Help", style="DarkButton.TButton", width=button_width)
    help_button.place(x=10, y=560)  # Position at x=0, y=500
    
    # Draw a vertical line after the left column
    vertical_line = ttk.Separator(root, orient="vertical")
    vertical_line.place(x=160, y=0, relheight=600)  # Adjust x-coordinate as needed

    # Workspace Header
    header_label = ttk.Label(root, text="Workspace", style="Header.TLabel", font=tkFont.Font(size=17))
    header_label.place(x=200, y=10)  # Position at x=250, y=10
    # header_label.grid(row=0, column=1, padx=10, pady=10, sticky="nw")
    # header_label = ttk.Label(root, text="\nAll workspace you have (0)", style="Header.TLabel", font=tkFont.Font(size=11))
    # header_label.grid(row=1, column=1, padx=10, pady=10, sticky="nw")

    # Description1
    description_label = ttk.Label(root, text="All workspace you have (0)", style="Description.TLabel")
    description_label.place(x=200, y=70)  # Position at x=250, y=50

    # Define coordinates
    x_start = 630
    y_start = 70
    button_width = 120  
    spacing = 5

    # Create and place buttons
    # create_video_button = ttk.Button(root, text="All Workspaces", style="MenuButton.TButton")
    # create_video_button.place(x=600, y=70)

    
    # bluetooth_devices = ["Device 1", "Device 2", "Device 3"]  # Example device names
    # selected_device = tk.StringVar()
    # selected_device.set(bluetooth_devices[0])  # Set default value

    # upload_video_button = ttk.Button(root, text="+ Create New Video", style="LargeButton.TButton")
    # upload_video_button.place(x=x_start + button_width + spacing, y=y_start)
    # upload_video_dropdown = OptionMenu(root, selected_device, *bluetooth_devices, command=select_bluetooth_device)
    # upload_video_dropdown.place(x=x_start + button_width + spacing, y=y_start + button_height)  # Adjust position as needed

    # Description2
    description_label = ttk.Label(root, text="Welcome to BladeSense. Create a new workspace to get started.\nYou can create a new video from scratch or upload a video from your desktop.", style="Description.TLabel")
    description_label.place(x=200, y=130)  # Position at x=250, y=50
    # description_label.grid(row=1, column=1, padx=10, pady=10, sticky="w")

    # Load .ico icons
    # search_icon = Image.open("search.ico")
    notifications_icon = Image.open("notification.ico")
    userImage_icon=Image.open("user1.png")

    # Resize icons
    # search_icon = search_icon.resize((20, 20), Image.Resampling.LANCZOS)
    notifications_icon = notifications_icon.resize((15, 15), Image.LANCZOS)
    userImage_icon=userImage_icon.resize((16,16),Image.LANCZOS)

    # Convert to PhotoImage
    # search_icon = ImageTk.PhotoImage(search_icon)
    notifications_icon = ImageTk.PhotoImage(notifications_icon)
    userImage_icon=ImageTk.PhotoImage(userImage_icon)

    # button = Button(root, text="Click me!")
    # img = PhotoImage(file="C:/Users/Mansi/Desktop/clone/BTP/SEM8/btp gui/user1.png") # make sure to add "/" not "\"
    # button.config(image=img)
    # button.place(x=780, y=20)
    # button.pack() # Displaying the button

    # Navbar
    # search_button = ttk.Button(root, text="Search Workspaces", compound="left", style="NavBarButton.TButton")
    # search_button.place(x=640, y=20)  # Position at x=600, y=20

    notifications_button = ttk.Button(root, image=notifications_icon, compound="left", style="NavbarButton.TButton")
    notifications_button.place(x=780, y=20)  # Position at x=800, y=20
    
    account_button = ttk.Button(root, image=userImage_icon, compound="left", style="NavbarButton.TButton")
    account_button.place(x=840, y=20)  # Position at x=800, y=20

    # Change background color to white
    notifications_button.configure(style="White.TButton")
    account_button.configure(style="White.TButton")

    # Define a style with white background
    style = ttk.Style()
    style.configure("Custom.TButton", borderwidth=0, background="#343a40")

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
    horizontal_line.place(x=160, y=60, width=730)

    # Define coordinates
    x_start = 200
    y_start = 200
    button_width = 150  
    spacing = 20

    def create_video_window():
        video_window = VideoRecorderWindow(root)

    create_video_button = ttk.Button(root, text="Create New Video", style="LargeButton.TButton", command=create_video_window)
    create_video_button.place(x=x_start, y=y_start)

    # Create and place buttons
    # create_video_button = ttk.Button(root, text="Create New Video", style="LargeButton.TButton")
    # create_video_button.place(x=x_start, y=y_start)

    def open_video_player_window():
        video_player_window = VideoPlayerWindow(root)

    upload_video_button = ttk.Button(root, text="Upload New Video", style="LargeButton.TButton", command=open_video_player_window)
    upload_video_button.place(x=x_start + button_width + spacing, y=y_start)

    # upload_video_button = ttk.Button(root, text="Upload New Video", style="LargeButton.TButton")
    # upload_video_button.place(x=x_start + button_width + spacing, y=y_start)

    recently_closed_button = ttk.Button(root, text="Open Recently Closed", style="LargeButton.TButton")
    recently_closed_button.place(x=x_start + 2*(button_width + spacing), y=y_start)

    # Draw horizontal line
    horizontal_line = ttk.Separator(root, orient="horizontal")
    horizontal_line.place(x=160, y=255, width=730)

    # Description2
    description_label = ttk.Label(root, text="To make your onboarding process more appealing, discover this helpful resources before you begin.")
    description_label.place(x=200, y=270)  # Position at x=250, y=50
    
    # Function to open YouTube video
    def open_youtube_video(url):
        webbrowser.open(url)

    # Button 1
    button_1 = ttk.Button(root, text="Product Overview\nHow BladeSense works\n4:12", style="LargeButton.TButton", command=lambda: open_youtube_video("https://www.youtube.com/watch?v=video1"))
    button_1.place(x=200, y=310)  # Adjust x and y coordinates as needed
    # button_1.config(font=("Arial", 10))  # Set font size

    # Button 2
    button_2 = ttk.Button(root, text="Features Overview\nOur Features\n4:12", style="LargeButton.TButton", command=lambda: open_youtube_video("https://www.youtube.com/watch?v=video2"))
    button_2.place(x=370, y=310)  # Adjust x and y coordinates as needed
    # button_2.config(font=("Arial", 10))  # Set font size

    # Button 3
    button_3 = ttk.Button(root, text="Product Walkthrough\nOnboarding\n4:12", style="LargeButton.TButton", command=lambda: open_youtube_video("https://www.youtube.com/watch?v=video3"))
    button_3.place(x=540, y=310)  # Adjust x and y coordinates as needed
    # button_3.config(font=("Arial", 10))  # Set font size

    # Button 4
    button_4 = ttk.Button(root, text="Frequently Asked Question\nGet Help", style="LargeButton.TButton", command=lambda: open_youtube_video("https://www.youtube.com/watch?v=video4"))
    button_4.place(x=710, y=310)  # Adjust x and y coordinates as needed
    # button_4.config(font=("Arial", 10))  # Set font size


    root.mainloop()


if __name__ == "__main__":
    create_workspace_screen()


