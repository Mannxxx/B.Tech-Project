import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class VideoProcessorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Video Processor")

        # Set the fixed workspace size
        self.workspace_width = 800
        self.workspace_height = 600

        # Create a label to display the video
        self.video_label = tk.Label(self.master)
        self.video_label.pack()

        # Create buttons for loading and processing the video
        self.load_button = tk.Button(self.master, text="Load Video", command=self.load_video)
        self.load_button.pack()

        self.process_button = tk.Button(self.master, text="Process Video", command=self.process_video)
        self.process_button.pack()

    def load_video(self):
        # Open a file dialog to choose a video file
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;")])

        # Read the video and get its properties
        self.cap = cv2.VideoCapture(file_path)
        self.video_width = int(self.cap.get(3))
        self.video_height = int(self.cap.get(4))

        # Update the workspace size based on the aspect ratio
        aspect_ratio = self.video_height / self.video_width
        new_width = int(self.workspace_height / aspect_ratio)
        new_height = self.workspace_height
        self.workspace_width = min(new_width, self.workspace_width)

        # Resize the video to match the workspace size
        self.cap.set(3, self.workspace_width)
        self.cap.set(4, self.workspace_height)

        # Update the label to display the video
        self.update_video()

    def process_video(self):
        # Check if a video is loaded
        if hasattr(self, 'cap'):
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Convert the frame to RGB for displaying in Tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize the frame to match the workspace size
                frame_resized = cv2.resize(frame_rgb, (self.workspace_width, self.workspace_height))

                # Convert the resized frame to ImageTk format
                img = Image.fromarray(frame_resized)
                img_tk = ImageTk.PhotoImage(image=img)

                # Update the label with the resized frame
                self.video_label.configure(image=img_tk)
                self.video_label.img = img_tk

                # Update the Tkinter window
                self.master.update()

    def update_video(self):
        # Read the first frame from the video
        ret, frame = self.cap.read()
        if ret:
            # Convert the frame to RGB for displaying in Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize the frame to match the workspace size
            frame_resized = cv2.resize(frame_rgb, (self.workspace_width, self.workspace_height))

            # Convert the resized frame to ImageTk format
            img = Image.fromarray(frame_resized)
            img_tk = ImageTk.PhotoImage(image=img)

            # Update the label with the resized frame
            self.video_label.configure(image=img_tk)
            self.video_label.img = img_tk

    def __del__(self):
        # Release the video capture object when the application is closed
        if hasattr(self, 'cap'):
            self.cap.release()

# Create the Tkinter root window
root = tk.Tk()
app = VideoProcessorApp(root)

# Run the Tkinter event loop
root.mainloop()
