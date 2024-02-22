import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2

class VideoPlayerWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video Player")
        self.geometry("1200x750")

        self.create_widgets()

    def create_widgets(self):
        # Frame to display video or thumbnail
        self.video_frame = ttk.Frame(self, width=640, height=580)
        self.video_frame.pack(side="top", padx=10, pady=10)

        # Button frame
        self.button_frame = ttk.Frame(self)
        self.button_frame.pack(side="top", padx=10, pady=10)

        # Video player
        self.cap = None
        self.frame_number = tk.IntVar(value=0)

        # Buttons
        upload_button = ttk.Button(self.button_frame, text="Upload Video", command=self.upload_video)
        upload_button.pack(side="left", padx=5, pady=5)

        previous_button = ttk.Button(self.button_frame, text="Previous Frame", command=self.previous_frame)
        previous_button.pack(side="left", padx=5, pady=5)

        next_button = ttk.Button(self.button_frame, text="Next Frame", command=self.next_frame)
        next_button.pack(side="left", padx=5, pady=5)

        close_button = ttk.Button(self.button_frame, text="Close Window", command=self.destroy)
        close_button.pack(side="left", padx=5, pady=5)

        # Label to display frame number
        self.frame_label = ttk.Label(self.button_frame, textvariable=self.frame_number)
        self.frame_label.pack(side="left", padx=5, pady=5)

    def upload_video(self):
        file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video files", "*.mp4;*.avi;*.mkv;*.mov")])
        if file_path:
            # Open the selected video file
            self.cap = cv2.VideoCapture(file_path)
            self.play_video()

    def play_video(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = self.resize_image(frame, self.video_frame.winfo_width(), self.video_frame.winfo_height())
                frame = ImageTk.PhotoImage(frame)
                if hasattr(self, 'video_label'):
                    self.video_label.configure(image=frame)
                    self.video_label.image = frame
                else:
                    self.video_label = ttk.Label(self.video_frame, image=frame)
                    self.video_label.pack(fill="both", expand=True)
                self.frame_number.set(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))  # Update frame number
                self.video_label.after(10, self.play_video)  # Update video frame every 10 milliseconds

    def previous_frame(self):
        if self.cap is not None and self.frame_number.get() > 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number.get() - 2)
            self.frame_number.set(self.frame_number.get() - 2)
            self.play_video()

    def next_frame(self):
        if self.cap is not None:
            self.play_video()

    def close_video(self):
        # Close the video (release resources)
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def resize_image(self, image, width, height):
        """
        Resize the given image to fit within the specified width and height
        while maintaining the aspect ratio.
        """
        aspect_ratio = image.width / image.height
        if width / height > aspect_ratio:
            width = int(height * aspect_ratio)
        else:
            height = int(width / aspect_ratio)
        return image.resize((width, height), Image.Resampling.LANCZOS)

if __name__ == "__main__":
    app = VideoPlayerWindow()
    app.mainloop()
