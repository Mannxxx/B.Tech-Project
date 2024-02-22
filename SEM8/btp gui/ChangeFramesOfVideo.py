import tkinter as tk
from tkinter import ttk, filedialog
import cv2

class VideoPlayer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video Player")
        self.geometry("800x600")
        
        self.create_widgets()

    def create_widgets(self):
        self.video_frame = ttk.Frame(self)
        self.video_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(self.video_frame)
        self.canvas.pack(fill="both", expand=True)

        control_frame = ttk.Frame(self)
        control_frame.pack(fill="x")

        self.button_back = ttk.Button(control_frame, text="Back", command=self.back_frame)
        self.button_back.pack(side="left", padx=5, pady=5)

        self.button_forward = ttk.Button(control_frame, text="Forward", command=self.forward_frame)
        self.button_forward.pack(side="left", padx=5, pady=5)

        self.label_frame_number = ttk.Label(control_frame, text="Frame: 0")
        self.label_frame_number.pack(side="left", padx=5, pady=5)

        self.video_path = None
        self.cap = None
        self.current_frame = 0

    def open_video(self):
        self.video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video files", "*.mp4;*.avi;*.mkv;*.mov")])
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.play_video()

    def play_video(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.photo = tk.PhotoImage(data=cv2.imencode('.png', frame)[1].tobytes())
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
                self.current_frame += 1
                self.label_frame_number.config(text=f"Frame: {self.current_frame}")
            else:
                self.cap.release()

    def back_frame(self):
        if self.cap is not None and self.current_frame > 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame - 2)
            self.current_frame -= 2
            self.play_video()

    def forward_frame(self):
        if self.cap is not None:
            self.play_video()

if __name__ == "__main__":
    app = VideoPlayer()

    button_frame = ttk.Frame(app)
    button_frame.pack(fill="x")

    upload_button = ttk.Button(button_frame, text="Upload Video", command=app.open_video)
    upload_button.pack(side="left", padx=5, pady=5)

    app.mainloop()
