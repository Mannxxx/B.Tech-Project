############# Change video size with the workspace size #####################

import tkinter as tk
from tkinter import filedialog
import vlc

class VLCPlayer:
    def __init__(self, master):
        self.master = master
        self.master.title("BladeSense Workspace")

        self.instance = vlc.Instance('--no-xlib')
        self.player = self.instance.media_player_new()
        self.file_path = None  # Store the selected file path

        self.create_widgets()

    def create_widgets(self):
        # Video Frame
        self.video_frame = tk.Frame(self.master)
        self.video_frame.pack(fill=tk.BOTH, expand=True)

        # VLC Widget
        self.vlc_player = self.instance.media_player_new()
        self.vlc_player.set_hwnd(self.video_frame.winfo_id())

        # Play Button
        self.play_button = tk.Button(self.master, text="Play", command=self.play)
        self.play_button.pack(side=tk.LEFT)

        # Pause Button
        self.pause_button = tk.Button(self.master, text="Pause", command=self.pause)
        self.pause_button.pack(side=tk.LEFT)

        # Stop Button
        self.stop_button = tk.Button(self.master, text="Stop", command=self.stop)
        self.stop_button.pack(side=tk.LEFT)

        # Load Video Button
        self.load_button = tk.Button(self.master, text="Load Video", command=self.load_video)
        self.load_button.pack(side=tk.LEFT)

    def play(self):
        if self.file_path:
            media = self.instance.media_new(self.file_path)
            self.player.set_media(media)
            self.player.play()

    def pause(self):
        self.player.pause()

    def stop(self):
        self.player.stop()

    def load_video(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])

if __name__ == "__main__":
    root = tk.Tk()
    player = VLCPlayer(root)
    root.mainloop()
