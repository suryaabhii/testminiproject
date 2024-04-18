import tkinter as tk
import cv2
from PIL import Image, ImageTk
import os

class SurviellanceApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.btn_snapshot = tk.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        self.delay = 10
        self.output_folder = "detected_faces"
        os.makedirs(self.output_folder, exist_ok=True)

        self.update()

    def snapshot(self):
        ret, frame = self.vid.read()
        if ret:
            faces = self.detect_faces(frame)
            if faces.any():
                self.save_image(frame, faces)

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            faces = self.detect_faces(frame)
            self.draw_faces(frame, faces)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

    def draw_faces(self, frame, faces):
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    def save_image(self, frame, faces):
        for i, (x, y, w, h) in enumerate(faces):
            face_img = frame[y:y+h, x:x+w]
            filename = os.path.join(self.output_folder, f"face_{i}.png")
            cv2.imwrite(filename, face_img)

def main():
    root = tk.Tk()
    app = SurviellanceApp(root, "SurviellanceApp")
    root.mainloop()

if __name__ == "__main__":
    main()
