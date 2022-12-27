from tkinter import StringVar, messagebox
from PIL import Image, ImageTk
import ttkbootstrap as tb
#from ttkbootstrap.dialogs.dialogs import Messagebox

import numpy as np
from threading import Thread
import time
import os
import shutil
from pathlib import Path
import subprocess
import webbrowser

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mpPose = mp.solutions.pose

class VideoCap:
    EXERCISES = {'jumping jacks': 'https://youtu.be/yDSMdd8hiFg?t=20',
                 'squat': 'https://youtu.be/jFjgWYIB_4Y?t=21',
                 'high knees': 'https://www.youtube.com/watch?v=ZZZoCNMU48U',
                 'side stretch': 'https://youtu.be/DvJa9tiMivw?t=414',
                 'arm circles ': 'https://youtu.be/140RTNMciH8?t=3',
                 'stand': 'https://letmegooglethat.com/?q=%D7%A1%D7%AA%D7%9D+%D7%A1%D7%A8%D7%98%D7%95%D7%9F+%D7%A9%D7%9C%D7%9B%D7%9D+%D7%A2%D7%95%D7%9E%D7%93%D7%99%D7%9D'
                 }

    def __init__(self, *args, **kwargs):
        super(VideoCap, self).__init__(*args, **kwargs)
        self.style = 'superhero'
        self.win = tb.Window(themename='flatly', resizable=(False, False))
        self.win.protocol('WM_DELETE_WINDOW', self.endProgram)  # root is your root window
        try:
            self.cap = cv2.VideoCapture(0)
            cv2image = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB)
        except:
            messagebox.showerror("Camera not found", "Error")
            exit()

        height, width = 1100, 700
        self.win.geometry(f"{height}x{width}")

        self.current_ex_var = StringVar()
        self.counter_label_var = StringVar()
        self.counter_label_var.set('Start 5 second counter')

        self.alive = True

        if not os.path.exists('result'):
            os.makedirs('result')

        self.combo()
        self.top_label()
        self.dark_mode()
        self.image_frame()
        self.bottom_buttons()
        self.win.mainloop()
        self.cap.release()

    def endProgram(self):
        self.kill()
        self.win.destroy()

    def combo(self):
        container = tb.Frame(master=self.win, padding=2)
        container.grid(row=0, column=0, sticky="nsew")
        self.current_ex_var.set('jumping jacks')
        self.ex = tb.Combobox(master=container, bootstyle="primary", textvariable=self.current_ex_var, values=list(self.EXERCISES.keys()), state='readonly')
        self.ex.grid(row=0, column=0)

    def top_label(self):
        container = tb.Frame(master=self.win, padding=2)
        container.grid(row=0, column=1, sticky="nsew")
        tb.Label(master=container).pack()
        tb.Label(master=container, textvariable=self.counter_label_var, font=("Arial", 25)).pack()

    def threaded_function(self):
        for i in range(5, -1, -1):
            self.counter_label_var.set(f'Start {self.current_ex_var.get()} in {i}')
            time.sleep(1)
            if not self.alive:
                return
        thread = Thread(target=self.counter)
        thread.start()
        self.save_images()

    def counter(self):
        for i in range(30, -1, -1):
            self.counter_label_var.set(f'{self.current_ex_var.get()} {i}')
            time.sleep(1)
            if not self.alive:
                return
        self.kill()
        # tb.Progressbar(bootstyle="success", maximum=30, value=i).grid(row=3, column=1)

    def save_images(self):
        num = 0
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while self.cap.isOpened() and self.alive:
                try:
                    image = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_RGB2BGR)
                    image.flags.writeable = False  # Image is no longer writeable
                    results = holistic.process(image)  # Make prediction
                    image.flags.writeable = True

                    mp_drawing.draw_landmarks(
                        image,
                        results.face_landmarks,
                        mp_holistic.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles
                            .get_default_pose_landmarks_style())
                    h, w, c = image.shape  # get shape of original frame
                    opImg = np.zeros([128, 128, c])  # create blank image with original frame size
                    opImg.fill(255)  # set white background. put 0 if you want to make it black

                    # draw extracted pose on black white iqmage
                    mp_drawing.draw_landmarks(opImg, results.pose_landmarks, mpPose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec((255, 0, 0), 1, 1),
                                              mp_drawing.DrawingSpec((255, 0, 255), 1, 1),
                                              )

                    gray = cv2.cvtColor(opImg.astype('uint8'), cv2.COLOR_RGB2GRAY)
                    cv2.imwrite(f'result/{self.ex.get()}/{num}.png', cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
                    num += 1
                except:
                    print(num)
        self.ex['state'] = 'readonly'

    def start_record(self):
        if not os.path.exists(f'result/{self.ex.get()}'):
            os.makedirs(f'result/{self.ex.get()}')
        self.ex['state'] = 'disabled'
        self.alive = True
        thread = Thread(target=self.threaded_function)
        thread.start()

    def kill(self):
        self.ex['state'] = 'readonly'
        self.counter_label_var.set('Start 5 second counter')
        self.alive = False

    def download(self):
        shutil.make_archive(str(Path.home() / "Downloads/results"), 'zip', 'result')
        #messagebox.showinfo("result.zip created in the download folder", "Success")
        subprocess.Popen('explorer ' + str(Path.home() / "Downloads"), shell=True)

    def open_example(self):
        webbrowser.open(self.EXERCISES[self.current_ex_var.get()])


    def bottom_buttons(self):
        container = tb.Frame(master=self.win, padding=2)
        container.grid(row=2, column=1, sticky="ns")
        tb.Label(master=container).grid(row=0, column=0)
        tb.Button(text="Start", master=container, command=self.start_record).grid(row=1, column=0)
        tb.Label(text="\t", master=container).grid(row=1, column=1)
        tb.Button(text="Stop", master=container, command=self.kill, bootstyle="danger").grid(row=1, column=2)
        tb.Label(text="\t", master=container).grid(row=1, column=3)
        tb.Button(text="Download", master=container, command=self.download, bootstyle="success").grid(row=1, column=4)
        tb.Label(text="\t", master=container).grid(row=1, column=5)
        tb.Button(text="Show example", master=container, command=self.open_example, bootstyle="info").grid(row=1, column=6)

    def dark_mode(self):
        container = tb.Frame(master=self.win, padding=2)
        container.grid(row=0, column=2, sticky="nsew")
        tb.Checkbutton(master=container, bootstyle="round-toggle", command=self.dMode, text="dark mode",
                       state=True).grid(row=0, column=0)
        tb.Label(text="\t\t\t").grid(row=1, column=0)

    def dMode(self):
        self.win._style = tb.Style(self.style)
        if self.style == 'superhero':
            self.style = 'flatly'
        else:
            self.style = 'superhero'

    def image_frame(self):
        container = tb.Frame(master=self.win, padding=2, bootstyle="info")
        container.grid(row=1, column=1)

        self.label = tb.Label(container)
        self.label.grid(row=1, column=1)

        cv2image = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)
        self.label.after(20, self.image_frame)


if __name__ == "__main__":
    VideoCap()
