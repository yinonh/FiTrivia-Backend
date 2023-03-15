from tkinter import StringVar, messagebox
from PIL import Image, ImageTk
import ttkbootstrap as tb
from tkinter import ttk

#from ttkbootstrap.dialogs.dialogs import Messagebox

import queue
import numpy as np
from threading import Thread, enumerate, Event, Condition
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
                 'arm circles': 'https://youtu.be/140RTNMciH8?t=3',
                 'stand': 'https://letmegooglethat.com/?q=%D7%A1%D7%AA%D7%9D+%D7%A1%D7%A8%D7%98%D7%95%D7%9F+%D7%A9%D7%9C%D7%9B%D7%9D+%D7%A2%D7%95%D7%9E%D7%93%D7%99%D7%9D'
                 }

    def __init__(self, *args, **kwargs):
        super(VideoCap, self).__init__(*args, **kwargs)
        self.style = 'superhero'
        self.win = tb.Window(themename='flatly', resizable=(False, False))
        self.win.protocol("WM_DELETE_WINDOW", self.onClose)
        self.stopEvent = Event()
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cv2image = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB)
        except:
            messagebox.showerror("Camera not found", "Error")
            exit()
        self.camera_on = False

        height, width = 1100, 700
        self.win.geometry(f"{height}x{width}")

        self.current_ex_var = StringVar()
        self.counter_label_var = StringVar()
        self.counter_label_var.set('Start 5 second counter')

        self.frame_queue = queue.Queue()
        self.alive = False
        self.recording = False
        self.saving = False
        self.process = False

        if not os.path.exists('result'):
            os.makedirs('result')

        self.container = tb.Frame(master=self.win, padding=2, bootstyle="info")
        self.container.grid(row=1, column=1)
        self.label = None

        thread = Thread(target=self.long_running_task)
        thread.start()
        try:
            self.combo()
            self.top_label()
            self.dark_mode()
            self.start_thread()
            #self.show_frame()
            self.bottom_buttons()
            self.win.mainloop()
        except:
            pass
        finally:
            self.onClose()

    def onClose(self):
        if not self.frame_queue.empty():
            if messagebox.askquestion("Still processing the images", "Are you sure you want to exit?") == 'no':
                return
        self.stopEvent.set()
        self.cap.release()
        self.win.destroy()
        os._exit(0)

    def long_running_task(self):
        self.progressbar = ttk.Progressbar(master=self.container, mode='indeterminate', maximum=100)
        self.progressbar.grid(row=1, column=1)
        while not self.camera_on:
            self.progressbar.step()
            self.progressbar.update()
            time.sleep(0.1)  # simulating a long-running task
        self.progressbar.stop()

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
        self.recording = True
        thread = Thread(target=self.counter)
        thread.start()
        self.save_images(self.ex.get())

    def counter(self):
        for i in range(30, -1, -1):
            self.counter_label_var.set(f'{self.current_ex_var.get()} {i}')
            time.sleep(1)
            if not self.alive:
                return
        self.recording = False
        self.kill()

    def create_dir_with_next_num(self, path):
        dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        if dirs:
            max_num = max(int(d) for d in dirs)
            new_dir_name = str(max_num + 1)
        else:
            new_dir_name = '1'
        os.mkdir(os.path.join(path, new_dir_name))
        return os.path.join(path, new_dir_name)

    def save_images(self, ex):
        time.sleep(1)
        self.process = True
        num = 0
        path = self.create_dir_with_next_num(f'result/{ex}')
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while not self.frame_queue.empty():
                try:
                    image = cv2.cvtColor(self.frame_queue.get(), cv2.COLOR_RGB2BGR)
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
                    cv2.imwrite(f'{path}/{num}.png', cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
                    num += 1
                except:
                    print(num)
        self.process = False
        self.saving = False
        self.start_b['state'] = 'normal'
        self.download_b['state'] = 'normal'
        self.ex['state'] = 'readonly'

    def start_record(self):
        if not os.path.exists(f'result/{self.ex.get()}'):
            os.makedirs(f'result/{self.ex.get()}')
        self.start_b['state'] = 'disabled'
        self.download_b['state'] = 'disabled'
        self.ex['state'] = 'disabled'
        self.alive = True
        thread = Thread(target=self.threaded_function)
        thread.start()

    def kill(self):
        if not self.process:
            self.start_b['state'] = 'normal'
            self.download_b['state'] = 'normal'
        self.ex['state'] = 'readonly'
        self.counter_label_var.set('Start 5 second counter')
        self.alive = False
        self.recording = False

    def download(self):
        if self.frame_queue.empty():
            if not os.path.exists("result"):
                messagebox.showerror("Error", "There is nothing to save")
            else:
                result_file_path = str(Path.home() / "Downloads/results.zip")
                counter = 1
                while os.path.exists(result_file_path):
                    result_file_path = str(Path.home() / f"Downloads/results({counter}).zip")
                    counter += 1

                shutil.make_archive(os.path.splitext(result_file_path)[0], 'zip', 'result')
                subprocess.Popen('explorer ' + str(Path.home() / "Downloads"), shell=True)
                shutil.rmtree('result')
        else:
            messagebox.showerror("Error", "Still processing the images, please waite")

    def open_example(self):
        webbrowser.open(self.EXERCISES[self.current_ex_var.get()])


    def bottom_buttons(self):
        container = tb.Frame(master=self.win, padding=2)
        container.grid(row=2, column=1, sticky="ns")
        tb.Label(master=container).grid(row=0, column=0)
        self.start_b = tb.Button(text="Start", master=container, command=self.start_record)
        self.start_b.grid(row=1, column=0)
        tb.Label(text="\t", master=container).grid(row=1, column=1)
        self.stop_b = tb.Button(text="Stop", master=container, command=self.kill, bootstyle="danger")
        self.stop_b.grid(row=1, column=2)
        tb.Label(text="\t", master=container).grid(row=1, column=3)
        self.download_b = tb.Button(text="Download", master=container, command=self.download, bootstyle="success")
        self.download_b.grid(row=1, column=4)
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


    def start_thread(self):
        video_thread = Thread(target=self.videoLoop, args=())
        video_thread.start()

    def videoLoop(self):
        try:
            self.cap = cv2.VideoCapture(0)
            self.camera_on = True
            while not self.stopEvent.is_set():
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                    if self.recording:
                        self.ex.style = 'danger'
                        self.frame_queue.put(cv2image)
                        if not self.saving:
                            self.saving = True
                    img = Image.fromarray(cv2image)
                    imgtk = ImageTk.PhotoImage(image=img)
                    if self.label is None:
                        self.label = tb.Label(self.container)
                        self.label.configure(image=imgtk)
                        self.label.image = imgtk
                        self.label.grid(row=1, column=1)
                    else:
                        self.label.configure(image=imgtk)
                        self.label.image = imgtk
                self.win.update()
                time.sleep(0.015)
        except Exception as e:
            print(f"[INFO] caught an error: {e}")
            exit()


if __name__ == "__main__":
    VideoCap()


