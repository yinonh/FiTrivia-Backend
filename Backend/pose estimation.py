from threading import Thread
from time import sleep
import numpy as np


import cv2
from keras.models import load_model
from matplotlib import pyplot as plt

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mpPose = mp.solutions.pose





# def show_video():
#     # Connect to webcam
#     cap = cv2.VideoCapture(0)
#     # Loop through every frame until we close our webcam
#     while cap.isOpened():
#         ret, image = cap.read()
#
#         # Show image
#         cv2.imshow('Webcam', image)
#
#         # Checks whether q has been hit and stops the loop
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     # Releases the webcam
#     cap.release()
#     # Closes the frame
#     cv2.destroyAllWindows()

FRAME = ''
categories = ['jumping jacks', 'squat']
model = load_model('Model')


def take_photo(cap):
    ret, frame = cap.read()
    return frame

def video():
    global FRAME
    cap = cv2.VideoCapture(0)
    while cap.isOpened:
        frame = take_photo(cap)
        FRAME = frame
        # Show image
        cv2.imshow('Webcam', frame)

        # Checks whether q has been hit and stops the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


def skeleton_from_image(index):
    image = FRAME
    with mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            refine_face_landmarks=True) as holistic:
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print("too soon")
            return
        results = holistic.process(image)

    # Draw landmark annotation on the image.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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
                              mp_drawing.DrawingSpec((255, 0, 0), 2, 2),
                              mp_drawing.DrawingSpec((255, 0, 255), 2, 2),
                              )

    gray = cv2.cvtColor(opImg.astype('uint8'), cv2.COLOR_RGB2GRAY)
    img = [gray]
    test_data = np.array(img)
    test_data = test_data / 255
    #cv2.imwrite(f'{index}.png', gray)
    y_proba = model.predict(test_data)
    print(y_proba)
    x = categories[0] if y_proba[0][0] > y_proba[0][1] else categories[1]
    x += f" {int(max(y_proba[0]) * 100)}" + "%"
    print(x)

    #return gray


if __name__ == "__main__":
    thread1 = Thread(target=video)
    thread1.start()
    for i in range(25):
        sleep(1)
        thread2 = Thread(target=skeleton_from_image, args=(i,))
        thread2.start()



