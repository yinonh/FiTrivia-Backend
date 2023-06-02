from rest_framework.views import APIView
from rest_framework.response import Response
import cv2
import numpy as np
import mediapipe as mp
import datetime
import os

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mpPose = mp.solutions.pose
from keras.models import load_model

NUM_OF_IMAGES = 20


class images(APIView):
    model = load_model('newest model.h5')
    categories = ['jumping jacks', 'squat', 'stand', 'side stretch', 'arm circles', 'high knees']

    # create skeleton from image
    def skeleton_from_image(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False  # Image is no longer writeable
        results = model.process(image)  # Make prediction
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

        # try:
        #     landmarks = results.pose_landmarks.landmark
        #     if landmarks[mpPose.PoseLandmark.LEFT_SHOULDER].visibility < 0.5 or landmarks[
        #         mpPose.PoseLandmark.RIGHT_SHOULDER].visibility < 0.5:
        #         print("not WRISTS")
        #
        #     if landmarks[mpPose.PoseLandmark.NOSE].visibility < 0.6:
        #         print("not NOSE")
        #
        #     if landmarks[mpPose.PoseLandmark.RIGHT_KNEE].visibility < 0.5 or landmarks[
        #         mpPose.PoseLandmark.LEFT_KNEE].visibility < 0.5:
        #         print("not KNEES")
        #         print(landmarks[mpPose.PoseLandmark.RIGHT_KNEE].visibility)
        # except:
        #     pass

        h, w, c = image.shape  # get shape of original frame
        opImg = np.zeros([128, 128, c])  # create blank image with original frame size
        opImg.fill(255)  # set white background. put 0 if you want to make it black

        # draw extracted pose on black white iqmage
        mp_drawing.draw_landmarks(opImg, results.pose_landmarks, mpPose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec((255, 0, 0), 1, 1),
                                  mp_drawing.DrawingSpec((255, 0, 255), 1, 1),
                                  )

        gray = cv2.cvtColor(opImg.astype('uint8'), cv2.COLOR_RGB2GRAY)
        return gray

    def create_motion_images(self, image_arr):
        image_arr = list(map(lambda x, i: np.where(x < 255, (len(image_arr) - i) * (255 // NUM_OF_IMAGES), x), image_arr,
                             [i for i in range(len(image_arr))]))
        total_image = image_arr[0]
        for i in range(1, len(image_arr)):
            total_image = cv2.bitwise_and(total_image, image_arr[i])

        return total_image

    def post(self, request):
        images = request.FILES.getlist('images')
        np_images = []
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for i, image in enumerate(images):
                np_image = np.frombuffer(image.read(), dtype=np.uint8)
                np_image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
                np_image = cv2.resize(np_image, (128, 128))  # resize image
                np_image = self.skeleton_from_image(np_image, holistic)
                np_images.append(np_image)
        print(len(np_images))
        total_image = self.create_motion_images(np_images)
        # output_path = os.path.join(os.getcwd(), 'assets', f'{datetime.datetime.now().strftime("%H%M%S")}.png')
        # cv2.imwrite(output_path, total_image)
        res = self.model.predict(np.expand_dims(total_image, axis=0))[0]
        print(self.categories[np.argmax(res)])

        return Response({'message': self.categories[np.argmax(res)]})





