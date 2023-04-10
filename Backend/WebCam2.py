import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mpPose = mp.solutions.pose
from keras.models import load_model

# create skeleton from image
def skeleton_from_image(image, model):
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

    try:
        landmarks = results.pose_landmarks.landmark
        if landmarks[mpPose.PoseLandmark.LEFT_SHOULDER].visibility < 0.5 or landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER].visibility < 0.5:
            print("not WRISTS")

        if landmarks[mpPose.PoseLandmark.NOSE].visibility < 0.6:
            print("not NOSE")

        if landmarks[mpPose.PoseLandmark.RIGHT_KNEE].visibility < 0.5 or landmarks[mpPose.PoseLandmark.LEFT_KNEE].visibility < 0.5:
            print("not KNEES")
            print(landmarks[mpPose.PoseLandmark.RIGHT_KNEE].visibility)
    except:
        pass

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


colors = [(255, 0, 0), (255, 0, 239), (51, 0, 255), (0, 247, 255), (0, 255, 17), (255, 247, 0)]


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * len(actions[num]) * 18), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return output_frame

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * len(actions[num]) * 18), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return output_frame


def create_motion_images(image_arr):
    image_arr = list(map(lambda x, i: np.where(x < 255, (len(image_arr) - i) * (255 // 10), x), image_arr, [i for i in range(len(image_arr))]))
    total_image = image_arr[0]
    for i in range(1, len(image_arr)):
        total_image = cv2.bitwise_and(total_image, image_arr[i])

    return total_image


# 1. New detection variables
sequence = []
sentence = []
threshold = 0.8
model = load_model('newest model.h5')
categories = ['jumping jacks', 'squat', 'stand', 'side stretch', 'arm circles', 'high knees']

cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()
        frame1 = frame.copy()

        # Make detections
        image = skeleton_from_image(frame, holistic)

        # 2. Prediction logic
        sequence.append(image)
        sequence = sequence[-10:]

        if len(sequence) == 10:
            total_image = create_motion_images(sequence)
            res = model.predict(np.expand_dims(total_image, axis=0))[0]
            # print(categories[np.argmax(res)])

            # 3. Viz logic
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if categories[np.argmax(res)] != sentence[-1]:
                        sentence.append(categories[np.argmax(res)])
                else:
                    sentence.append(categories[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Viz probabilities
            frame1 = prob_viz(res, categories, frame1, colors)

        # Show to screen
        cv2.imshow('OpenCV Feed', frame1)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
