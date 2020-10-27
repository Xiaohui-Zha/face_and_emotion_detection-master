## Emotion detection
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import face_recognition
import keras
from keras.models import load_model
import cv2

from config import VIDEO_PATH

# 预测表情传入图片
def predicted_emotion(face_image):
    emotion_dict = {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}
    face_image = cv2.resize(face_image, (48, 48))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
    # Load the model trained for detecting emotions of a face
    model = load_model("./emotion_detector_models/model_v6_23.hdf5")
    print(face_image.shape)
    predicted_class = np.argmax(model.predict(face_image))
    label_map = dict((v, k) for k, v in emotion_dict.items())
    predicted_label = label_map[predicted_class]
    print(predicted_label)
    return predicted_label


if __name__ == '__main__':
    i = 1
    video_capture = cv2.VideoCapture(VIDEO_PATH)
    while True:
        for j in range(10):
            video_capture.read()
        ret, frame_background = video_capture.read()
        face_locations = face_recognition.face_locations(frame_background)
        for face_location in face_locations:
            top = face_location[0]
            right = face_location[1]
            bottom = face_location[2]
            left = face_location[3]
            face_image1 = frame_background[top:bottom, left:right]
            plt.imshow(face_image1)
            emotion_class = predicted_emotion(face_image1)
            image_save = Image.fromarray(face_image1)
            # image_save.save("image_"+str(i)+".jpg")
            i += 1
            cv2.rectangle(frame_background, (int(left), int(top)), (int(right), int(bottom)),
                          (255, 0, 0), 2)
            cv2.putText(frame_background, emotion_class, (int(left), int(top - 6)), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 0, 0), 1)
        cv2.imshow('Video', frame_background)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()

cv2.destroyAllWindows()


    # 传入图片测试代码
    # image = face_recognition.load_image_file("./test_images/040wrmpyTF5l.jpg")
    # face_locations = face_recognition.face_locations(image)
    # i=1
    # for face_location in face_locations:
    #     top = face_location[0]
    #     right = face_location[1]
    #     bottom = face_location[2]
    #     left = face_location[3]
    #     face_image1 = image[top:bottom, left:right]
    #
    #     plt.imshow(face_image1)
    #     emotion_class=predicted_emotion(face_image1)
    #     image_save = Image.fromarray(face_image1)
    #     image_save.save("image_"+str(i)+".jpg")
    #     i+=1
    # # cv2.rectangle(image, (pic_left, pic_top), (pic_right, pic_bottom), (0, 0, 255), 2)
    #     cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)),
    #                   (255, 0, 0), 2)
    #     cv2.putText(image, emotion_class, (int(left), int(top-6)), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
    #                 (255, 0,0), 1)
    # plt.imshow(image)
    # image_save = Image.fromarray(image)
    # image_save.save("image.jpg")