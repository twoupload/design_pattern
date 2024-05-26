import pygame
import cv2
import numpy as np
import tensorflow as tf

# 게임 초기화
pygame.init()

# 화면 설정
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption('가위바위보 게임')

# 웹캠 설정
cap = cv2.VideoCapture(0)

# Teachable Machine 모델 로드
model = tf.keras.models.load_model('path_to_your_model/model.h5')

# 클래스 레이블
class_names = ['가위', '바위', '보']

# 게임 로직
def get_prediction(frame):
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    prediction = model.predict(np.expand_dims(normalized_frame, axis=0))
    return class_names[np.argmax(prediction)]

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    ret, frame = cap.read()
    if not ret:
        break

    # OpenCV에서 색상 변환
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.fliplr(frame)
    frame = np.rot90(frame)

    # 예측
    prediction = get_prediction(frame)

    # 화면에 이미지와 예측 결과 표시
    frame_surface = pygame.surfarray.make_surface(frame)
    screen.blit(frame_surface, (0, 0))
    font = pygame.font.Font(None, 36)
    text = font.render(f'Prediction: {prediction}', True, (255, 0, 0))
    screen.blit(text, (10, 10))

    pygame.display.flip()

cap.release()
pygame.quit()