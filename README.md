# Boopathi.
import cv2
import mediapipe  as mp
#print(mp.__file__)

#print(dir(mp))

mp_hand =mp.solutions.hands
hands =mp_hand.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

while True:
    success,img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    if result.multi_hand_landmarks:
        for multi_hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                img,
                multi_hand_landmarks,
                mp_hand.HAND_CONNECTIONS
            )
    
    cv2.imshow("Hand Detection",img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyALLWindows()
