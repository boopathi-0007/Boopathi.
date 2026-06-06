import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

tip_ids = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()

    if not success:
        break

    img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    finger_count = 0

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            lm_list = []

            h, w, c = img.shape

            for id, lm in enumerate(hand_landmarks.landmark):
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                lm_list.append((cx, cy))

            if len(lm_list) != 0:

                fingers = []

                # Thumb
                if lm_list[4][0] > lm_list[3][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Index
                if lm_list[8][1] < lm_list[6][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Middle
                if lm_list[12][1] < lm_list[10][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Ring
                if lm_list[16][1] < lm_list[14][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Pinky
                if lm_list[20][1] < lm_list[18][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                finger_count = fingers.count(1)

    cv2.putText(
        img,
        f"Count: {finger_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        3
    )

    cv2.imshow("Finger Counter", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
