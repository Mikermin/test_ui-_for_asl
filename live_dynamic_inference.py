import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import ollama
from collections import deque
from tensorflow.keras.models import load_model #type: ignore

# -------- CONFIG --------
MODEL_PATH = "dynamic_asl_model_demo.keras"
SCALER_PATH = "feature_scaler.save"
LABEL_PATH = "label_classes.npy"

SEQUENCE_LENGTH = 60
CONF_THRESHOLD = 0.90
STABLE_REQUIRED = 5
NO_SIGN_FRAME_LIMIT = 100  # 🔥 trigger grammar after 100 frames of no sign
# ------------------------

# -------- GRAMMAR SUBPROCESS --------
def correct_sentence(word_list):
    input_shorthand = ", ".join(word_list)
    
    # System prompt gives the model its 'rules' without hard-coding words
    system_prompt = (
        "You are a text expansion engine. Convert shorthand tokens into a "
        "natural, grammatically correct English sentence. Do not add fluff. "
        "Tokens with underscores like 'me_my' or 'dont_want' should be "
        "interpreted as their semantic equivalents (e.g., 'I', 'don't want')."
        " Include every word in the given list or at least their semantic equivalents in the output sentence. The output should be concise and to the point." \
        " DO NOT exclude any tokens from the output sentence. If a token is not directly translatable, use your best judgment to include its meaning in the sentence." \
    )

    response = ollama.chat(model='gemma3:1b', messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': f"Tokens: {word_list}"}
    ])
    
    return response['message']['content'].strip('"')
#-------------------------------------


print("Loading model...")
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
labels = np.load(LABEL_PATH)

mp_hands = mp.solutions.hands #type: ignore
mp_face = mp.solutions.face_mesh #type: ignore

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # 🔥 720p
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_queue = deque(maxlen=SEQUENCE_LENGTH)

stable_counter = 0
last_prediction = None
sentence = []
final_sentence = ""

no_sign_counter = 0

prev_time = time.time()


def extract_features(hand_results, face_results):
    frame_data = np.zeros((2, 21, 3))

    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            if idx >= 2:
                break

            handedness = hand_results.multi_handedness[idx].classification[0].label
            hand_index = 0 if handedness == "Left" else 1

            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            )

            wrist = landmarks[0]
            landmarks = landmarks - wrist
            frame_data[hand_index] = landmarks

    face_features = np.zeros(5)

    if face_results.multi_face_landmarks and hand_results.multi_hand_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0].landmark

        forehead_y = face_landmarks[10].y
        left_brow_y = face_landmarks[105].y
        right_brow_y = face_landmarks[334].y
        nose_y = face_landmarks[1].y
        mouth_y = face_landmarks[13].y
        chin_y = face_landmarks[152].y

        eyebrow_avg_y = (left_brow_y + right_brow_y) / 2.0

        thumb_y = None
        for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            handedness = hand_results.multi_handedness[idx].classification[0].label
            if handedness == "Right":
                thumb_y = hand_landmarks.landmark[4].y
                break

        if thumb_y is None and hand_results.multi_hand_landmarks:
            thumb_y = hand_results.multi_hand_landmarks[0].landmark[4].y

        face_height = chin_y - forehead_y
        if face_height != 0 and thumb_y is not None:
            face_features = np.array([
                (thumb_y - forehead_y) / face_height,
                (thumb_y - eyebrow_avg_y) / face_height,
                (thumb_y - nose_y) / face_height,
                (thumb_y - mouth_y) / face_height,
                (thumb_y - chin_y) / face_height
            ])

    return np.concatenate([frame_data.reshape(-1), face_features])


while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(rgb)
    face_results = face_mesh.process(rgb)

    # -------- HAND DETECTION CHECK --------
    if not hand_results.multi_hand_landmarks:
        no_sign_counter += 1
    else:
        no_sign_counter = 0

    features = extract_features(hand_results, face_results)
    frame_queue.append(features)

    if len(frame_queue) == SEQUENCE_LENGTH and hand_results.multi_hand_landmarks:

        sequence = np.array(frame_queue)
        seq_reshaped = sequence.reshape(-1, sequence.shape[-1])
        seq_scaled = scaler.transform(seq_reshaped)
        sequence_scaled = seq_scaled.reshape(1, SEQUENCE_LENGTH, -1)

        prediction = model.predict(sequence_scaled, verbose=0)[0]

        confidence = np.max(prediction)
        predicted_class = np.argmax(prediction)
        predicted_word = labels[predicted_class]

        if confidence > CONF_THRESHOLD:

            if predicted_word == last_prediction:
                stable_counter += 1
            else:
                stable_counter = 0
                last_prediction = predicted_word

            if stable_counter >= STABLE_REQUIRED:

                if len(sentence) == 0 or sentence[-1] != predicted_word:
                    sentence.append(predicted_word)
                    print("Current word list:", sentence)  # 🔥 Print list live

                stable_counter = 0
                frame_queue.clear()

        else:
            stable_counter = 0

        cv2.putText(frame,
                    f"{predicted_word} ({confidence:.2f})",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3)

    # -------- TRIGGER GRAMMAR --------
    if no_sign_counter > NO_SIGN_FRAME_LIMIT and len(sentence) > 0:
        print("\nSending to grammar corrector:", sentence)
        final_sentence = correct_sentence(sentence)

        print("Corrected sentence:", final_sentence, "\n")

        sentence = []
        no_sign_counter = 0

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(frame, f"FPS: {int(fps)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2)

    # Display current word list
    cv2.putText(frame,
                "Words: " + " ".join(sentence),
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2)

    # Display final corrected sentence at bottom
    cv2.putText(frame,
                "Final: " + final_sentence,
                (20, 680),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                3)

    cv2.imshow("ASL Dynamic Interpreter - Demo", frame)

    key = cv2.waitKey(1)

    if key == ord('c'):
        sentence = []
        final_sentence = ""

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()