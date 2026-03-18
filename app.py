import base64
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import ollama
from collections import deque
from tensorflow.keras.models import load_model #type: ignore
from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading

app = Flask(__name__)
# Enable CORS for the React Dev Server port
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=50000000, max_decode_packets=500, async_mode='threading')

# -------- CONFIG --------
MODEL_PATH = "dynamic_asl_model_demo.keras"
SCALER_PATH = "feature_scaler.save"
LABEL_PATH = "label_classes.npy"

SEQUENCE_LENGTH = 60
CONF_THRESHOLD = 0.90
STABLE_REQUIRED = 5
NO_SIGN_FRAME_LIMIT = 45
# ------------------------

# -------- GRAMMAR SUBPROCESS --------
def correct_sentence(word_list):
    input_shorthand = ", ".join(word_list)
    system_prompt = (
        "You are a text expansion engine. Convert shorthand tokens into a "
        "natural, grammatically correct English sentence. Do not add fluff. "
        "Tokens with underscores like 'me_my' or 'dont_want' should be "
        "interpreted as their semantic equivalents (e.g., 'I', 'don't want')."
        " Include every word in the given list or at least their semantic equivalents in the output sentence. The output should be concise and to the point." \
        " DO NOT exclude any tokens from the output sentence. If a token is not directly translatable, use your best judgment to include its meaning in the sentence." \
    )

    try:
        response = ollama.chat(model='gemma:2b', messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Tokens: {word_list}"}
        ])
        return response['message']['content'].strip('"')
    except Exception as e:
        print(f"Ollama Error: {e}")
        return " ".join(word_list) # Fallback to raw words if ollama fails

print("Loading model...")
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
labels = np.load(LABEL_PATH)

mp_hands = mp.solutions.hands #type: ignore
mp_face = mp.solutions.face_mesh #type: ignore

mp_lock = threading.Lock()

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.85,
    min_tracking_confidence=0.85
)

face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Manage state per connected client (in a real app you'd map by request.sid)
# Since this is local UI we keep it simple with global state for a single connection.
client_state = {
    'frame_queue': deque(maxlen=SEQUENCE_LENGTH),
    'stable_counter': 0,
    'last_prediction': None,
    'sentence': [],
    'no_sign_counter': 0,
    'prev_time': time.time()
}

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

@socketio.on('connect')
def test_connect():
    print('Client connected')
    client_state['frame_queue'].clear()
    client_state['stable_counter'] = 0
    client_state['last_prediction'] = None
    client_state['sentence'] = []
    client_state['no_sign_counter'] = 0

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

@socketio.on('clear_text')
def handle_clear_text():
    client_state['sentence'] = []
    emit('current_words', {'words': []})
    emit('final_sentence', {'text': ''})

@socketio.on('video_frame')
def handle_video_frame(data):
    # data is expected to be a dict with a 'image' key containing base64 string
    img_data = data['image'].split(',')[1]
    nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return

    # React frontend standardizes orientation, but mediapipe expects RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_lock:
        hand_results = hands.process(rgb)
        face_results = face_mesh.process(rgb)

    # -------- HAND DETECTION CHECK --------
    if not hand_results.multi_hand_landmarks:
        client_state['no_sign_counter'] += 1
        if client_state['no_sign_counter'] > 5:
            client_state['frame_queue'].clear()
            client_state['stable_counter'] = 0
            # Don't append 0s when no hand is present for a while
    else:
        client_state['no_sign_counter'] = 0
        features = extract_features(hand_results, face_results)
        client_state['frame_queue'].append(features)

    # Manage inference
    current_prediction = "None"
    current_confidence = 0.0

    if len(client_state['frame_queue']) == SEQUENCE_LENGTH and hand_results.multi_hand_landmarks:
        sequence = np.array(client_state['frame_queue'])
        seq_reshaped = sequence.reshape(-1, sequence.shape[-1])
        seq_scaled = scaler.transform(seq_reshaped)
        sequence_scaled = seq_scaled.reshape(1, SEQUENCE_LENGTH, -1)

        prediction = model.predict(sequence_scaled, verbose=0)[0]
        current_confidence = float(np.max(prediction))
        predicted_class = int(np.argmax(prediction))
        predicted_word = str(labels[predicted_class])
        
        current_prediction = predicted_word

        if current_confidence > CONF_THRESHOLD:
            if predicted_word == client_state['last_prediction']:
                client_state['stable_counter'] += 1
            else:
                client_state['stable_counter'] = 0
                client_state['last_prediction'] = predicted_word

            if client_state['stable_counter'] >= STABLE_REQUIRED:
                if len(client_state['sentence']) == 0 or client_state['sentence'][-1] != predicted_word:
                    client_state['sentence'].append(predicted_word)
                    print("Current word list:", client_state['sentence'])
                    emit('current_words', {'words': client_state['sentence']})

                client_state['stable_counter'] = 0
                client_state['frame_queue'].clear()
        else:
            client_state['stable_counter'] = 0

    # FPS Calculation
    curr_time = time.time()
    fps = 1.0 / (curr_time - client_state['prev_time']) if curr_time > client_state['prev_time'] else 0.0
    client_state['prev_time'] = curr_time

    # Always emit live stats
    emit('live_stats', {
        'fps': round(fps),
        'prediction': current_prediction,
        'confidence': current_confidence,
        'hand_visible': bool(hand_results.multi_hand_landmarks)
    })

    # -------- TRIGGER GRAMMAR --------
    if client_state['no_sign_counter'] > NO_SIGN_FRAME_LIMIT and len(client_state['sentence']) > 0:
        # Prevent re-entrancy by grabbing the sentence and clearing the state immediately
        sentence_to_process = list(client_state['sentence'])
        client_state['sentence'] = []
        client_state['no_sign_counter'] = 0

        print("\nSending to grammar corrector:", sentence_to_process)
        
        # We can notify frontend that Ollama is processing (optional loading state)
        emit('processing_grammar', {'status': True})
        
        final_sentence = correct_sentence(sentence_to_process)
        print("Corrected sentence:", final_sentence, "\n")
        
        emit('final_sentence', {'text': final_sentence})
        emit('processing_grammar', {'status': False})

if __name__ == '__main__':
    print("Starting Flask-SocketIO server on port 5000...")
    socketio.run(app, host='0.0.0.0', port=5000)
