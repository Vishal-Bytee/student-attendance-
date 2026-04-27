import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import cv2
import pandas as pd
from datetime import datetime, time
import pickle
import numpy as np
from collections import Counter

try:
    from deepface import DeepFace
    DEEPFACE_OK = True
    print("DeepFace loaded successfully")
except:
    DEEPFACE_OK = False
    print("DeepFace not available, using fallback")


class AttendanceSystem:
    def __init__(self):
        self.dataset_path = "student_dataset"
        self.models_path = "trained_models"
        self.attendance_path = "attendance_records"

        for folder in [self.dataset_path, self.models_path, self.attendance_path]:
            os.makedirs(folder, exist_ok=True)

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        self.student_templates = {}
        self.student_names = []
        self.marked_today = set()
        self.attendance_records = []
        self.detected_emotions = {}
        self.emotion_history = {}

        # attendance is only allowed between 9:30 and 10:00
        self.start_time = time(9, 30)
        self.end_time = time(10, 0)
        self.debug_mode = False

        print("="*60)
        print("       FACE RECOGNITION ATTENDANCE SYSTEM")
        print("="*60)


    def _get_gray_and_rgb(self, face_img):
        if len(face_img.shape) == 2:
            gray = cv2.resize(face_img, (200, 200))
            rgb = cv2.cvtColor(cv2.resize(face_img, (224, 224)), cv2.COLOR_GRAY2RGB)
        else:
            gray = cv2.resize(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY), (200, 200))
            rgb = cv2.cvtColor(cv2.resize(face_img, (224, 224)), cv2.COLOR_BGR2RGB)
        return gray, rgb


    def detect_emotion(self, face_img, name="unknown"):
        try:
            gray, rgb = self._get_gray_and_rgb(face_img)
            votes = []

            # deepface gives the most accurate result so try it first
            if DEEPFACE_OK:
                try:
                    result = DeepFace.analyze(rgb, actions=['emotion'], enforce_detection=False, silent=True)
                    if isinstance(result, list):
                        result = result[0]
                    top_emotion = result['dominant_emotion']
                    conf = result['emotion'][top_emotion]
                    if conf > 30:
                        votes.extend([top_emotion] * int(conf / 20))
                except:
                    pass

            smiles = self.smile_cascade.detectMultiScale(gray, 1.4, 12, minSize=(25, 25))
            eyes   = self.eye_cascade.detectMultiScale(gray, 1.05, 2, minSize=(20, 20))
            bright = np.mean(gray)

            # basic heuristic rules as backup
            if len(smiles) >= 2:
                votes.extend(['happy'] * 3)
            elif len(smiles) == 1:
                votes.extend(['happy'] * 2)

            if len(eyes) == 0 or bright < 80:
                votes.append('sad')

            if bright < 75 and len(smiles) == 0:
                votes.append('angry')

            if len(eyes) >= 3:
                votes.append('surprise')

            if 90 < bright < 130 and len(eyes) == 2 and len(smiles) == 0:
                votes.append('neutral')

            current = Counter(votes).most_common(1)[0][0] if votes else 'neutral'

            # smooth out flickering by keeping last 5 readings
            if name not in self.emotion_history:
                self.emotion_history[name] = []
            self.emotion_history[name].append(current)
            self.emotion_history[name] = self.emotion_history[name][-5:]

            if len(self.emotion_history[name]) >= 3:
                current = Counter(self.emotion_history[name]).most_common(1)[0][0]

            labels = {
                'angry': 'Angry', 'disgust': 'Disgusted',
                'fear': 'Fearful', 'happy': 'Happy',
                'sad': 'Sad', 'surprise': 'Surprised', 'neutral': 'Neutral'
            }
            return labels.get(current, 'Neutral')

        except:
            return 'Neutral'


    def collect_student_data(self, student_name, num_images=50):
        save_dir = os.path.join(self.dataset_path, student_name)
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(0)
        count = 0

        print(f"\nCapturing {num_images} images for --> {student_name}")
        print("SPACE = capture  |  Q = quit\n")

        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                emotion = self.detect_emotion(frame[y:y+h, x:x+w], student_name)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"Saved: {count}/{num_images}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            cv2.imshow('Data Collection', frame)

            key = cv2.waitKey(1)
            if key == ord(' ') and len(faces) == 1:
                x, y, w, h = faces[0]
                face_gray = gray[y:y+h, x:x+w]
                cv2.imwrite(os.path.join(save_dir, f"{count}.jpg"), face_gray)
                count += 1
                print(f"  [{count}/{num_images}] saved")
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"\nDone! {count} images saved for {student_name}\n")


    def train_recognizer(self):
        print("\nStarting training...\n")
        self.student_templates = {}
        self.student_names = []

        for student_name in os.listdir(self.dataset_path):
            folder = os.path.join(self.dataset_path, student_name)

            if not os.path.isdir(folder):
                continue
            if not student_name or student_name.isdigit() or len(student_name) < 2:
                print(f"  Skipping invalid folder: '{student_name}'")
                continue

            imgs = []
            for fname in os.listdir(folder):
                img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.equalizeHist(img)
                    imgs.append(cv2.resize(img, (100, 100)))

            if imgs:
                self.student_templates[student_name] = np.mean(imgs, axis=0).astype(np.uint8)
                self.student_names.append(student_name)
                print(f"  Trained: {student_name} ({len(imgs)} images)")

        model_file = os.path.join(self.models_path, 'templates.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump({'templates': self.student_templates, 'names': self.student_names}, f)

        print(f"\nTraining done! {len(self.student_names)} student(s) registered.\n")


    def load_model(self):
        model_file = os.path.join(self.models_path, 'templates.pkl')
        if not os.path.exists(model_file):
            print("No trained model found. Please train first.")
            return False

        with open(model_file, 'rb') as f:
            data = pickle.load(f)

        self.student_templates = data['templates']
        self.student_names = data['names']
        print(f"Model loaded — {len(self.student_names)} student(s) found.")
        return True


    def recognize_face(self, face_img):
        face = cv2.resize(cv2.equalizeHist(face_img), (100, 100))
        best_name, best_score = None, float('inf')

        for name, template in self.student_templates.items():
            score = np.sum(cv2.absdiff(face, template))
            if score < best_score:
                best_score = score
                best_name = name

        # anything above this diff is too uncertain to mark
        THRESHOLD = 1000000
        if best_score < THRESHOLD:
            confidence = max(0, min(100, 100 - (best_score / 10000)))
            return best_name, confidence

        return "Unknown", 0


    def is_attendance_time(self):
        if self.debug_mode:
            return True
        now = datetime.now().time()
        return self.start_time <= now <= self.end_time


    def mark_attendance(self, name, emotion):
        if name in self.marked_today:
            return False

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.detected_emotions[name] = emotion
        self.attendance_records.append({
            'Name': name,
            'Status': 'Present',
            'Emotion': emotion,
            'Time': timestamp
        })
        self.marked_today.add(name)
        print(f"  Marked present: {name}  |  Emotion: {emotion}")
        return True


    def run_attendance(self):
        if not self.is_attendance_time():
            print("Attendance is only allowed between 9:30 AM and 10:00 AM.")
            return

        if not self.load_model():
            return

        print("\n" + "="*60)
        print("         ATTENDANCE SESSION STARTED")
        print("="*60)
        print("Q = quit  |  S = save and exit")
        print("="*60 + "\n")

        cap = cv2.VideoCapture(0)

        while self.is_attendance_time():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

            for (x, y, w, h) in faces:
                face_gray  = gray[y:y+h, x:x+w]
                face_color = frame[y:y+h, x:x+w]

                name, confidence = self.recognize_face(face_gray)

                if confidence > 55:
                    color = (0, 255, 0)
                    if name in self.marked_today:
                        emotion = self.detected_emotions.get(name, 'Neutral')
                    else:
                        emotion = self.detect_emotion(face_color, name)
                        self.mark_attendance(name, emotion)
                else:
                    name, emotion, color = "Unknown", "N/A", (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, name,    (x, y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,  color, 2)

            cv2.putText(frame, f"Present: {len(self.marked_today)}/{len(self.student_names)}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow('Attendance System', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('s'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.save_attendance()


    def save_attendance(self):
        print("\n" + "="*60)
        print("            SAVING ATTENDANCE")
        print("="*60)

        today = datetime.now().strftime('%Y%m%d')
        filepath = os.path.join(self.attendance_path, f"attendance_{today}.csv")

        records = []
        for student in self.student_names:
            if not student or student.isdigit() or len(student) < 2:
                continue
            if student in self.marked_today:
                records.append({
                    'Name': student.strip(),
                    'Status': 'Present',
                    'Emotion': self.detected_emotions.get(student, 'Neutral'),
                    'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            else:
                records.append({
                    'Name': student.strip(),
                    'Status': 'Absent',
                    'Emotion': 'N/A',
                    'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })

        df = pd.DataFrame(records).fillna('N/A').sort_values('Name').reset_index(drop=True)
        df.to_csv(filepath, index=False)

        present = len(df[df['Status'] == 'Present'])
        absent  = len(df[df['Status'] == 'Absent'])

        print(f"\n  Total Students : {len(self.student_names)}")
        print(f"  Present        : {present}")
        print(f"  Absent         : {absent}")

        emotions = df[df['Status'] == 'Present']['Emotion'].value_counts()
        if not emotions.empty:
            print(f"\n  Emotion Breakdown:")
            for emotion, cnt in emotions.items():
                if emotion != 'N/A':
                    print(f"    {emotion}: {cnt}")

        print(f"\n  File saved: {filepath}")
        print("\n" + df.to_string(index=False))
        print("\n" + "="*60 + "\n")


def main():
    system = AttendanceSystem()

    while True:
        print("\n" + "="*60)
        print("                  MAIN MENU")
        print("="*60)
        print("  1. Add new student (collect images)")
        print("  2. Train the recognizer")
        print("  3. Start attendance")
        print("  4. Exit")
        print("="*60)

        choice = input("Enter your choice (1-4): ").strip()

        if choice == '1':
            name = input("Enter student name: ").strip()
            if name and len(name) >= 2 and not name.isdigit():
                system.collect_student_data(name)
            else:
                print("Invalid name. Use at least 2 alphabetic characters.")

        elif choice == '2':
            system.train_recognizer()

        elif choice == '3':
            system.run_attendance()

        elif choice == '4':
            print("\nGoodbye!\n")
            break

        else:
            print("Please enter a number between 1 and 4.")


if __name__ == "__main__":
    print("\nStarting Attendance System...")
    print("Working directory:", os.getcwd(), "\n")
    main()