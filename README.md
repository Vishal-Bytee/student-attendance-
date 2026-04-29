## Student Attendance System

A face recognition based attendance system that automatically marks student attendance using your webcam, with real-time emotion detection for each student.

# Features

Automatic face detection and recognition via webcam
Real-time emotion detection for each student
Attendance only allowed in a fixed time window (9:30 AM – 10:00 AM)
Saves daily attendance as CSV with present/absent status
Emotion breakdown report after every session
Fallback emotion detection if DeepFace is unavailable


How It Works
Webcam Feed
      |
Detect Face (Haar Cascade)
      |
Recognize Student (Template Matching)
      |
Detect Emotion (DeepFace + Heuristics)
      |
Mark Attendance (only between 9:30–10:00 AM)
      |
Save to CSV (attendance_records/)

Technologies Used
ToolPurposePython 3.10/3.11Core programming languageOpenCVFace detection and recognitionDeepFaceEmotion detectionTensorFlowDeepFace backendPandasAttendance CSV handlingNumPyImage processing

System Requirements

Python 3.10 or 3.11 (3.12 may not support TensorFlow)
Webcam
RAM: 4GB minimum (8GB recommended)
Storage: 2GB free space
OS: Windows 10/11, Linux, macOS


Installation
1. Clone Repository
bashgit clone https://github.com/Vishal-Bytee/student-attendance-.git
cd attendance-system
2. Create Virtual Environment
bashpython -m venv attendance_env

# Windows
attendance_env\Scripts\activate

# Linux/Mac
source attendance_env/bin/activate
3. Install Dependencies
bashpip install -r requirements.txt

## How to Run
bashpython attendance.py

# Menu Options
1. Add new student (collect images)
2. Train the recognizer
3. Start attendance
4. Exit
Adding a Student

Select option 1
Enter student name
Face appears in webcam window
Press SPACE to capture (50 images needed)
Press Q to quit early

Training

Select option 2
System trains on all collected student images
Model saved to trained_models/

Taking Attendance

Select option 3
Only works between 9:30 AM and 10:00 AM
Students look at webcam — attendance marked automatically
Press Q or S to end session
CSV saved to attendance_records/


Project Structure
attendance-system/

─ attendance.py            # Main application
─ requirements.txt         # Dependencies
─ README.md                # Documentation

─ student_dataset/         # Captured student images
   ─ StudentName/
       └── 0.jpg ... 49.jpg

─ trained_models/          # Saved model templates
   ─ templates.pkl

─ attendance_records/      # Daily CSV reports
    ─ attendance_20250429.csv

CSV Output Format
NameStatusEmotionTimeJohnPresentHappy2025-04-29 09:35:22JaneAbsentN/A2025-04-29 09:35:22

# Emotions Detected
Emotion,Happy,Sad,Angry,Surprised,Fearful,Disgusted,Neutral

# Limitations

- Attendance only works in the 9:30–10:00 AM window
- Works best in good lighting conditions
- Template matching may confuse students who look similar
- Emotion detection is less accurate without DeepFace
- One face per frame gives the most accurate results


# Future Improvements

- GUI interface
- Multiple time window support
- Email report after session
- Anti-spoofing (photo attack prevention)
- Database integration instead of CSV
