import cv2
import os
import csv
from datetime import datetime
import numpy as np

print("+--------------------------------------------------------+")
print("|               SMART ATTENDANCE SYSTEM                  |")
print("+--------------------------------------------------------+")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

student_faces = []
student_names = []

if not os.path.exists("Student_image"):
    print("| Please run admin.py                                    |")
    print("+--------------------------------------------------------+")
    exit()

print("| Starting up systems.....                               |")
for student_name in os.listdir("Student_image"):
    student_path = os.path.join("Student_image", student_name)
    if os.path.isdir(student_path):
        faces_for_student = []
        for img_file in os.listdir(student_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(student_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces = face_cascade.detectMultiScale(img, 1.3, 5)
                    if len(faces) > 0:
                        (x, y, w, h) = faces[0]
                        face = img[y:y+h, x:x+w]
                        face = cv2.resize(face, (100, 100))
                        faces_for_student.append(face)
                    else:
                        face = cv2.resize(img, (100, 100))
                        faces_for_student.append(face)
        
        if faces_for_student:
            student_faces.append(faces_for_student)
            student_names.append(student_name)
            
print(f"| Loaded {len(student_names)} students                   |")
print("+--------------------------------------------------------+")

attendance = {name: "Absent" for name in student_names}
os.makedirs("register", exist_ok=True)

def compare_faces(face1, face2):
    face1 = cv2.resize(face1, (100, 100))
    face2 = cv2.resize(face2, (100, 100))
    diff = cv2.absdiff(face1, face2)
    similarity = 100 - (np.mean(diff) / 2.55)
    return similarity

cap = cv2.VideoCapture(0)
print("| Systems Ready !                                        |")
print("| Press 'q' to quit                                      |")
print("+--------------------------------------------------------+")

marked = set()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    if frame_count % 3 == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            camera_face = gray[y:y+h, x:x+w]
            camera_face = cv2.resize(camera_face, (100, 100))
            best_match = None
            best_score = 0
            
            for i, student_face_list in enumerate(student_faces):
                for ref_face in student_face_list:
                    score = compare_faces(camera_face, ref_face)
                    if score > best_score and score > 90:
                        best_score = score
                        best_match = student_names[i]
            
            if best_match and best_match not in marked:
                attendance[best_match] = "Present"
                marked.add(best_match)
                
                present = sum(1 for v in attendance.values() if v == "Present")
                print(f"| {best_match}-PRESENT! ({present}/{len(student_names)}) |")
                today = datetime.now().strftime("%Y-%m-%d")
                csv_path = os.path.join("register", f"{today}.csv")
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    for name, status in attendance.items():
                        writer.writerow([name, status])
                        
            color = (0, 255, 0) if best_match else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            if best_match:
                cv2.putText(frame, f"{best_match} ({best_score:.0f}%)", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    present = sum(1 for v in attendance.values() if v == "Present")
    cv2.putText(frame, f"Present: {present}/{len(student_names)}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Attendance System', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

present = sum(1 for v in attendance.values() if v == "Present")
print("\n+--------------------------------------------------------+")
print(f"| FINAL ATTENDANCE -{datetime.now().strftime('%Y-%m-%d')}|")
print("+--------------------------------------------------------+")
print(f"| Present: {present}                                     |")
print(f"| Absent: {len(student_names) - present}                 |")
print("+--------------------------------------------------------+")


if present < len(student_names):
    absentees = [name for name, status in attendance.items() if status == "Absent"]
    print(f"| Absentees: {', '.join(absentees)}                     |")
    
print("+--------------------------------------------------------+")
