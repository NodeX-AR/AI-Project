import face_recognition
import cv2
import os
from datetime import datetime

# Configuration
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.txt"

def load_known_faces():
    known_encodings = []
    known_names = []
    
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        
    for filename in os.listdir(KNOWN_FACES_DIR):
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{filename}")
        encoding = face_recognition.face_encodings(image)[0]
        known_encodings.append(encoding)
        known_names.append(os.path.splitext(filename)[0])
    return known_encodings, known_names

def register_attendance(name):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(ATTENDANCE_FILE, "a") as f:
        f.write(f"{name}, {timestamp}\n")
    print(f"✅ Attendance recorded for {name} at {timestamp}")

def main():
    known_encodings, known_names = load_known_faces()
    video_capture = cv2.VideoCapture(0)

    print("--- Attendance System Active (Press 'q' to quit) ---")

    while True:
        ret, frame = video_capture.read()
        # Convert BGR to RGB for face_recognition library
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]

            if name != "Unknown":
                # Prompt 1: Verify Identity
                confirm = input(f"Is this {name}? [Y/N]: ").strip().upper()
                if confirm == 'Y':
                    register_attendance(name)
                else:
                    print("Skipping attendance...")
            else:
                # Prompt 2: New Person Registration
                print("Face not recognized.")
                new_name = input("Enter name to register this person (or enter to skip): ").strip()
                if new_name:
                    # Save the current frame as their reference photo
                    img_path = f"{KNOWN_FACES_DIR}/{new_name}.jpg"
                    cv2.imwrite(img_path, frame)
                    print(f"Registered {new_name}! Restarting to update database...")
                    video_capture.release()
                    cv2.destroyAllWindows()
                    return main() # Reload database

        cv2.imshow('Attendance System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
