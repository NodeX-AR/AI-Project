import os
import cv2
import csv
from datetime import datetime
import shutil
import hashlib

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_and_create_folders():
    if not os.path.exists("Student_image"):
        os.makedirs("Student_image")
        print("Created Student_image folder")
    
    if not os.path.exists("register"):
        os.makedirs("register")
        print("Created register folder")

def get_todays_csv():
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join("register", f"{today}.csv"), today

def init_pass_file():
    if not os.path.exists("pass.csv"):
        with open("pass.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["username", "password", "role"])
            writer.writerow(["admin", hash_password("admin"), "admin"]

def login():
    print("+--------------------------------------------------------+")
    print("|                      LOGIN                             |")
    print("+--------------------------------------------------------+")
    
    username = input("| Username: ").strip()
    password = input("| Password: ").strip()
    print("+--------------------------------------------------------+")
    
    if not os.path.exists("pass.csv"):
        init_pass_file()
    
    with open("pass.csv", 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2 and row[0] == username and row[1] == hash_password(password):
                print(f"| Welcome {username}!                                    |")
                print("+--------------------------------------------------------+")
                return username
    
    print("| Invalid username or password!                          |")
    print("+--------------------------------------------------------+")
    return None

def register():
    print("+--------------------------------------------------------+")
    print("|                    REGISTER                            |")
    print("+--------------------------------------------------------+")
    
    username = input("| New username: ").strip()
    
    if os.path.exists("pass.csv"):
        with open("pass.csv", 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 1 and row[0] == username:
                    print("| Username already exists!                               |")
                    print("+--------------------------------------------------------+")
                    return None
    
    password = input("| New password: ").strip()
    confirm = input("| Confirm password: ").strip()
    
    if password != confirm:
        print("| Passwords do not match!                                |")
        print("+--------------------------------------------------------+")
        return None
    
    if len(password) < 4:
        print("| Password must be at least 4 characters!                |")
        print("+--------------------------------------------------------+")
        return None
    
    with open("pass.csv", 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([username, hash_password(password), "user"])
    
    print("| Registration successful! Please login.                 |")
    print("+--------------------------------------------------------+")
    return username

def add_student():
    input("| Bring the student in front of camera and press Enter...|")
    
    name = input("| Whats your name: ").strip()
    print("+--------------------------------------------------------+")
    
    student_folder = os.path.join("Student_image", name)
    if os.path.exists(student_folder):
        print(f"| Student {name} already exists!                         |")
        print("+--------------------------------------------------------+")
        return
    
    os.makedirs(student_folder)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("| Error: Cannot access camera                            |")
        print("+--------------------------------------------------------+")
        return
    
    print("| Taking images. Please look at camera...                |")
    count = 0
    
    while count < 100:
        ret, frame = cap.read()
        if ret:
            img_path = os.path.join(student_folder, f"{name}_{count}.jpg")
            cv2.imwrite(img_path, frame)
            count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    csv_path, _ = get_todays_csv()
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            rows = list(csv.reader(f))
        
        found = False
        for row in rows:
            if row[0] == name:
                row[1] = "Present"
                found = True
                break
        
        if not found:
            rows.append([name, "Present"])
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(rows)
    
    print(f"| Done! Student {name} added successfully!               |")
    print("+--------------------------------------------------------+")

def delete_student():
    students = [f for f in os.listdir("Student_image") 
                if os.path.isdir(os.path.join("Student_image", f))]
    
    if not students:
        print("| No students found!                                     |")
        print("+--------------------------------------------------------+")
        return
    
    print("+--------------------------------------------------------+")
    print("|                    STUDENTS LIST                       |")
    print("+--------------------------------------------------------+")
    for i, s in enumerate(students, 1):
        print(f"| {i}. {s:<52} |")
    
    try:
        choice = int(input("| Select number to delete: "))
        
        if 1 <= choice <= len(students):
            name = students[choice-1]
            confirm = input(f"| Delete {name}? (y/n): ")
            
            if confirm.lower() == 'y':
                shutil.rmtree(os.path.join("Student_image", name))
                
                for csv_file in os.listdir("register"):
                    if csv_file.endswith('.csv'):
                        csv_path = os.path.join("register", csv_file)
                        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                            rows = list(csv.reader(f))
                        
                        new_rows = [row for row in rows if row[0] != name]
                        
                        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                            csv.writer(f).writerows(new_rows)
                
                print(f"| Deleted {name}                                         |")
                print("+--------------------------------------------------------+")
    except:
        print("| Invalid choice                                         |")
        print("+--------------------------------------------------------+")

def show_absentees():
    csv_path, today = get_todays_csv()
    
    if not os.path.exists(csv_path):
        print(f"| No attendance record for {today}                       |")
        print("+--------------------------------------------------------+")
        return
    
    absent = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[1] == "Absent":
                absent.append(row[0])
    
    print("+--------------------------------------------------------+")
    if absent:
        print(f"| Today's Absentees ({today}):                           |")
        print("+--------------------------------------------------------+")
        for s in absent:
            print(f"|    • {s:<52} |")
    else:
        print("| Everyone is present today!                             |")
    print("+--------------------------------------------------------+")

def show_register():
    csv_path, today = get_todays_csv()
    
    if not os.path.exists(csv_path):
        print(f"| No attendance record for {today}                       |")
        print("+--------------------------------------------------------+")
        return
    
    print("+--------------------------------------------------------+")
    print(f"| ATTENDANCE REGISTER - {today:<26}                      |")
    print("+--------------------------------------------------------+")
    
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                status = "Present" if row[1] == "Present" else "Absent"
                print(f"| {row[0]:<20} : {status:<22} |")
    print("+--------------------------------------------------------+")

check_and_create_folders()
init_pass_file()

while True:
    print("+--------------------------------------------------------+")
    print("|               SMART ATTENDANCE SYSTEM                  |")
    print("|                     ADMIN PANEL                        |")
    print("+--------------------------------------------------------+")
    print("| 1. Login                                               |")
    print("| 2. Register                                            |")
    print("| 3. Exit                                                |")
    print("+--------------------------------------------------------+")
    
    choice = input("| Enter choice: ").strip()
    print("+--------------------------------------------------------+")
    
    if choice == "1":
        logged_in_user = login()
        if logged_in_user:
            break
    elif choice == "2":
        register()
    elif choice == "3":
        print("| Exiting...                                             |")
        print("+--------------------------------------------------------+")
        exit()
    else:
        print("| Invalid choice! Please select 1-3                      |")
        print("+--------------------------------------------------------+")

while True:
    print("+--------------------------------------------------------+")
    print("|               SMART ATTENDANCE SYSTEM                  |")
    print("|                     MAIN MENU                          |")
    print("+--------------------------------------------------------+")
    print("| 1. Add a student                                       |")
    print("| 2. Delete a student                                    |")
    print("| 3. See today's absentees                               |")
    print("| 4. Show full register                                  |")
    print("| 5. Logout                                              |")
    print("+--------------------------------------------------------+")
    
    choice = input("| Your choice: ").strip()
    print("+--------------------------------------------------------+")
    
    if choice == "1":
        add_student()
    elif choice == "2":
        delete_student()
    elif choice == "3":
        show_absentees()
    elif choice == "4":
        show_register()
    elif choice == "5":
        print("| Logging out...                                         |")
        print("+--------------------------------------------------------+")
        while True:
            print("+--------------------------------------------------------+")
            print("|               SMART ATTENDANCE SYSTEM                  |")
            print("|                     ADMIN PANEL                        |")
            print("+--------------------------------------------------------+")
            print("| 1. Login                                               |")
            print("| 2. Register                                            |")
            print("| 3. Exit                                                |")
            print("+--------------------------------------------------------+")
            
            choice = input("| Enter choice: ").strip()
            print("+--------------------------------------------------------+")
            
            if choice == "1":
                logged_in_user = login()
                if logged_in_user:
                    break
            elif choice == "2":
                register()
            elif choice == "3":
                print("| Exiting...                                             |")
                print("+--------------------------------------------------------+")
                exit()
            else:
                print("| Invalid choice! Please select 1-3                      |")
                print("+--------------------------------------------------------+")
        continue
    else:
        print("| Invalid choice! Please select 1-5                      |")
        print("+--------------------------------------------------------+")
    
    input("| Press Enter to continue...                             |")
    print("+--------------------------------------------------------+")
