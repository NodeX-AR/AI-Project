# Smart Attendance System – CBSE Class 12 AI Project

**Author:** Aswanth R  
**Subject:** Artificial Intelligence (Code 843)  kl

---

## License

Educational Use License – see LICENSE file.  
Free for educational and evaluation purposes. Commercial use prohibited.

---

## How to Run

### Step 1 – Create Required Folders

1. Create a folder on your PC named "AI Project"
2. Download both Python files - `main.py` and `admin.py`
3. Move those files to the folder you just created
4. Open Command Prompt on your PC
5. Run this command: `pip install opencv-python numpy`
6. If issues arise, check your Python installation and pip configuration
7. Run `admin.py` first before `main.py`
8. After setup and adding students, run `main.py` to test the project

### Common Setup Errors

| Error | Problem | Fix |
|-------|---------|-----|
| Module not found | Missing library | Run `pip install opencv-python numpy` |
| Camera not opening | No webcam detected | Check camera permissions |
| No face detected | Poor lighting | Adjust lighting or background |
| pass.csv not found | First time setup | Auto-creates on first run |

### Step 2 – Run Admin Panel

Run: ```python admin.py ```

First run creates:
- `Student_image` folder (stores student face data)
- `register` folder (stores attendance CSV files)
- `pass.csv` (default admin: admin/admin123)

Admin Menu Options:
1. Add a student (captures 100 face images)
2. Delete a student
3. See today's absentees
4. Show full register
5. Logout

### Step 3 – Run Main Attendance System

Run: ```python main.py ```
If that fails, try: ``` py main.py ```

Camera will open automatically. Face detection runs every 3rd frame.
Press `q` to quit the camera.

---

## Demo Output (Sample)

```bash
PS C:\Users\ASWANTH\AI Project> python admin.py
+--------------------------------------------------------+
|               SMART ATTENDANCE SYSTEM                  |
|                     ADMIN PANEL                        |
+--------------------------------------------------------+
| 1. Login                                               |
| 2. Register                                            |
| 3. Exit                                                |
+--------------------------------------------------------+
| Enter choice: 1
+--------------------------------------------------------+
|                      LOGIN                             |
+--------------------------------------------------------+
| Username: admin
| Password: admin123
+--------------------------------------------------------+
| Welcome admin!                                  |
+--------------------------------------------------------+
+--------------------------------------------------------+
|               SMART ATTENDANCE SYSTEM                  |
|                     MAIN MENU                          |
+--------------------------------------------------------+
| 1. Add a student                                       |
| 2. Delete a student                                    |
| 3. See today's absentees                               |
| 4. Show full register                                  |
| 5. Logout                                              |
+--------------------------------------------------------+
| Your choice: 1
+--------------------------------------------------------+
| Bring the student in front of camera and press Enter... |
| Whats your name: Aswanth
+--------------------------------------------------------+
| Taking images. Please look at camera...              |
| Done! Student Aswanth added successfully!            |
+--------------------------------------------------------+
PS C:\Users\ASWANTH\AI Project> python main.py
+---------------------------------------------------------+
|               SMART ATTENDANCE SYSTEM                   |
+---------------------------------------------------------+
| Starting up systems.....                               |
| Loaded 1 students                   |
+---------------------------------------------------------+
| Systems Ready !                                        |
| Press 'q' to quit                                      |
+---------------------------------------------------------+
| Aswanth - PRESENT! (1/1) |
+---------------------------------------------------------+
| FINAL ATTENDANCE -2026-05-20                  |
+---------------------------------------------------------+
| Present: 1                                                  |
| Absent: 0                                                  |
+---------------------------------------------------------+
| Thank You for using Smart Attendance System :)
+---------------------------------------------------------+
```
---
Note: Full demo not shown as camera output is too long to display here.
