# Smart Attendance System (Scaffold)

This repository is a scaffold for a Smart Attendance System using Flask and face_recognition.

Quick start:

1. Create a virtualenv and install dependencies:
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
2. Run the app:
```
python app.py
```

Notes:
- `face_recognition` requires dlib and proper build toolchain on Windows; follow its install docs if pip install fails.
- This scaffold implements core routes, face registration (base64 image upload), and recognition endpoints.
# Smart Attendance System (X10 THINK)

This is a starter full-stack Smart Attendance System using Flask, OpenCV and `face_recognition` for face-based attendance.

Quick start (recommended in a virtualenv):

1. Install system dependencies for `face_recognition` (dlib). On Windows, follow the library instructions or use prebuilt wheels.

2. Install Python deps:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

Open http://127.0.0.1:5000

Notes:
- Login credentials per organization are hardcoded in `app.py` per the spec.
- This scaffold demonstrates core flows: splash, login, school dashboard, add classes/students, face register, mark attendance.
- Further work: College/Institution modules, OTP for forgot password, PDF report generation, improved UI/UX and security.
