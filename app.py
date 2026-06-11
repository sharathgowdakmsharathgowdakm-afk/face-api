from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file, flash
from flask_sqlalchemy import SQLAlchemy
from flask_talisman import Talisman
from flask_wtf.csrf import CSRFProtect
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import os
import cv2
import numpy as np
import pickle
import random
import string
import face_recognition
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib import colors
from functools import wraps
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

secret_key = os.environ.get('SECRET_KEY')
flask_env = os.environ.get('FLASK_ENV', 'development')

if flask_env == 'production':
    if not secret_key or secret_key == 'your-secret-key-here' or len(secret_key) < 32:
        raise ValueError("A strong SECRET_KEY (at least 32 characters) must be set in environment variables for production environments.")
    app.config['SECRET_KEY'] = secret_key
    # Enforce HTTPS and secure headers in production
    Talisman(app, content_security_policy=None)
else:
    app.config['SECRET_KEY'] = secret_key or 'dev-fallback-secret-key-for-local-testing'

csrf = CSRFProtect(app)
# Support production Postgres or SQLite locally
database_url = os.environ.get('DATABASE_URL', 'sqlite:///attendance.db')
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

db = SQLAlchemy(app)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('face_encodings', exist_ok=True)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    org_type = db.Column(db.String(20), nullable=False)  # school, college, institution
    organization_id = db.Column(db.Integer, db.ForeignKey('organization.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Organization(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(20), nullable=False)  # school, college, institution
    logo_path = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    users = db.relationship('User', backref='organization', lazy=True)
    classes = db.relationship('Class_', backref='organization', lazy=True)
    students = db.relationship('Student', backref='organization', lazy=True)
    courses = db.relationship('Course', backref='organization', lazy=True)

class Course(db.Model):
    __tablename__ = 'course'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    organization_id = db.Column(db.Integer, db.ForeignKey('organization.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    classes = db.relationship('Class_', backref='course', lazy=True)
    subjects = db.relationship('Subject', backref='course', lazy=True)
    study_years = db.relationship('StudyYear', backref='course', lazy=True)

class StudyYear(db.Model):
    __tablename__ = 'study_year'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    organization_id = db.Column(db.Integer, db.ForeignKey('organization.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Branch(db.Model):
    __tablename__ = 'branch'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    organization_id = db.Column(db.Integer, db.ForeignKey('organization.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Semester(db.Model):
    __tablename__ = 'semester'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    organization_id = db.Column(db.Integer, db.ForeignKey('organization.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Subject(db.Model):
    __tablename__ = 'subject'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    study_year = db.Column(db.String(50))
    organization_id = db.Column(db.Integer, db.ForeignKey('organization.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Class_(db.Model):
    __tablename__ = 'class'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    academic_year = db.Column(db.String(20), nullable=False)
    study_year = db.Column(db.String(50), nullable=True)
    branch = db.Column(db.String(100), nullable=True)
    semester = db.Column(db.String(100), nullable=True)
    organization_id = db.Column(db.Integer, db.ForeignKey('organization.id'), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    students = db.relationship('Student', backref='class_', lazy=True)
    attendances = db.relationship('Attendance', backref='class_', lazy=True)

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    roll_number = db.Column(db.String(20), nullable=False)
    phone = db.Column(db.String(15))
    password = db.Column(db.String(200))
    organization_id = db.Column(db.Integer, db.ForeignKey('organization.id'), nullable=False)
    class_id = db.Column(db.Integer, db.ForeignKey('class.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    attendances = db.relationship('Attendance', backref='student', lazy=True)
    face_encodings = db.relationship('FaceEncoding', backref='student', lazy=True)

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    class_id = db.Column(db.Integer, db.ForeignKey('class.id'), nullable=False)
    subject_id = db.Column(db.Integer, db.ForeignKey('subject.id'), nullable=True)
    date = db.Column(db.Date, nullable=False)
    time = db.Column(db.Time, nullable=False)
    _status = db.Column('status', db.String(10), nullable=False)  # present, absent
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    subject = db.relationship('Subject', backref='attendances', lazy=True)
    
    @property
    def status(self):
        from datetime import time
        if self._status == 'present' and self.time:
            if self.time < time(6, 0, 0) or self.time >= time(18, 0, 0):
                return 'absent'
        return self._status

    @status.setter
    def status(self, value):
        self._status = value

    @property
    def day(self):
        return self.date.strftime('%A')

class FaceEncoding(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    encoding_path = db.Column(db.String(200), nullable=True)
    encoding_data = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Template Filters
@app.template_filter('strftime')
def _jinja2_filter_datetime(date, fmt=None):
    if date == 'now':
        date = datetime.now()
    return date.strftime(fmt)

# Decorators
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def org_required(org_types):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'org_type' not in session or session['org_type'] not in org_types:
                flash('Unauthorized access', 'danger')
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Routes
@app.route('/')
def splash():
    return render_template('splash.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        org_type = request.form.get('org_type')
        username = request.form.get('username')
        password = request.form.get('password')

        if not all([org_type, username, password]):
            flash('All fields are required', 'danger')
            return redirect(url_for('login'))

        user = User.query.filter_by(username=username, org_type=org_type).first()

        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['org_type'] = user.org_type
            session['org_id'] = user.organization_id
            session['org_name'] = user.organization.name

            if org_type == 'school':
                return redirect(url_for('school_dashboard'))
            elif org_type == 'college':
                return redirect(url_for('college_dashboard'))
            elif org_type == 'institution':
                return redirect(url_for('institution_dashboard'))
        else:
            flash('Invalid credentials', 'danger')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        org_name = request.form.get('org_name')
        org_type = request.form.get('org_type')
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if not all([org_name, org_type, username, email, password]):
            flash('All fields are required', 'danger')
            return redirect(url_for('register'))

        # Create organization
        org = Organization(name=org_name, type=org_type)
        db.session.add(org)
        db.session.commit()

        # Create user
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            org_type=org_type,
            organization_id=org.id
        )
        db.session.add(user)
        db.session.commit()

        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

# School Routes
@app.route('/school/dashboard')
@org_required(['school'])
def school_dashboard():
    classes = Class_.query.filter_by(organization_id=int(session.get('org_id'))).all()
    total_classes = len(classes)
    total_students = Student.query.filter_by(organization_id=int(session.get('org_id'))).count()
    today_attendance = Attendance.query.filter_by(
        date=datetime.now().date()
    ).join(Student).filter_by(organization_id=int(session.get('org_id'))).count()

    return render_template('school/dashboard.html',
                         classes=classes,
                         total_classes=total_classes,
                         total_students=total_students,
                         today_attendance=today_attendance)

@app.route('/school/classes')
@org_required(['school'])
def school_classes():
    classes = Class_.query.filter_by(organization_id=session.get('org_id')).all()
    for c in classes:
        c.student_count = len(c.students)
    return render_template('school/classes.html', classes=classes)

@app.route('/school/add-class', methods=['GET', 'POST'])
@org_required(['school'])
def school_add_class():
    if request.method == 'POST':
        class_name = request.form.get('class_name')
        academic_year = request.form.get('academic_year')

        if not all([class_name, academic_year]):
            flash('All fields are required', 'danger')
            return redirect(url_for('school_add_class'))

        class_ = Class_(
            name=class_name,
            academic_year=academic_year,
            organization_id=session.get('org_id')
        )
        db.session.add(class_)
        db.session.commit()

        flash('Class added successfully!', 'success')
        return redirect(url_for('school_dashboard'))

    return render_template('school/add_class.html')

@app.route('/school/add-student', methods=['GET', 'POST'])
@org_required(['school'])
def school_add_student():
    classes = Class_.query.filter_by(organization_id=session.get('org_id')).all()

    if request.method == 'POST':
        name = request.form.get('name')
        roll_number = request.form.get('roll_number')
        phone = request.form.get('phone')
        class_id = request.form.get('class_id')

        if not all([name, roll_number, class_id]):
            flash('Name, roll number and class are required', 'danger')
            return redirect(url_for('school_add_student'))

        student = Student(
            name=name,
            roll_number=roll_number,
            phone=phone,
            organization_id=session.get('org_id'),
            class_id=int(class_id)
        )
        db.session.add(student)
        db.session.commit()

        flash('Student added successfully!', 'success')
        return redirect(url_for('school_dashboard'))

    return render_template('school/add_student.html', classes=classes)

@app.route('/school/face-register', methods=['GET', 'POST'])
@org_required(['school'])
def school_face_register():
    classes = Class_.query.filter_by(organization_id=session.get('org_id')).all()
    students = Student.query.filter_by(
        organization_id=session.get('org_id')
    ).all()

    if request.method == 'POST':
        student_id = request.form.get('student_id', '')

        if not student_id:
            return jsonify({'error': 'Student not selected'}), 400

        if 'face_image' not in request.files:
            return jsonify({'error': 'No face image provided'}), 400

        file = request.files['face_image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Load image using face_recognition
            image = face_recognition.load_image_file(filepath)
            
            # Find face encodings
            face_encodings = face_recognition.face_encodings(image)

            if len(face_encodings) == 0:
                return jsonify({'error': 'No face detected in image'}), 400

            # Use the first detected face encoding
            face_encoding = face_encodings[0]

            # Save face encoding to database as JSON list
            face_record = FaceEncoding(
                student_id=int(student_id),
                encoding_path="",
                encoding_data=face_encoding.tolist(),
                created_at=datetime.utcnow()
            )
            db.session.add(face_record)
            db.session.commit()

            return jsonify({'success': 'Face registered successfully'})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('school/face_register.html', students=students, classes=classes)

@app.route('/school/mark-attendance', methods=['GET', 'POST'])
@org_required(['school'])
def school_mark_attendance():
    classes = Class_.query.filter_by(
        organization_id=session.get('org_id')
    ).all()

    if request.method == 'POST':
        class_id = request.form.get('class_id', '')
        print(f"DEBUG: Marking attendance for Class ID: {class_id}, Org ID: {session.get('org_id')}")

        if not class_id:
            return jsonify({'error': 'Class not selected'}), 400

        if 'attendance_image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['attendance_image']
        try:
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_attendance.jpg')
            file.save(temp_path)

            # Load image using face_recognition
            image = face_recognition.load_image_file(temp_path)
            
            # Find all face locations and encodings in the image
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            if not face_encodings:
                return jsonify({'error': 'No faces detected'}), 400

            students = Student.query.filter_by(class_id=int(class_id)).all()
            
            # Load all stored encodings for this class
            known_encodings = []
            known_students = []
            
            for student in students:
                face_recs = FaceEncoding.query.filter_by(student_id=student.id).all()
                for face_rec in face_recs:
                    try:
                        if face_rec.encoding_data:
                            encoding = np.array(face_rec.encoding_data, dtype=np.float64)
                        else:
                            with open(face_rec.encoding_path, 'rb') as f:
                                encoding = pickle.load(f)
                        # Verify this is a 128-d encoding from face_recognition
                        if isinstance(encoding, np.ndarray) and encoding.shape == (128,):
                            known_encodings.append(encoding)
                            known_students.append(student)
                    except Exception:
                        continue

            if not known_encodings:
                return jsonify({'error': 'No registered faces found for this class'}), 400

            recognized_students = []

            for face_encoding in face_encodings:
                # Compare detected face with all known faces
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                
                if True in matches:
                    # Use the smallest distance to find the best match
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        best_match_student = known_students[best_match_index]
                        
                        # Check if already processed in this batch to prevent duplicates
                        already_recognized = any(s['roll_number'] == best_match_student.roll_number for s in recognized_students)
                        
                        if not already_recognized:
                            existing = Attendance.query.filter_by(
                                student_id=best_match_student.id,
                                date=datetime.now().date()
                            ).first()

                            if not existing:
                                attendance = Attendance(
                                    student_id=best_match_student.id,
                                    class_id=int(class_id),
                                    date=datetime.now().date(),
                                    time=datetime.now().time(),
                                    status='present',
                                )
                                db.session.add(attendance)
                                print(f"DEBUG: Created NEW attendance record for {best_match_student.name}")
                                status_str = attendance.status
                            else:
                                print(f"DEBUG: Student {best_match_student.name} already processed today")
                                status_str = 'already present' if existing.status == 'present' else 'absent'
                            
                            recognized_students.append({
                                'name': best_match_student.name,
                                'roll_number': best_match_student.roll_number,
                                'status': status_str
                            })

            db.session.commit()
            print(f"DEBUG: Successfully committed changes for {len(recognized_students)} recognized students.")
            print(f"DEBUG: Successfully marked attendance for {len(recognized_students)} students. Database committed.")
            return jsonify({
                'success': f'Attendance marked for {len(recognized_students)} students',
                'recognized': recognized_students
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('school/mark_attendance.html', classes=classes)

@app.route('/school/attendance-records')
@org_required(['school'])
def school_attendance_records():
    classes = Class_.query.filter_by(organization_id=session.get('org_id')).all()
    class_id = request.args.get('class_id')
    date_str = request.args.get('date')
    month_str = request.args.get('month')

    query = Attendance.query.join(Class_).filter(Class_.organization_id == int(session.get('org_id')))
    if class_id:
        query = query.filter(Attendance.class_id == class_id)
    date_obj = None
    if date_str:
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
            query = query.filter(Attendance.date == date_obj)
        except ValueError:
            pass
    if month_str:
        try:
            from calendar import monthrange
            from datetime import date
            y_val, m_val = map(int, month_str.split('-'))
            _, last_day = monthrange(y_val, m_val)
            query = query.filter(Attendance.date.between(date(y_val, m_val, 1), date(y_val, m_val, last_day)))
        except Exception:
            pass

    # If class_id and date_str are both provided, show ALL students of that class (including virtual absent ones)
    if class_id and date_obj:
        students = Student.query.filter_by(class_id=int(class_id)).all()
        present_records = Attendance.query.filter_by(class_id=int(class_id), date=date_obj).all()
        present_student_ids = {r.student_id: r for r in present_records}
        
        records = []
        for s in students:
            if s.id in present_student_ids:
                records.append(present_student_ids[s.id])
            else:
                from types import SimpleNamespace
                absent_rec = SimpleNamespace(
                    id=None,
                    date=date_obj,
                    day=date_obj.strftime('%A'),
                    time=None,
                    status='absent',
                    student=s,
                    class_=s.class_,
                    subject=None
                )
                records.append(absent_rec)
    else:
        records = query.order_by(Attendance.date.desc(), Attendance.time.desc()).all()
        
    print(f"DEBUG: Found {len(records)} attendance records for Org ID: {session.get('org_id')}")
    return render_template('school/attendance_records.html', records=records, classes=classes, selected_class=class_id, selected_date=date_str, selected_month=month_str)

@app.route('/school/reports', methods=['GET', 'POST'])
@org_required(['school'])
def school_reports():
    if request.method == 'POST':
        report_type = request.form.get('report_type', 'student')
        student_id = request.form.get('student_id')
        class_id = request.form.get('class_id')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'],
                                     fontSize=16, spaceAfter=20)
        normal_style = styles['Normal']
        
        present_style = ParagraphStyle(
            'PresentStyle',
            parent=normal_style,
            textColor=colors.HexColor('#2e7d32'),
            fontName='Helvetica-Bold',
            alignment=1
        )
        absent_style = ParagraphStyle(
            'AbsentStyle',
            parent=normal_style,
            textColor=colors.HexColor('#c62828'),
            fontName='Helvetica-Bold',
            alignment=1
        )
        
        if report_type == 'class':
            elements.append(Paragraph("Class Attendance Report", title_style))
            if class_id:
                class_ = Class_.query.get(int(class_id))
                elements.append(Paragraph(f"Class: {class_.name}", normal_style))
                if start_date and end_date:
                    elements.append(Paragraph(f"Period: {start_date} to {end_date}", normal_style))
                elements.append(Spacer(1, 0.2*inch))
                
                # Fetch class students
                students = Student.query.filter_by(class_id=int(class_id)).order_by(Student.roll_number, Student.name).all()
                
                # Fetch present records
                att_query = Attendance.query.filter_by(class_id=int(class_id))
                if start_date and end_date:
                    att_query = att_query.filter(Attendance.date.between(start_date, end_date))
                present_attendances = att_query.all()
                
                # Get unique sessions (date, subject_id)
                sessions = sorted(list(set((att.date, att.subject_id) for att in present_attendances)), key=lambda x: (x[0], x[1] or 0), reverse=True)
                
                present_table_data = [['S.No', 'Date', 'Roll No', 'Student Name', 'Subject', 'Status']]
                absent_table_data = [['S.No', 'Date', 'Roll No', 'Student Name', 'Subject', 'Status']]
                
                pres_sno = 1
                abs_sno = 1
                
                for session_date, subj_id in sessions:
                    session_present = [att for att in present_attendances if att.date == session_date and att.subject_id == subj_id]
                    present_map = {att.student_id: att for att in session_present}
                    
                    subj_name = '-'
                    if subj_id:
                        subj_obj = Subject.query.get(subj_id)
                        if subj_obj:
                            subj_name = subj_obj.name
                            
                    for s in students:
                        if s.id in present_map:
                            present_table_data.append([
                                str(pres_sno),
                                str(session_date),
                                s.roll_number,
                                s.name,
                                subj_name,
                                Paragraph('Present', present_style)
                            ])
                            pres_sno += 1
                        else:
                            absent_table_data.append([
                                str(abs_sno),
                                str(session_date),
                                s.roll_number,
                                s.name,
                                subj_name,
                                Paragraph('Absent', absent_style)
                            ])
                            abs_sno += 1
                            
                has_records = False
                
                # Add Absent Students List
                if len(absent_table_data) > 1:
                    elements.append(Paragraph("Absent Students List (S.No wise)", ParagraphStyle('SubTitleAbs', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor('#c62828'), spaceAfter=10)))
                    abs_table = Table(absent_table_data)
                    abs_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#c62828')),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0,0), (-1,0), 10),
                        ('BOTTOMPADDING', (0,0), (-1,0), 8),
                        ('TOPPADDING', (0,0), (-1,0), 8),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#bdc3c7')),
                        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#fff5f5')])
                    ]))
                    elements.append(abs_table)
                    has_records = True
                    
                # Add Present Students List
                if len(present_table_data) > 1:
                    if has_records:
                        elements.append(Spacer(1, 0.3*inch))
                    elements.append(Paragraph("Present Students List (S.No wise)", ParagraphStyle('SubTitlePres', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor('#2e7d32'), spaceAfter=10)))
                    pres_table = Table(present_table_data)
                    pres_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2e7d32')),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0,0), (-1,0), 10),
                        ('BOTTOMPADDING', (0,0), (-1,0), 8),
                        ('TOPPADDING', (0,0), (-1,0), 8),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#bdc3c7')),
                        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5fff5')])
                    ]))
                    elements.append(pres_table)
                    has_records = True
                    
                if not has_records:
                    elements.append(Paragraph("No attendance records found.", normal_style))
        else:
            elements.append(Paragraph("Student Attendance Report", title_style))
            if student_id:
                student = Student.query.get(int(student_id))
                elements.append(Paragraph(f"Student: {student.name}", normal_style))
                elements.append(Paragraph(f"Roll Number: {student.roll_number}", normal_style))
                if start_date and end_date:
                    elements.append(Paragraph(f"Period: {start_date} to {end_date}", normal_style))
                elements.append(Spacer(1, 0.2*inch))
                
                # Header row for main table
                data = [['Date', 'Time', 'Subject', 'Status']]
                
                # Get the class sessions for the student's class
                att_query = Attendance.query.filter_by(class_id=student.class_id)
                if start_date and end_date:
                    att_query = att_query.filter(Attendance.date.between(start_date, end_date))
                class_attendances = att_query.all()
                
                sessions = sorted(list(set((att.date, att.subject_id) for att in class_attendances)), key=lambda x: (x[0], x[1] or 0), reverse=True)
                student_present_map = {(att.date, att.subject_id): att for att in class_attendances if att.student_id == student.id}
                
                for session_date, subj_id in sessions:
                    subj_name = '-'
                    if subj_id:
                        subj_obj = Subject.query.get(subj_id)
                        if subj_obj:
                            subj_name = subj_obj.name
                            
                    if (session_date, subj_id) in student_present_map:
                        att = student_present_map[(session_date, subj_id)]
                        data.append([
                            str(session_date),
                            att.time.strftime('%H:%M:%S') if att.time else '-',
                            subj_name,
                            Paragraph('Present', present_style)
                        ])
                    else:
                        data.append([
                            str(session_date),
                            '-',
                            subj_name,
                            Paragraph('Absent', absent_style)
                        ])

                if len(data) > 1:
                    table = Table(data)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a252f')),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0,0), (-1,0), 11),
                        ('BOTTOMPADDING', (0,0), (-1,0), 8),
                        ('TOPPADDING', (0,0), (-1,0), 8),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#bdc3c7')),
                        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')])
                    ]))
                    elements.append(table)
                else:
                    elements.append(Paragraph("No attendance records found.", normal_style))

        doc.build(elements)
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name='attendance_report.pdf',
                         mimetype='application/pdf')
    
    students = Student.query.filter_by(organization_id=session.get('org_id')).all()
    classes = Class_.query.filter_by(organization_id=session.get('org_id')).all()
    return render_template('school/reports.html', students=students, classes=classes)

@app.route('/school/students')
@org_required(['school'])
def school_students():
    class_id = request.args.get('class_id')
    query = Student.query.filter_by(organization_id=session.get('org_id'))
    if class_id:
        query = query.filter_by(class_id=class_id)
    students = query.all()
    return render_template('school/students.html', students=students)

# ─────────────────────────────────────────────
# COLLEGE ROUTES
# ─────────────────────────────────────────────

@app.route('/college/dashboard')
@org_required(['college'])
def college_dashboard():
    org_id = session.get('org_id')
    courses = Course.query.filter_by(organization_id=org_id).all()
    
    course_id = request.args.get('course_id')
    year = request.args.get('year')
    
    if not course_id or course_id == 'None':
        return render_template('college/select_course.html', courses=courses)
        
    try:
        course = Course.query.get(int(course_id))
    except (ValueError, TypeError):
        flash("Invalid course selection.", "danger")
        return redirect(url_for('college_dashboard'))
        
    if not course:
        flash("Course not found.", "danger")
        return redirect(url_for('college_dashboard'))
    
    if not year:
        study_years_records = StudyYear.query.filter_by(organization_id=org_id, course_id=course_id).all()
        db_years = [y.name for y in study_years_records]
        
        # Include legacy years from classes/subjects just in case
        classes = Class_.query.filter_by(organization_id=org_id, course_id=course_id).all()
        subjects = Subject.query.filter_by(organization_id=org_id, course_id=course_id).all()
        class_years = [c.study_year for c in classes if c.study_year]
        subj_years = [s.study_year for s in subjects if s.study_year]
        
        years = sorted(list(set(db_years + class_years + subj_years)))
        return render_template('college/select_year.html', course=course, years=years, study_years=study_years_records)
        
    classes = Class_.query.filter_by(organization_id=org_id, course_id=course_id, study_year=year).all()
    total_classes = len(classes)
    class_ids = [c.id for c in classes]
    total_students = Student.query.filter(Student.class_id.in_(class_ids)).count() if class_ids else 0
    today_attendance = Attendance.query.filter_by(date=datetime.now().date()).filter(Attendance.class_id.in_(class_ids)).count() if class_ids else 0
    
    return render_template('college/dashboard.html',
                           total_classes=total_classes,
                           total_students=total_students,
                           today_attendance=today_attendance,
                           course=course,
                           year=year,
                           course_id=course_id)




@app.route('/college/add-year', methods=['GET', 'POST'])
@org_required(['college'])
def college_add_year():
    course_id = request.args.get('course_id')
    if request.method == 'POST':
        year_name = request.form.get('year_name')
        if not year_name:
            flash('Year name is required', 'danger')
            return redirect(url_for('college_add_year', course_id=course_id))
            
        existing = StudyYear.query.filter_by(name=year_name, course_id=int(course_id), organization_id=session.get('org_id')).first()
        if not existing:
            study_year = StudyYear(name=year_name, course_id=int(course_id), organization_id=session.get('org_id'))
            db.session.add(study_year)
            db.session.commit()
            
        return redirect(url_for('college_dashboard', course_id=course_id, year=year_name))
    return render_template('college/add_year.html', course_id=course_id)
@app.route('/college/add-course', methods=['GET', 'POST'])
@org_required(['college'])
def college_add_course():
    if request.method == 'POST':
        name = request.form.get('course_name')
        if not name:
            flash('Course name is required', 'danger')
            return redirect(url_for('college_add_course'))
        course = Course(name=name, organization_id=session.get('org_id'))
        db.session.add(course)
        db.session.commit()
        flash('Course added successfully!', 'success')
        return redirect(url_for('college_dashboard'))
    return render_template('college/add_course.html')

@app.route('/college/add-subject', methods=['GET', 'POST'])
@org_required(['college'])
def college_add_subject():
    course_id = request.args.get('course_id')
    year = request.args.get('year')
    if request.method == 'POST':
        name = request.form.get('subject_name')
        if not name or not course_id:
            flash('Subject name and course are required', 'danger')
            return redirect(url_for('college_add_subject', course_id=course_id, year=year))
        subject = Subject(name=name, course_id=int(course_id), study_year=year, organization_id=session.get('org_id'))
        db.session.add(subject)
        db.session.commit()
        flash('Subject added successfully!', 'success')
        return redirect(url_for('college_dashboard', course_id=course_id, year=year))
    return render_template('college/add_subject.html', course_id=course_id, year=year)
@app.route('/college/classes')
@org_required(['college'])
def college_classes():
    course_id = request.args.get('course_id')
    year = request.args.get('year')
    query = Class_.query.filter_by(organization_id=session.get('org_id'))
    if course_id:
        query = query.filter_by(course_id=course_id)
    if year:
        query = query.filter_by(study_year=year)
    classes = query.all()
    for c in classes:
        c.student_count = len(c.students)
    return render_template('college/classes.html', classes=classes, course_id=course_id, year=year)

@app.route('/college/add-class', methods=['GET', 'POST'])
@org_required(['college'])
def college_add_class():
    course_id = request.args.get('course_id')
    year = request.args.get('year')
    if course_id == 'None': course_id = None
    if year == 'None': year = None
    courses = Course.query.filter_by(organization_id=session.get('org_id')).all()
    
    if request.method == 'POST':
        class_name = request.form.get('class_name')
        course_id_post = request.form.get('course_id')
        study_year = request.form.get('study_year')
        
        if not all([class_name, study_year, course_id_post]):
            flash('All fields are required', 'danger')
            return redirect(url_for('college_add_class', course_id=course_id, year=year))
            
        class_ = Class_(name=class_name, academic_year='2023-2024', study_year=study_year,
                        organization_id=session.get('org_id'), course_id=int(course_id_post))
        db.session.add(class_)
        db.session.commit()
        flash('Class added successfully!', 'success')
        return redirect(url_for('college_dashboard', course_id=course_id_post, year=study_year))
        
    return render_template('college/add_class.html', courses=courses, selected_course=course_id, selected_year=year)

@app.route('/college/students')
@org_required(['college'])
def college_students():
    class_id = request.args.get('class_id')
    course_id = request.args.get('course_id')
    year = request.args.get('year')
    
    query = Student.query.join(Class_).filter(Student.organization_id == session.get('org_id'))
    if class_id:
        query = query.filter(Student.class_id == class_id)
    if course_id:
        query = query.filter(Class_.course_id == course_id)
    if year:
        query = query.filter(Class_.study_year == year)
        
    students = query.all()
    return render_template('college/students.html', students=students, course_id=course_id, year=year)

@app.route('/college/add-student', methods=['GET', 'POST'])
@org_required(['college'])
def college_add_student():
    course_id = request.args.get('course_id')
    year = request.args.get('year')
    
    query = Class_.query.filter_by(organization_id=session.get('org_id'))
    if course_id: query = query.filter_by(course_id=course_id)
    if year: query = query.filter_by(study_year=year)
    classes = query.all()
    if request.method == 'POST':
        name = request.form.get('name')
        roll_number = request.form.get('roll_number')
        phone = request.form.get('phone')
        class_id = request.form.get('class_id')
        if not all([name, roll_number, class_id]):
            flash('Name, roll number and class are required', 'danger')
            return redirect(url_for('college_add_student'))
        student = Student(name=name, roll_number=roll_number, phone=phone,
                          organization_id=session.get('org_id'), class_id=int(class_id))
        db.session.add(student)
        db.session.commit()
        flash('Student added successfully!', 'success')
        return redirect(url_for('college_dashboard'))
    return render_template('college/add_student.html', classes=classes)

@app.route('/college/face-register', methods=['GET', 'POST'])
@org_required(['college'])
def college_face_register():
    course_id = request.args.get('course_id')
    year = request.args.get('year')
    
    cls_query = Class_.query.filter_by(organization_id=session.get('org_id'))
    if course_id: cls_query = cls_query.filter_by(course_id=course_id)
    if year: cls_query = cls_query.filter_by(study_year=year)
    classes = cls_query.all()
    class_ids = [c.id for c in classes]
    
    if class_ids:
        students = Student.query.filter(Student.class_id.in_(class_ids)).all()
    else:
        students = []
    if request.method == 'POST':
        student_id = request.form.get('student_id', '')
        if not student_id:
            return jsonify({'error': 'Student not selected'}), 400
        if 'face_image' not in request.files:
            return jsonify({'error': 'No face image provided'}), 400
        file = request.files['face_image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Load image using face_recognition
            image = face_recognition.load_image_file(filepath)
            
            # Find face encodings
            face_encodings = face_recognition.face_encodings(image)

            if len(face_encodings) == 0:
                return jsonify({'error': 'No face detected in image'}), 400

            # Use the first detected face encoding
            face_encoding = face_encodings[0]
            
            face_record = FaceEncoding(student_id=int(student_id),
                                       encoding_path="",
                                       encoding_data=face_encoding.tolist(),
                                       created_at=datetime.utcnow())
            db.session.add(face_record)
            db.session.commit()
            return jsonify({'success': 'Face registered successfully'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return render_template('college/face_register.html', students=students, classes=classes)

@app.route('/college/mark-attendance', methods=['GET', 'POST'])
@org_required(['college'])
def college_mark_attendance():
    course_id = request.args.get('course_id')
    year = request.args.get('year')
    
    query = Class_.query.filter_by(organization_id=session.get('org_id'))
    if course_id: query = query.filter_by(course_id=course_id)
    if year: query = query.filter_by(study_year=year)
    classes = query.all()
    
    subj_query = Subject.query.filter_by(organization_id=session.get('org_id'))
    if course_id: subj_query = subj_query.filter_by(course_id=course_id)
    if year: subj_query = subj_query.filter_by(study_year=year)
    subjects = subj_query.all()
    if request.method == 'POST':
        class_id = request.form.get('class_id', '')
        subject_id = request.form.get('subject_id', '')
        if not class_id:
            return jsonify({'error': 'Class not selected'}), 400
        if not subject_id:
            return jsonify({'error': 'Subject not selected'}), 400
        if 'attendance_image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        file = request.files['attendance_image']
        try:
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_attendance.jpg')
            file.save(temp_path)
            # Load image using face_recognition
            image = face_recognition.load_image_file(temp_path)
            
            # Find all face locations and encodings in the image
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            if not face_encodings:
                return jsonify({'error': 'No faces detected'}), 400
                
            students = Student.query.filter_by(class_id=int(class_id)).all()
            
            # Load all stored encodings for this class
            known_encodings = []
            known_students = []
            
            for student in students:
                face_recs = FaceEncoding.query.filter_by(student_id=student.id).all()
                for face_rec in face_recs:
                    try:
                        if face_rec.encoding_data:
                            encoding = np.array(face_rec.encoding_data, dtype=np.float64)
                        else:
                            with open(face_rec.encoding_path, 'rb') as f:
                                encoding = pickle.load(f)
                        # Verify this is a 128-d encoding from face_recognition
                        if isinstance(encoding, np.ndarray) and encoding.shape == (128,):
                            known_encodings.append(encoding)
                            known_students.append(student)
                    except Exception:
                        continue

            if not known_encodings:
                return jsonify({'error': 'No registered faces found for this class'}), 400

            recognized_students = []
            
            for face_encoding in face_encodings:
                # Compare detected face with all known faces
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                
                if True in matches:
                    # Use the smallest distance to find the best match
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        best_match_student = known_students[best_match_index]
                        
                        already_recognized = any(s['roll_number'] == best_match_student.roll_number for s in recognized_students)
                        if not already_recognized:
                            existing = Attendance.query.filter_by(
                                student_id=best_match_student.id, date=datetime.now().date(), subject_id=int(subject_id)).first()
                            if not existing:
                                attendance = Attendance(
                                    student_id=best_match_student.id, class_id=int(class_id), subject_id=int(subject_id),
                                    date=datetime.now().date(), time=datetime.now().time(),
                                    status='present')
                                db.session.add(attendance)
                                status_str = attendance.status
                            else:
                                status_str = 'already present' if existing.status == 'present' else 'absent'
                            recognized_students.append({
                                'name': best_match_student.name,
                                'roll_number': best_match_student.roll_number,
                                'status': status_str
                            })
            db.session.commit()
            return jsonify({'success': f'Attendance marked for {len(recognized_students)} students',
                            'recognized': recognized_students})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('college/mark_attendance.html', classes=classes, subjects=subjects, course_id=course_id, year=year)

@app.route('/college/attendance-records')
@org_required(['college'])
def college_attendance_records():
    course_id = request.args.get('course_id')
    year = request.args.get('year')
    
    cls_query = Class_.query.filter_by(organization_id=session.get('org_id'))
    if course_id: cls_query = cls_query.filter_by(course_id=course_id)
    if year: cls_query = cls_query.filter_by(study_year=year)
    classes = cls_query.all()
    class_id = request.args.get('class_id')
    date_str = request.args.get('date')
    month_str = request.args.get('month')

    query = Attendance.query.join(Class_).filter(Class_.organization_id == int(session.get('org_id')))
    if class_id:
        query = query.filter(Attendance.class_id == class_id)
    date_obj = None
    if date_str:
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
            query = query.filter(Attendance.date == date_obj)
        except ValueError:
            pass
    if month_str:
        try:
            from calendar import monthrange
            from datetime import date
            y_val, m_val = map(int, month_str.split('-'))
            _, last_day = monthrange(y_val, m_val)
            query = query.filter(Attendance.date.between(date(y_val, m_val, 1), date(y_val, m_val, last_day)))
        except Exception:
            pass

    # If class_id and date_str are both provided, show ALL students of that class (including virtual absent ones)
    if class_id and date_obj:
        students = Student.query.filter_by(class_id=int(class_id)).all()
        
        # Find all unique subjects for which attendance was taken for this class on this date
        subjects_taken = db.session.query(Attendance.subject_id).filter_by(class_id=int(class_id), date=date_obj).distinct().all()
        subjects_taken = [s[0] for s in subjects_taken]
        
        if not subjects_taken:
            subjects_taken = [None]
            
        records = []
        for subj_id in subjects_taken:
            present_records = Attendance.query.filter_by(class_id=int(class_id), date=date_obj, subject_id=subj_id).all()
            present_student_ids = {r.student_id: r for r in present_records}
            
            for s in students:
                if s.id in present_student_ids:
                    records.append(present_student_ids[s.id])
                else:
                    from types import SimpleNamespace
                    subj = Subject.query.get(subj_id) if subj_id else None
                    absent_rec = SimpleNamespace(
                        id=None,
                        date=date_obj,
                        day=date_obj.strftime('%A'),
                        time=None,
                        status='absent',
                        student=s,
                        class_=s.class_,
                        subject=subj
                    )
                    records.append(absent_rec)
    else:
        records = query.order_by(Attendance.date.desc(), Attendance.time.desc()).all()
        
    return render_template('college/attendance_records.html', records=records, classes=classes, selected_class=class_id, selected_date=date_str, selected_month=month_str)

@app.route('/college/reports', methods=['GET', 'POST'])
@org_required(['college'])
def college_reports():
    if request.method == 'POST':
        report_type = request.form.get('report_type', 'student')
        student_id = request.form.get('student_id')
        class_id = request.form.get('class_id')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'],
                                     fontSize=16, spaceAfter=20)
        normal_style = styles['Normal']
        
        present_style = ParagraphStyle(
            'PresentStyle',
            parent=normal_style,
            textColor=colors.HexColor('#2e7d32'),
            fontName='Helvetica-Bold',
            alignment=1
        )
        absent_style = ParagraphStyle(
            'AbsentStyle',
            parent=normal_style,
            textColor=colors.HexColor('#c62828'),
            fontName='Helvetica-Bold',
            alignment=1
        )
        
        if report_type == 'class':
            elements.append(Paragraph("Class Attendance Report", title_style))
            if class_id:
                class_ = Class_.query.get(int(class_id))
                elements.append(Paragraph(f"Class: {class_.name}", normal_style))
                if start_date and end_date:
                    elements.append(Paragraph(f"Period: {start_date} to {end_date}", normal_style))
                elements.append(Spacer(1, 0.2*inch))
                
                # Fetch class students
                students = Student.query.filter_by(class_id=int(class_id)).order_by(Student.roll_number, Student.name).all()
                
                # Fetch present records
                att_query = Attendance.query.filter_by(class_id=int(class_id))
                if start_date and end_date:
                    att_query = att_query.filter(Attendance.date.between(start_date, end_date))
                present_attendances = att_query.all()
                
                # Get unique sessions (date, subject_id)
                sessions = sorted(list(set((att.date, att.subject_id) for att in present_attendances)), key=lambda x: (x[0], x[1] or 0), reverse=True)
                
                present_table_data = [['S.No', 'Date', 'Roll No', 'Student Name', 'Subject', 'Status']]
                absent_table_data = [['S.No', 'Date', 'Roll No', 'Student Name', 'Subject', 'Status']]
                
                pres_sno = 1
                abs_sno = 1
                
                for session_date, subj_id in sessions:
                    session_present = [att for att in present_attendances if att.date == session_date and att.subject_id == subj_id]
                    present_map = {att.student_id: att for att in session_present}
                    
                    subj_name = '-'
                    if subj_id:
                        subj_obj = Subject.query.get(subj_id)
                        if subj_obj:
                            subj_name = subj_obj.name
                            
                    for s in students:
                        if s.id in present_map:
                            present_table_data.append([
                                str(pres_sno),
                                str(session_date),
                                s.roll_number,
                                s.name,
                                subj_name,
                                Paragraph('Present', present_style)
                            ])
                            pres_sno += 1
                        else:
                            absent_table_data.append([
                                str(abs_sno),
                                str(session_date),
                                s.roll_number,
                                s.name,
                                subj_name,
                                Paragraph('Absent', absent_style)
                            ])
                            abs_sno += 1
                            
                has_records = False
                
                # Add Absent Students List
                if len(absent_table_data) > 1:
                    elements.append(Paragraph("Absent Students List (S.No wise)", ParagraphStyle('SubTitleAbs', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor('#c62828'), spaceAfter=10)))
                    abs_table = Table(absent_table_data)
                    abs_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#c62828')),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0,0), (-1,0), 10),
                        ('BOTTOMPADDING', (0,0), (-1,0), 8),
                        ('TOPPADDING', (0,0), (-1,0), 8),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#bdc3c7')),
                        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#fff5f5')])
                    ]))
                    elements.append(abs_table)
                    has_records = True
                    
                # Add Present Students List
                if len(present_table_data) > 1:
                    if has_records:
                        elements.append(Spacer(1, 0.3*inch))
                    elements.append(Paragraph("Present Students List (S.No wise)", ParagraphStyle('SubTitlePres', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor('#2e7d32'), spaceAfter=10)))
                    pres_table = Table(present_table_data)
                    pres_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2e7d32')),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0,0), (-1,0), 10),
                        ('BOTTOMPADDING', (0,0), (-1,0), 8),
                        ('TOPPADDING', (0,0), (-1,0), 8),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#bdc3c7')),
                        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5fff5')])
                    ]))
                    elements.append(pres_table)
                    has_records = True
                    
                if not has_records:
                    elements.append(Paragraph("No attendance records found.", normal_style))
        else:
            elements.append(Paragraph("Student Attendance Report", title_style))
            if student_id:
                student = Student.query.get(int(student_id))
                elements.append(Paragraph(f"Student: {student.name}", normal_style))
                elements.append(Paragraph(f"Roll Number: {student.roll_number}", normal_style))
                if start_date and end_date:
                    elements.append(Paragraph(f"Period: {start_date} to {end_date}", normal_style))
                elements.append(Spacer(1, 0.2*inch))
                
                # Header row for main table
                data = [['Date', 'Time', 'Subject', 'Status']]
                
                # Get the class sessions for the student's class
                att_query = Attendance.query.filter_by(class_id=student.class_id)
                if start_date and end_date:
                    att_query = att_query.filter(Attendance.date.between(start_date, end_date))
                class_attendances = att_query.all()
                
                sessions = sorted(list(set((att.date, att.subject_id) for att in class_attendances)), key=lambda x: (x[0], x[1] or 0), reverse=True)
                student_present_map = {(att.date, att.subject_id): att for att in class_attendances if att.student_id == student.id}
                
                for session_date, subj_id in sessions:
                    subj_name = '-'
                    if subj_id:
                        subj_obj = Subject.query.get(subj_id)
                        if subj_obj:
                            subj_name = subj_obj.name
                            
                    if (session_date, subj_id) in student_present_map:
                        att = student_present_map[(session_date, subj_id)]
                        data.append([
                            str(session_date),
                            att.time.strftime('%H:%M:%S') if att.time else '-',
                            subj_name,
                            Paragraph('Present', present_style)
                        ])
                    else:
                        data.append([
                            str(session_date),
                            '-',
                            subj_name,
                            Paragraph('Absent', absent_style)
                        ])

                if len(data) > 1:
                    table = Table(data)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a252f')),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0,0), (-1,0), 11),
                        ('BOTTOMPADDING', (0,0), (-1,0), 8),
                        ('TOPPADDING', (0,0), (-1,0), 8),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#bdc3c7')),
                        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')])
                    ]))
                    elements.append(table)
                else:
                    elements.append(Paragraph("No attendance records found.", normal_style))

        doc.build(elements)
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name='attendance_report.pdf',
                         mimetype='application/pdf')
    
    course_id = request.args.get('course_id')
    year = request.args.get('year')
    cls_query = Class_.query.filter_by(organization_id=session.get('org_id'))
    if course_id: cls_query = cls_query.filter_by(course_id=course_id)
    if year: cls_query = cls_query.filter_by(study_year=year)
    classes = cls_query.all()
    class_ids = [c.id for c in classes]
    if class_ids:
        students = Student.query.filter(Student.class_id.in_(class_ids)).all()
    else:
        students = []
    return render_template('college/reports.html', students=students, classes=classes)
    
    course_id = request.args.get('course_id')
    year = request.args.get('year')
    cls_query = Class_.query.filter_by(organization_id=session.get('org_id'))
    if course_id: cls_query = cls_query.filter_by(course_id=course_id)
    if year: cls_query = cls_query.filter_by(study_year=year)
    classes = cls_query.all()
    class_ids = [c.id for c in classes]
    if class_ids:
        students = Student.query.filter(Student.class_id.in_(class_ids)).all()
    else:
        students = []
    return render_template('college/reports.html', students=students, classes=classes)

# ─────────────────────────────────────────────
# INSTITUTION ROUTES
# ─────────────────────────────────────────────

@app.route('/institution/dashboard')
@org_required(['institution'])
def institution_dashboard():
    org_id = session.get('org_id')
    courses = Course.query.filter_by(organization_id=org_id).all()
    
    course_id = request.args.get('course_id')
    branch = request.args.get('branch')
    year = request.args.get('year')
    sem = request.args.get('sem')
    
    if not course_id or course_id == 'None':
        return render_template('institution/select_course.html', courses=courses)
        
    try:
        course = Course.query.get(int(course_id))
    except (ValueError, TypeError):
        flash("Invalid course selection.", "danger")
        return redirect(url_for('institution_dashboard'))
        
    if not course:
        flash("Course not found.", "danger")
        return redirect(url_for('institution_dashboard'))
    
    if not branch or branch == 'None':
        branches = Branch.query.filter_by(organization_id=org_id, course_id=course_id).all()
        return render_template('institution/select_branch.html', course=course, branches=branches)
        
    if not year or year == 'None':
        years = StudyYear.query.filter_by(organization_id=org_id, course_id=course_id).all()
        return render_template('institution/select_year.html', course=course, branch=branch, years=years)
        
    if not sem or sem == 'None':
        sems = Semester.query.filter_by(organization_id=org_id, course_id=course_id).all()
        return render_template('institution/select_sem.html', course=course, branch=branch, year=year, sems=sems)
        
    classes = Class_.query.filter_by(organization_id=org_id, course_id=course_id, branch=branch, study_year=year, semester=sem).all()
    total_classes = len(classes)
    class_ids = [c.id for c in classes]
    total_students = Student.query.filter(Student.class_id.in_(class_ids)).count() if class_ids else 0
    today_attendance = Attendance.query.filter_by(date=datetime.now().date()).filter(Attendance.class_id.in_(class_ids)).count() if class_ids else 0
    
    return render_template('institution/dashboard.html',
                           total_classes=total_classes,
                           total_students=total_students,
                           today_attendance=today_attendance,
                           course=course,
                           branch=branch,
                           year=year,
                           sem=sem,
                           course_id=course_id)




@app.route('/institution/add-branch', methods=['GET', 'POST'])
@org_required(['institution'])
def institution_add_branch():
    course_id = request.args.get('course_id')
    if request.method == 'POST':
        branch_name = request.form.get('branch_name')
        if not branch_name:
            flash('Branch name is required', 'danger')
            return redirect(url_for('institution_add_branch', course_id=course_id))
        branch = Branch(name=branch_name, course_id=int(course_id), organization_id=session.get('org_id'))
        db.session.add(branch)
        db.session.commit()
        return redirect(url_for('institution_dashboard', course_id=course_id, branch=branch_name))
    return render_template('institution/add_branch.html', course_id=course_id)

@app.route('/institution/add-year', methods=['GET', 'POST'])
@org_required(['institution'])
def institution_add_year():
    course_id = request.args.get('course_id')
    branch = request.args.get('branch')
    if request.method == 'POST':
        year_name = request.form.get('year_name')
        if not year_name:
            flash('Year name is required', 'danger')
            return redirect(url_for('institution_add_year', course_id=course_id, branch=branch))
        year = StudyYear(name=year_name, course_id=int(course_id), organization_id=session.get('org_id'))
        db.session.add(year)
        db.session.commit()
        return redirect(url_for('institution_dashboard', course_id=course_id, branch=branch, year=year_name))
    return render_template('institution/add_year.html', course_id=course_id, branch=branch)

@app.route('/institution/add-sem', methods=['GET', 'POST'])
@org_required(['institution'])
def institution_add_sem():
    course_id = request.args.get('course_id')
    branch = request.args.get('branch')
    year = request.args.get('year')
    if request.method == 'POST':
        sem_name = request.form.get('sem_name')
        if not sem_name:
            flash('Semester name is required', 'danger')
            return redirect(url_for('institution_add_sem', course_id=course_id, branch=branch, year=year))
        sem = Semester(name=sem_name, course_id=int(course_id), organization_id=session.get('org_id'))
        db.session.add(sem)
        db.session.commit()
        return redirect(url_for('institution_dashboard', course_id=course_id, branch=branch, year=year, sem=sem_name))
    return render_template('institution/add_sem.html', course_id=course_id, branch=branch, year=year)
@app.route('/institution/add-course', methods=['GET', 'POST'])
@org_required(['institution'])
def institution_add_course():
    if request.method == 'POST':
        name = request.form.get('course_name')
        if not name:
            flash('Course name is required', 'danger')
            return redirect(url_for('institution_add_course'))
        course = Course(name=name, organization_id=session.get('org_id'))
        db.session.add(course)
        db.session.commit()
        flash('Course added successfully!', 'success')
        return redirect(url_for('institution_dashboard'))
    return render_template('institution/add_course.html')

@app.route('/institution/add-subject', methods=['GET', 'POST'])
@org_required(['institution'])
def institution_add_subject():
    course_id = request.args.get('course_id')
    branch = request.args.get('branch')
    year = request.args.get('year')
    sem = request.args.get('sem')
    if request.method == 'POST':
        name = request.form.get('subject_name')
        if not name or not course_id:
            flash('Subject name and course are required', 'danger')
            return redirect(url_for('institution_add_subject', course_id=course_id, branch=branch, year=year, sem=sem))
        subject = Subject(name=name, course_id=int(course_id), study_year=year, organization_id=session.get('org_id'))
        db.session.add(subject)
        db.session.commit()
        flash('Subject added successfully!', 'success')
        return redirect(url_for('institution_dashboard', course_id=course_id, branch=branch, year=year, sem=sem))
    return render_template('institution/add_subject.html', course_id=course_id, branch=branch, year=year, sem=sem)
@app.route('/institution/classes')
@org_required(['institution'])
def institution_classes():
    course_id = request.args.get('course_id')
    branch = request.args.get('branch')
    year = request.args.get('year')
    sem = request.args.get('sem')
    query = Class_.query.filter_by(organization_id=session.get('org_id'))
    if course_id:
        query = query.filter_by(course_id=course_id)
    if branch:
        query = query.filter_by(branch=branch)
    if year:
        query = query.filter_by(study_year=year)
    if sem:
        query = query.filter_by(semester=sem)
    classes = query.all()
    for c in classes:
        c.student_count = len(c.students)
    return render_template('institution/classes.html', classes=classes, course_id=course_id, branch=branch, year=year, sem=sem)

@app.route('/institution/add-class', methods=['GET', 'POST'])
@org_required(['institution'])
def institution_add_class():
    course_id = request.args.get('course_id')
    branch = request.args.get('branch')
    year = request.args.get('year')
    sem = request.args.get('sem')
    if course_id == 'None': course_id = None
    if branch == 'None': branch = None
    if year == 'None': year = None
    if sem == 'None': sem = None
    courses = Course.query.filter_by(organization_id=session.get('org_id')).all()
    
    if request.method == 'POST':
        class_name = request.form.get('class_name')
        course_id_post = request.form.get('course_id')
        branch_post = request.form.get('branch')
        study_year = request.form.get('study_year')
        sem_post = request.form.get('sem')
        
        if not all([class_name, study_year, course_id_post]):
            flash('All fields are required', 'danger')
            return redirect(url_for('institution_add_class', course_id=course_id, branch=branch, year=year, sem=sem))
            
        class_ = Class_(name=class_name, academic_year='2023-2024', study_year=study_year, branch=branch_post, semester=sem_post,
                        organization_id=session.get('org_id'), course_id=int(course_id_post))
        db.session.add(class_)
        db.session.commit()
        flash('Class added successfully!', 'success')
        return redirect(url_for('institution_dashboard', course_id=course_id_post, branch=branch_post, year=study_year, sem=sem_post))
        
    return render_template('institution/add_class.html', courses=courses, selected_course=course_id, selected_branch=branch, selected_year=year, selected_sem=sem)

@app.route('/institution/students')
@org_required(['institution'])
def institution_students():
    class_id = request.args.get('class_id')
    course_id = request.args.get('course_id')
    branch = request.args.get('branch')
    year = request.args.get('year')
    sem = request.args.get('sem')
    
    query = Student.query.join(Class_).filter(Student.organization_id == session.get('org_id'))
    if class_id:
        query = query.filter(Student.class_id == class_id)
    if course_id:
        query = query.filter(Class_.course_id == course_id)
    if branch:
        query = query.filter(Class_.branch == branch)
    if year:
        query = query.filter(Class_.study_year == year)
    if sem:
        query = query.filter(Class_.semester == sem)
        
    students = query.all()
    return render_template('institution/students.html', students=students, course_id=course_id, branch=branch, year=year, sem=sem)

@app.route('/institution/add-student', methods=['GET', 'POST'])
@org_required(['institution'])
def institution_add_student():
    course_id = request.args.get('course_id')
    branch = request.args.get('branch')
    year = request.args.get('year')
    sem = request.args.get('sem')
    
    query = Class_.query.filter_by(organization_id=session.get('org_id'))
    if course_id: query = query.filter_by(course_id=course_id)
    if branch: query = query.filter_by(branch=branch)
    if year: query = query.filter_by(study_year=year)
    if sem: query = query.filter_by(semester=sem)
    classes = query.all()
    if request.method == 'POST':
        name = request.form.get('name')
        roll_number = request.form.get('roll_number')
        phone = request.form.get('phone')
        class_id = request.form.get('class_id')
        if not all([name, roll_number, class_id]):
            flash('Name, roll number and class are required', 'danger')
            return redirect(url_for('institution_add_student'))
        student = Student(name=name, roll_number=roll_number, phone=phone,
                          organization_id=session.get('org_id'), class_id=int(class_id))
        db.session.add(student)
        db.session.commit()
        flash('Student added successfully!', 'success')
        return redirect(url_for('institution_dashboard'))
    return render_template('institution/add_student.html', classes=classes)

@app.route('/institution/face-register', methods=['GET', 'POST'])
@org_required(['institution'])
def institution_face_register():
    course_id = request.args.get('course_id')
    branch = request.args.get('branch')
    year = request.args.get('year')
    sem = request.args.get('sem')
    
    cls_query = Class_.query.filter_by(organization_id=session.get('org_id'))
    if course_id: cls_query = cls_query.filter_by(course_id=course_id)
    if branch: cls_query = cls_query.filter_by(branch=branch)
    if year: cls_query = cls_query.filter_by(study_year=year)
    if sem: cls_query = cls_query.filter_by(semester=sem)
    classes = cls_query.all()
    class_ids = [c.id for c in classes]
    
    if class_ids:
        students = Student.query.filter(Student.class_id.in_(class_ids)).all()
    else:
        students = []
    if request.method == 'POST':
        student_id = request.form.get('student_id', '')
        if not student_id:
            return jsonify({'error': 'Student not selected'}), 400
        if 'face_image' not in request.files:
            return jsonify({'error': 'No face image provided'}), 400
        file = request.files['face_image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Load image using face_recognition
            image = face_recognition.load_image_file(filepath)
            
            # Find face encodings
            face_encodings = face_recognition.face_encodings(image)

            if len(face_encodings) == 0:
                return jsonify({'error': 'No face detected in image'}), 400

            # Use the first detected face encoding
            face_encoding = face_encodings[0]
            
            face_record = FaceEncoding(student_id=int(student_id),
                                       encoding_path="",
                                       encoding_data=face_encoding.tolist(),
                                       created_at=datetime.utcnow())
            db.session.add(face_record)
            db.session.commit()
            return jsonify({'success': 'Face registered successfully'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return render_template('institution/face_register.html', students=students, classes=classes)

@app.route('/institution/mark-attendance', methods=['GET', 'POST'])
@org_required(['institution'])
def institution_mark_attendance():
    course_id = request.args.get('course_id')
    year = request.args.get('year')
    
    query = Class_.query.filter_by(organization_id=session.get('org_id'))
    if course_id: query = query.filter_by(course_id=course_id)
    if year: query = query.filter_by(study_year=year)
    classes = query.all()
    
    subj_query = Subject.query.filter_by(organization_id=session.get('org_id'))
    if course_id: subj_query = subj_query.filter_by(course_id=course_id)
    if year: subj_query = subj_query.filter_by(study_year=year)
    subjects = subj_query.all()
    if request.method == 'POST':
        class_id = request.form.get('class_id', '')
        subject_id = request.form.get('subject_id', '')
        if not class_id:
            return jsonify({'error': 'Class not selected'}), 400
        if not subject_id:
            return jsonify({'error': 'Subject not selected'}), 400
        if 'attendance_image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        file = request.files['attendance_image']
        try:
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_attendance.jpg')
            file.save(temp_path)
            # Load image using face_recognition
            image = face_recognition.load_image_file(temp_path)
            
            # Find all face locations and encodings in the image
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            if not face_encodings:
                return jsonify({'error': 'No faces detected'}), 400
                
            students = Student.query.filter_by(class_id=int(class_id)).all()
            
            # Load all stored encodings for this class
            known_encodings = []
            known_students = []
            
            for student in students:
                face_recs = FaceEncoding.query.filter_by(student_id=student.id).all()
                for face_rec in face_recs:
                    try:
                        if face_rec.encoding_data:
                            encoding = np.array(face_rec.encoding_data, dtype=np.float64)
                        else:
                            with open(face_rec.encoding_path, 'rb') as f:
                                encoding = pickle.load(f)
                        # Verify this is a 128-d encoding from face_recognition
                        if isinstance(encoding, np.ndarray) and encoding.shape == (128,):
                            known_encodings.append(encoding)
                            known_students.append(student)
                    except Exception:
                        continue

            if not known_encodings:
                return jsonify({'error': 'No registered faces found for this class'}), 400

            recognized_students = []
            
            for face_encoding in face_encodings:
                # Compare detected face with all known faces
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                
                if True in matches:
                    # Use the smallest distance to find the best match
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        best_match_student = known_students[best_match_index]
                        
                        already_recognized = any(s['roll_number'] == best_match_student.roll_number for s in recognized_students)
                        if not already_recognized:
                            existing = Attendance.query.filter_by(
                                student_id=best_match_student.id, date=datetime.now().date(), subject_id=int(subject_id)).first()
                            if not existing:
                                attendance = Attendance(
                                    student_id=best_match_student.id, class_id=int(class_id), subject_id=int(subject_id),
                                    date=datetime.now().date(), time=datetime.now().time(),
                                    status='present')
                                db.session.add(attendance)
                                status_str = attendance.status
                            else:
                                status_str = 'already present' if existing.status == 'present' else 'absent'
                            recognized_students.append({
                                'name': best_match_student.name,
                                'roll_number': best_match_student.roll_number,
                                'status': status_str
                            })
            db.session.commit()
            return jsonify({'success': f'Attendance marked for {len(recognized_students)} students',
                            'recognized': recognized_students})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('institution/mark_attendance.html', classes=classes, subjects=subjects, course_id=course_id, year=year)

@app.route('/institution/attendance-records')
@org_required(['institution'])
def institution_attendance_records():
    course_id = request.args.get('course_id')
    year = request.args.get('year')
    
    cls_query = Class_.query.filter_by(organization_id=session.get('org_id'))
    if course_id: cls_query = cls_query.filter_by(course_id=course_id)
    if year: cls_query = cls_query.filter_by(study_year=year)
    classes = cls_query.all()
    class_id = request.args.get('class_id')
    date_str = request.args.get('date')
    month_str = request.args.get('month')

    query = Attendance.query.join(Class_).filter(Class_.organization_id == int(session.get('org_id')))
    if class_id:
        query = query.filter(Attendance.class_id == class_id)
    date_obj = None
    if date_str:
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
            query = query.filter(Attendance.date == date_obj)
        except ValueError:
            pass
    if month_str:
        try:
            from calendar import monthrange
            from datetime import date
            y_val, m_val = map(int, month_str.split('-'))
            _, last_day = monthrange(y_val, m_val)
            query = query.filter(Attendance.date.between(date(y_val, m_val, 1), date(y_val, m_val, last_day)))
        except Exception:
            pass

    # If class_id and date_str are both provided, show ALL students of that class (including virtual absent ones)
    if class_id and date_obj:
        students = Student.query.filter_by(class_id=int(class_id)).all()
        
        # Find all unique subjects for which attendance was taken for this class on this date
        subjects_taken = db.session.query(Attendance.subject_id).filter_by(class_id=int(class_id), date=date_obj).distinct().all()
        subjects_taken = [s[0] for s in subjects_taken]
        
        if not subjects_taken:
            subjects_taken = [None]
            
        records = []
        for subj_id in subjects_taken:
            present_records = Attendance.query.filter_by(class_id=int(class_id), date=date_obj, subject_id=subj_id).all()
            present_student_ids = {r.student_id: r for r in present_records}
            
            for s in students:
                if s.id in present_student_ids:
                    records.append(present_student_ids[s.id])
                else:
                    from types import SimpleNamespace
                    subj = Subject.query.get(subj_id) if subj_id else None
                    absent_rec = SimpleNamespace(
                        id=None,
                        date=date_obj,
                        day=date_obj.strftime('%A'),
                        time=None,
                        status='absent',
                        student=s,
                        class_=s.class_,
                        subject=subj
                    )
                    records.append(absent_rec)
    else:
        records = query.order_by(Attendance.date.desc(), Attendance.time.desc()).all()
        
    return render_template('institution/attendance_records.html', records=records, classes=classes, selected_class=class_id, selected_date=date_str, selected_month=month_str)

@app.route('/institution/reports', methods=['GET', 'POST'])
@org_required(['institution'])
def institution_reports():
    if request.method == 'POST':
        report_type = request.form.get('report_type', 'student')
        student_id = request.form.get('student_id')
        class_id = request.form.get('class_id')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'],
                                     fontSize=16, spaceAfter=20)
        normal_style = styles['Normal']
        
        present_style = ParagraphStyle(
            'PresentStyle',
            parent=normal_style,
            textColor=colors.HexColor('#2e7d32'),
            fontName='Helvetica-Bold',
            alignment=1
        )
        absent_style = ParagraphStyle(
            'AbsentStyle',
            parent=normal_style,
            textColor=colors.HexColor('#c62828'),
            fontName='Helvetica-Bold',
            alignment=1
        )
        
        if report_type == 'class':
            elements.append(Paragraph("Class Attendance Report", title_style))
            if class_id:
                class_ = Class_.query.get(int(class_id))
                elements.append(Paragraph(f"Class: {class_.name}", normal_style))
                if start_date and end_date:
                    elements.append(Paragraph(f"Period: {start_date} to {end_date}", normal_style))
                elements.append(Spacer(1, 0.2*inch))
                
                # Fetch class students
                students = Student.query.filter_by(class_id=int(class_id)).order_by(Student.roll_number, Student.name).all()
                
                # Fetch present records
                att_query = Attendance.query.filter_by(class_id=int(class_id))
                if start_date and end_date:
                    att_query = att_query.filter(Attendance.date.between(start_date, end_date))
                present_attendances = att_query.all()
                
                # Get unique sessions (date, subject_id)
                sessions = sorted(list(set((att.date, att.subject_id) for att in present_attendances)), key=lambda x: (x[0], x[1] or 0), reverse=True)
                
                present_table_data = [['S.No', 'Date', 'Roll No', 'Student Name', 'Subject', 'Status']]
                absent_table_data = [['S.No', 'Date', 'Roll No', 'Student Name', 'Subject', 'Status']]
                
                pres_sno = 1
                abs_sno = 1
                
                for session_date, subj_id in sessions:
                    session_present = [att for att in present_attendances if att.date == session_date and att.subject_id == subj_id]
                    present_map = {att.student_id: att for att in session_present}
                    
                    subj_name = '-'
                    if subj_id:
                        subj_obj = Subject.query.get(subj_id)
                        if subj_obj:
                            subj_name = subj_obj.name
                            
                    for s in students:
                        if s.id in present_map:
                            present_table_data.append([
                                str(pres_sno),
                                str(session_date),
                                s.roll_number,
                                s.name,
                                subj_name,
                                Paragraph('Present', present_style)
                            ])
                            pres_sno += 1
                        else:
                            absent_table_data.append([
                                str(abs_sno),
                                str(session_date),
                                s.roll_number,
                                s.name,
                                subj_name,
                                Paragraph('Absent', absent_style)
                            ])
                            abs_sno += 1
                            
                has_records = False
                
                # Add Absent Students List
                if len(absent_table_data) > 1:
                    elements.append(Paragraph("Absent Students List (S.No wise)", ParagraphStyle('SubTitleAbs', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor('#c62828'), spaceAfter=10)))
                    abs_table = Table(absent_table_data)
                    abs_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#c62828')),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0,0), (-1,0), 10),
                        ('BOTTOMPADDING', (0,0), (-1,0), 8),
                        ('TOPPADDING', (0,0), (-1,0), 8),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#bdc3c7')),
                        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#fff5f5')])
                    ]))
                    elements.append(abs_table)
                    has_records = True
                    
                # Add Present Students List
                if len(present_table_data) > 1:
                    if has_records:
                        elements.append(Spacer(1, 0.3*inch))
                    elements.append(Paragraph("Present Students List (S.No wise)", ParagraphStyle('SubTitlePres', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor('#2e7d32'), spaceAfter=10)))
                    pres_table = Table(present_table_data)
                    pres_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2e7d32')),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0,0), (-1,0), 10),
                        ('BOTTOMPADDING', (0,0), (-1,0), 8),
                        ('TOPPADDING', (0,0), (-1,0), 8),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#bdc3c7')),
                        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5fff5')])
                    ]))
                    elements.append(pres_table)
                    has_records = True
                    
                if not has_records:
                    elements.append(Paragraph("No attendance records found.", normal_style))
        else:
            elements.append(Paragraph("Student Attendance Report", title_style))
            if student_id:
                student = Student.query.get(int(student_id))
                elements.append(Paragraph(f"Student: {student.name}", normal_style))
                elements.append(Paragraph(f"Roll Number: {student.roll_number}", normal_style))
                if start_date and end_date:
                    elements.append(Paragraph(f"Period: {start_date} to {end_date}", normal_style))
                elements.append(Spacer(1, 0.2*inch))
                
                # Header row for main table
                data = [['Date', 'Time', 'Subject', 'Status']]
                
                # Get the class sessions for the student's class
                att_query = Attendance.query.filter_by(class_id=student.class_id)
                if start_date and end_date:
                    att_query = att_query.filter(Attendance.date.between(start_date, end_date))
                class_attendances = att_query.all()
                
                sessions = sorted(list(set((att.date, att.subject_id) for att in class_attendances)), key=lambda x: (x[0], x[1] or 0), reverse=True)
                student_present_map = {(att.date, att.subject_id): att for att in class_attendances if att.student_id == student.id}
                
                for session_date, subj_id in sessions:
                    subj_name = '-'
                    if subj_id:
                        subj_obj = Subject.query.get(subj_id)
                        if subj_obj:
                            subj_name = subj_obj.name
                            
                    if (session_date, subj_id) in student_present_map:
                        att = student_present_map[(session_date, subj_id)]
                        data.append([
                            str(session_date),
                            att.time.strftime('%H:%M:%S') if att.time else '-',
                            subj_name,
                            Paragraph('Present', present_style)
                        ])
                    else:
                        data.append([
                            str(session_date),
                            '-',
                            subj_name,
                            Paragraph('Absent', absent_style)
                        ])

                if len(data) > 1:
                    table = Table(data)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a252f')),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0,0), (-1,0), 11),
                        ('BOTTOMPADDING', (0,0), (-1,0), 8),
                        ('TOPPADDING', (0,0), (-1,0), 8),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#bdc3c7')),
                        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')])
                    ]))
                    elements.append(table)
                else:
                    elements.append(Paragraph("No attendance records found.", normal_style))

        doc.build(elements)
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name='attendance_report.pdf',
                         mimetype='application/pdf')
    
    course_id = request.args.get('course_id')
    year = request.args.get('year')
    cls_query = Class_.query.filter_by(organization_id=session.get('org_id'))
    if course_id: cls_query = cls_query.filter_by(course_id=course_id)
    if year: cls_query = cls_query.filter_by(study_year=year)
    classes = cls_query.all()
    class_ids = [c.id for c in classes]
    if class_ids:
        students = Student.query.filter(Student.class_id.in_(class_ids)).all()
    else:
        students = []
    return render_template('institution/reports.html', students=students, classes=classes)

# Student Portal
@app.route('/student/login', methods=['GET', 'POST'])
def student_login():
    if request.method == 'POST':
        phone = request.form.get('phone')
        password = request.form.get('password')

        student = Student.query.filter_by(phone=phone).first()
        if student:
            if student.password and check_password_hash(student.password, password):
                session['student_id'] = student.id
                return redirect(url_for('student_dashboard'))
            elif not student.password and password == 'student123':
                session['student_id'] = student.id
                return redirect(url_for('student_dashboard'))

        flash('Invalid credentials', 'danger')

    return render_template('student_login.html')

@app.route('/student/register', methods=['GET', 'POST'])
def student_register():
    if request.method == 'POST':
        org_name = request.form.get('org_name')
        admin_email = request.form.get('admin_email')
        name = request.form.get('name')
        roll_number = request.form.get('roll_number')
        phone = request.form.get('phone')
        password = request.form.get('password')
        class_id = request.form.get('class_id')
        
        if not all([org_name, admin_email, name, roll_number, phone, password, class_id]):
            flash('All fields are required', 'danger')
            return redirect(url_for('student_register'))
            
        # Verify the organization name
        org = Organization.query.filter_by(name=org_name).first()
        if not org:
            flash('Organization not found. Please check the name.', 'danger')
            return redirect(url_for('student_register'))

        # Verify the admin email
        admin = User.query.filter_by(email=admin_email, organization_id=org.id).first()
        if not admin:
            flash('Admin Email not found or does not match the Organization.', 'danger')
            return redirect(url_for('student_register'))

        class_ = Class_.query.get(class_id)
        if not class_ or class_.organization_id != org.id:
            flash('Invalid Class Selected for this Organization', 'danger')
            return redirect(url_for('student_register'))

        existing_student = Student.query.filter_by(phone=phone).first()
        if existing_student:
            flash('Phone number already registered. Please login.', 'danger')
            return redirect(url_for('student_login'))

        student = Student(
            name=name,
            roll_number=roll_number,
            phone=phone,
            password=generate_password_hash(password),
            organization_id=class_.organization_id,
            class_id=int(class_id)
        )
        db.session.add(student)
        db.session.commit()

        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('student_login'))

    classes = Class_.query.all()
    return render_template('student_register.html', classes=classes)

@app.route('/student/dashboard')
def student_dashboard():
    if 'student_id' not in session:
        return redirect(url_for('student_login'))

    student = Student.query.get(session['student_id'])
    class_id = student.class_id
    
    # Query distinct (date, subject_id) where attendance was marked for this class
    sessions = db.session.query(Attendance.date, Attendance.subject_id).filter_by(class_id=class_id).distinct().all()
    
    # Query the student's actual attendance records
    student_att = Attendance.query.filter_by(student_id=student.id).all()
    # Map them for quick lookup: (date, subject_id) -> record
    att_map = {}
    for att in student_att:
        att_map[(att.date, att.subject_id)] = att
        
    # Build complete attendance history
    history = []
    present_count = 0
    total_attendance = 0
    
    for date_val, subj_id in sessions:
        total_attendance += 1
        att_record = att_map.get((date_val, subj_id))
        if att_record:
            present_count += 1
            history.append(att_record)
        else:
            from types import SimpleNamespace
            subj = Subject.query.get(subj_id) if subj_id else None
            absent_rec = SimpleNamespace(
                date=date_val,
                day=date_val.strftime('%A'),
                time=None,
                status='absent',
                student=student,
                class_=student.class_,
                subject=subj
            )
            history.append(absent_rec)
            
    # Sort history by date descending
    history.sort(key=lambda x: x.date, reverse=True)
    
    absent_count = total_attendance - present_count
    attendance_pct = (present_count / total_attendance * 100) if total_attendance > 0 else 0
    
    # Calculate monthly statistics
    today = datetime.now().date()
    start_of_month = today.replace(day=1)
    
    monthly_total = 0
    monthly_present = 0
    for h in history:
        if h.date >= start_of_month:
            monthly_total += 1
            if h.status == 'present':
                monthly_present += 1
                
    recent_history = history[:30]

    return render_template('student_report.html',
                          student=student,
                          records=recent_history,
                          present_count=present_count,
                          absent_count=absent_count,
                          total_attendance=total_attendance,
                          attendance_pct=round(attendance_pct, 1),
                          monthly_present=monthly_present,
                          monthly_total=monthly_total)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('splash'))

@app.route('/init-db')
def init_db():
    """Initialize database with default data"""
    db.create_all()

    # Automatic migration to add encoding_data JSON column if not exists
    try:
        from sqlalchemy import text
        db.session.execute(text("ALTER TABLE face_encoding ADD COLUMN encoding_data JSON"))
        db.session.commit()
    except Exception as e:
        db.session.rollback()

    # Create default organizations if they don't exist
    defaults = [
        ('SMVIT School', 'school', 'smvit_school', 'school123'),
        ('SMVIT College', 'college', 'smvit_college', 'college123'),
        ('SMVIT Institution', 'institution', 'smvit_institution', 'institution123')
    ]

    for org_name, org_type, username, password in defaults:
        org = Organization.query.filter_by(name=org_name).first()
        if not org:
            org = Organization(name=org_name, type=org_type)
            db.session.add(org)
            db.session.commit()

            user = User(
                username=username,
                email=f'{username}@example.com',
                password_hash=generate_password_hash(password),
                org_type=org_type,
                organization_id=org.id
            )
            db.session.add(user)
            db.session.commit()

    return "Database initialized successfully!"

# ==========================================
# DELETE ROUTES - GENERAL & ORGANIZATION-SPECIFIC
# ==========================================

# --- SCHOOL DELETE ROUTES ---
@app.route('/school/delete-class/<int:class_id>')
@org_required(['school'])
def school_delete_class(class_id):
    org_id = session.get('org_id')
    class_ = Class_.query.filter_by(id=class_id, organization_id=org_id).first_or_404()
    
    # 1. Delete all student face encodings and attendances
    students = Student.query.filter_by(class_id=class_.id).all()
    for student in students:
        FaceEncoding.query.filter_by(student_id=student.id).delete()
        Attendance.query.filter_by(student_id=student.id).delete()
        db.session.delete(student)
        
    # 2. Delete all attendance records of the class
    Attendance.query.filter_by(class_id=class_.id).delete()
    
    db.session.delete(class_)
    db.session.commit()
    flash('Class and all its students/attendance records deleted successfully.', 'success')
    return redirect(url_for('school_classes'))

@app.route('/school/delete-student/<int:student_id>')
@org_required(['school'])
def school_delete_student(student_id):
    org_id = session.get('org_id')
    student = Student.query.filter_by(id=student_id, organization_id=org_id).first_or_404()
    
    # Delete face encodings and attendance
    FaceEncoding.query.filter_by(student_id=student.id).delete()
    Attendance.query.filter_by(student_id=student.id).delete()
    
    db.session.delete(student)
    db.session.commit()
    flash('Student and all their records deleted successfully.', 'success')
    return redirect(url_for('school_students'))


# --- COLLEGE DELETE ROUTES ---
@app.route('/college/delete-course/<int:course_id>')
@org_required(['college'])
def college_delete_course(course_id):
    org_id = session.get('org_id')
    course = Course.query.filter_by(id=course_id, organization_id=org_id).first_or_404()
    
    # 1. Delete all classes under this course (which cascades to students, attendances, face encodings)
    classes = Class_.query.filter_by(course_id=course.id).all()
    for class_ in classes:
        students = Student.query.filter_by(class_id=class_.id).all()
        for student in students:
            FaceEncoding.query.filter_by(student_id=student.id).delete()
            Attendance.query.filter_by(student_id=student.id).delete()
            db.session.delete(student)
        Attendance.query.filter_by(class_id=class_.id).delete()
        db.session.delete(class_)
        
    # 2. Delete subjects and their attendance
    subjects = Subject.query.filter_by(course_id=course.id).all()
    for subj in subjects:
        Attendance.query.filter_by(subject_id=subj.id).delete()
        db.session.delete(subj)
        
    # 3. Delete StudyYears
    StudyYear.query.filter_by(course_id=course.id).delete()
    
    db.session.delete(course)
    db.session.commit()
    flash('Course and all its classes, subjects, students, and attendance records deleted successfully.', 'success')
    return redirect(url_for('college_dashboard'))

@app.route('/college/delete-year/<int:year_id>')
@org_required(['college'])
def college_delete_year(year_id):
    org_id = session.get('org_id')
    study_year = StudyYear.query.filter_by(id=year_id, organization_id=org_id).first_or_404()
    course_id = study_year.course_id
    
    db.session.delete(study_year)
    db.session.commit()
    flash('Study Year deleted successfully.', 'success')
    return redirect(url_for('college_dashboard', course_id=course_id))

@app.route('/college/delete-subject/<int:subject_id>')
@org_required(['college'])
def college_delete_subject(subject_id):
    org_id = session.get('org_id')
    subject = Subject.query.filter_by(id=subject_id, organization_id=org_id).first_or_404()
    
    # Delete subject attendances
    Attendance.query.filter_by(subject_id=subject.id).delete()
    
    db.session.delete(subject)
    db.session.commit()
    flash('Subject and its attendance records deleted successfully.', 'success')
    return redirect(url_for('college_dashboard'))

@app.route('/college/delete-class/<int:class_id>')
@org_required(['college'])
def college_delete_class(class_id):
    org_id = session.get('org_id')
    class_ = Class_.query.filter_by(id=class_id, organization_id=org_id).first_or_404()
    course_id = class_.course_id
    year = class_.study_year
    
    students = Student.query.filter_by(class_id=class_.id).all()
    for student in students:
        FaceEncoding.query.filter_by(student_id=student.id).delete()
        Attendance.query.filter_by(student_id=student.id).delete()
        db.session.delete(student)
    Attendance.query.filter_by(class_id=class_.id).delete()
    
    db.session.delete(class_)
    db.session.commit()
    flash('Class and all its students/attendance records deleted successfully.', 'success')
    return redirect(url_for('college_classes', course_id=course_id, year=year))

@app.route('/college/delete-student/<int:student_id>')
@org_required(['college'])
def college_delete_student(student_id):
    org_id = session.get('org_id')
    student = Student.query.filter_by(id=student_id, organization_id=org_id).first_or_404()
    class_ = Class_.query.get(student.class_id)
    course_id = class_.course_id if class_ else None
    year = class_.study_year if class_ else None
    
    FaceEncoding.query.filter_by(student_id=student.id).delete()
    Attendance.query.filter_by(student_id=student.id).delete()
    
    db.session.delete(student)
    db.session.commit()
    flash('Student deleted successfully.', 'success')
    return redirect(url_for('college_students', course_id=course_id, year=year))


# --- INSTITUTION DELETE ROUTES ---
@app.route('/institution/delete-course/<int:course_id>')
@org_required(['institution'])
def institution_delete_course(course_id):
    org_id = session.get('org_id')
    course = Course.query.filter_by(id=course_id, organization_id=org_id).first_or_404()
    
    # 1. Delete all classes under this course (which cascades to students, attendances, face encodings)
    classes = Class_.query.filter_by(course_id=course.id).all()
    for class_ in classes:
        students = Student.query.filter_by(class_id=class_.id).all()
        for student in students:
            FaceEncoding.query.filter_by(student_id=student.id).delete()
            Attendance.query.filter_by(student_id=student.id).delete()
            db.session.delete(student)
        Attendance.query.filter_by(class_id=class_.id).delete()
        db.session.delete(class_)
        
    # 2. Delete subjects and their attendance
    subjects = Subject.query.filter_by(course_id=course.id).all()
    for subj in subjects:
        Attendance.query.filter_by(subject_id=subj.id).delete()
        db.session.delete(subj)
        
    # 3. Delete StudyYears, Branches, Semesters
    StudyYear.query.filter_by(course_id=course.id).delete()
    Branch.query.filter_by(course_id=course.id).delete()
    Semester.query.filter_by(course_id=course.id).delete()
    
    db.session.delete(course)
    db.session.commit()
    flash('Course and all its classes, subjects, students, branches, semesters, and attendance records deleted successfully.', 'success')
    return redirect(url_for('institution_dashboard'))

@app.route('/institution/delete-branch/<int:branch_id>')
@org_required(['institution'])
def institution_delete_branch(branch_id):
    org_id = session.get('org_id')
    branch = Branch.query.filter_by(id=branch_id, organization_id=org_id).first_or_404()
    course_id = branch.course_id
    
    db.session.delete(branch)
    db.session.commit()
    flash('Branch deleted successfully.', 'success')
    return redirect(url_for('institution_dashboard', course_id=course_id))

@app.route('/institution/delete-year/<int:year_id>')
@org_required(['institution'])
def institution_delete_year(year_id):
    org_id = session.get('org_id')
    study_year = StudyYear.query.filter_by(id=year_id, organization_id=org_id).first_or_404()
    course_id = study_year.course_id
    branch = request.args.get('branch')
    
    db.session.delete(study_year)
    db.session.commit()
    flash('Study Year deleted successfully.', 'success')
    return redirect(url_for('institution_dashboard', course_id=course_id, branch=branch))

@app.route('/institution/delete-sem/<int:sem_id>')
@org_required(['institution'])
def institution_delete_sem(sem_id):
    org_id = session.get('org_id')
    sem = Semester.query.filter_by(id=sem_id, organization_id=org_id).first_or_404()
    course_id = sem.course_id
    branch = request.args.get('branch')
    year = request.args.get('year')
    
    db.session.delete(sem)
    db.session.commit()
    flash('Semester deleted successfully.', 'success')
    return redirect(url_for('institution_dashboard', course_id=course_id, branch=branch, year=year))

@app.route('/institution/delete-subject/<int:subject_id>')
@org_required(['institution'])
def institution_delete_subject(subject_id):
    org_id = session.get('org_id')
    subject = Subject.query.filter_by(id=subject_id, organization_id=org_id).first_or_404()
    
    # Delete subject attendances
    Attendance.query.filter_by(subject_id=subject.id).delete()
    
    db.session.delete(subject)
    db.session.commit()
    flash('Subject and its attendance records deleted successfully.', 'success')
    return redirect(url_for('institution_dashboard'))

@app.route('/institution/delete-class/<int:class_id>')
@org_required(['institution'])
def institution_delete_class(class_id):
    org_id = session.get('org_id')
    class_ = Class_.query.filter_by(id=class_id, organization_id=org_id).first_or_404()
    course_id = class_.course_id
    branch = class_.branch
    year = class_.study_year
    sem = class_.semester
    
    students = Student.query.filter_by(class_id=class_.id).all()
    for student in students:
        FaceEncoding.query.filter_by(student_id=student.id).delete()
        Attendance.query.filter_by(student_id=student.id).delete()
        db.session.delete(student)
    Attendance.query.filter_by(class_id=class_.id).delete()
    
    db.session.delete(class_)
    db.session.commit()
    flash('Class and all its students/attendance records deleted successfully.', 'success')
    return redirect(url_for('institution_classes', course_id=course_id, branch=branch, year=year, sem=sem))

@app.route('/institution/delete-student/<int:student_id>')
@org_required(['institution'])
def institution_delete_student(student_id):
    org_id = session.get('org_id')
    student = Student.query.filter_by(id=student_id, organization_id=org_id).first_or_404()
    class_ = Class_.query.get(student.class_id)
    course_id = class_.course_id if class_ else None
    branch = class_.branch if class_ else None
    year = class_.study_year if class_ else None
    sem = class_.semester if class_ else None
    
    FaceEncoding.query.filter_by(student_id=student.id).delete()
    Attendance.query.filter_by(student_id=student.id).delete()
    
    db.session.delete(student)
    db.session.commit()
    flash('Student deleted successfully.', 'success')
    return redirect(url_for('institution_students', course_id=course_id, branch=branch, year=year, sem=sem))

@app.route('/manifest.json')
def serve_manifest():
    return send_file('static/manifest.json')

@app.route('/.well-known/assetlinks.json')
def assetlinks():
    # Retrieve Google Play SHA256 fingerprint & package name from environment variables
    sha256 = os.environ.get('PLAY_STORE_SHA256', 'YOUR_PLAY_STORE_SHA256_HERE')
    package_name = os.environ.get('PLAY_STORE_PACKAGE', 'com.smartattendance.app')
    return jsonify([{
        "relation": ["delegate_permission/common.handle_all_urls"],
        "target": {
            "namespace": "android_app",
            "package_name": package_name,
            "sha256_cert_fingerprints": [sha256]
        }
    }])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)