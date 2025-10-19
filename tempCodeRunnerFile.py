from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import math
from dotenv import load_dotenv
import csv
from io import StringIO
import base64
from io import BytesIO
from PIL import Image

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'fallback-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///health_tracker.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

@app.context_processor
def inject_datetime():
    return {
        'datetime': datetime,
        'date': date,
        'now': datetime.now,
        'math': math
    }


# Models definition
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    date_joined = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    vitals = db.relationship('Vitals', backref='user', lazy=True)
    immunisations = db.relationship('Immunisation', backref='user', lazy=True)
    predictions = db.relationship('DiseasePrediction', backref='user', lazy=True)
    chat_messages = db.relationship('ChatMessage', backref='user', lazy=True)
    doctor_appointments = db.relationship('DoctorAppointment', backref='user', lazy=True)
    profile = db.relationship('UserProfile', backref='user', uselist=False, cascade='all, delete-orphan')

class UserProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    profile_picture = db.Column(db.Text)  # Store as base64
    phone = db.Column(db.String(20))
    address = db.Column(db.Text)
    date_of_birth = db.Column(db.Date)
    gender = db.Column(db.String(20))
    emergency_contact = db.Column(db.String(100))
    emergency_phone = db.Column(db.String(20))
    blood_group = db.Column(db.String(10))
    allergies = db.Column(db.Text)
    medical_conditions = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

@property
def age(self):
        if self.date_of_birth:
            today = date.today()
            return today.year - self.date_of_birth.year - (
                (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
            )
        return None

class Vitals(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date_recorded = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Basic measurements
    height = db.Column(db.Float)  # in cm
    weight = db.Column(db.Float)  # in kg
    bmi = db.Column(db.Float)     # calculated
    
    # Cardiovascular
    bp_systolic = db.Column(db.Integer)   # systolic blood pressure
    bp_diastolic = db.Column(db.Integer)  # diastolic blood pressure
    heart_rate = db.Column(db.Integer)    # beats per minute
    
    # Metabolic
    glucose = db.Column(db.Float)         # blood glucose level
    
    # Other
    temperature = db.Column(db.Float)     # body temperature in Celsius
    oxygen_saturation = db.Column(db.Integer)  # SpO2 percentage

class Immunisation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    vaccine_name = db.Column(db.String(200), nullable=False)
    date = db.Column(db.Date, nullable=False)
    next_due_date = db.Column(db.Date)
    notes = db.Column(db.Text)

class DiseasePrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prediction_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Input features
    age = db.Column(db.Integer)
    bmi = db.Column(db.Float)
    bp_systolic = db.Column(db.Integer)
    bp_diastolic = db.Column(db.Integer)
    glucose = db.Column(db.Float)
    
    # Prediction results
    risk_level = db.Column(db.String(50))
    risk_probability = db.Column(db.Float)

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    is_user = db.Column(db.Boolean, default=True)

class DoctorAppointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    doctor_name = db.Column(db.String(200), nullable=False)
    specialization = db.Column(db.String(100), nullable=False)
    appointment_date = db.Column(db.DateTime, nullable=False)
    symptoms = db.Column(db.Text)
    status = db.Column(db.String(50), default='Scheduled')  # Scheduled, Completed, Cancelled
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ML Model for disease prediction
def train_prediction_model():
    # This is a simplified example - in a real app, you'd use a proper dataset
    # Sample training data (features: age, bmi, bp_systolic, bp_diastolic, glucose)
    X_train = np.array([
        [25, 22, 120, 80, 90],
        [45, 28, 140, 90, 110],
        [35, 25, 130, 85, 100],
        [50, 30, 150, 95, 120],
        [30, 23, 125, 82, 95],
        [55, 32, 160, 100, 130],
        [40, 27, 135, 88, 105]
    ])
    
    # Sample labels (0: low risk, 1: medium risk, 2: high risk)
    y_train = np.array([0, 1, 0, 2, 0, 2, 1])
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, 'disease_prediction_model.pkl')
    return model

# Load or train the model
if os.path.exists('disease_prediction_model.pkl'):
    prediction_model = joblib.load('disease_prediction_model.pkl')
else:
    prediction_model = train_prediction_model()

# Doctor Database (In real app, this would be in database)
DOCTORS = [
    {"name": "Dr. Sharma", "specialization": "Cardiologist", "experience": "15 years", "rating": 4.8},
    {"name": "Dr. Patel", "specialization": "General Physician", "experience": "10 years", "rating": 4.6},
    {"name": "Dr. Kumar", "specialization": "Neurologist", "experience": "12 years", "rating": 4.7},
    {"name": "Dr. Gupta", "specialization": "Orthopedic", "experience": "8 years", "rating": 4.5},
    {"name": "Dr. Singh", "specialization": "Pediatrician", "experience": "14 years", "rating": 4.9},
    {"name": "Dr. Reddy", "specialization": "Dermatologist", "experience": "11 years", "rating": 4.6}
]

# Health Chatbot Logic
class HealthChatbot:
    def __init__(self):
        self.responses = {
            'greeting': [
                "Hello! I'm your Health Assistant. How can I help you with your health concerns today?",
                "Hi there! I'm here to help with your health questions. What would you like to know?",
                "Welcome! I'm your health companion. How can I assist you today?"
            ],
            'symptoms': {
                'fever': "Fever can be caused by infections. Rest, stay hydrated, and monitor your temperature. If it's above 102°F or lasts more than 3 days, consult a doctor.",
                'headache': "Headaches can be due to stress, dehydration, or tension. Try resting in a quiet room, staying hydrated, and if persistent, consult a doctor.",
                'cough': "For cough, stay hydrated and use honey in warm water. If accompanied by fever or breathing difficulty, seek medical attention.",
                'blood pressure': "Normal blood pressure is around 120/80 mmHg. High BP can be managed with diet, exercise, and medication. Low BP may need increased salt intake.",
                'diabetes': "Diabetes management involves monitoring blood sugar, healthy diet, regular exercise, and medication as prescribed by your doctor.",
                'weight': "Maintain healthy weight through balanced diet and regular exercise. BMI between 18.5-24.9 is considered healthy."
            },
            'general_health': [
                "Regular exercise, balanced diet, and adequate sleep are key to good health.",
                "Stay hydrated by drinking 8-10 glasses of water daily for optimal health.",
                "Preventive care including regular check-ups and vaccinations is important."
            ],
            'vitals': [
                "You can track your vitals like blood pressure, heart rate, and glucose levels in the Vitals section.",
                "Regular monitoring of vitals helps in early detection of health issues."
            ],
            'immunization': [
                "Keep your immunizations up to date for protection against various diseases.",
                "Check the Immunization section to track your vaccination records."
            ],
            'doctor': [
                "I can help you find a doctor based on your symptoms. What type of specialist are you looking for?",
                "You can book appointments with doctors through our platform. Let me know your health concern."
            ],
            'emergency': [
                "For emergencies, please call emergency services immediately! I can also help you find nearby hospitals."
            ],
            'fallback': [
                "I'm not sure I understand. Could you rephrase your question about health?",
                "I specialize in health-related questions. Could you ask about symptoms, medications, or health tips?",
                "I'm here to help with health concerns. Try asking about symptoms, medications, or health tips."
            ]
        }

    def get_response(self, message, user_data=None):
        message = message.lower().strip()
        
        # Greeting detection
        if any(word in message for word in ['hello', 'hi', 'hey', 'greetings']):
            return np.random.choice(self.responses['greeting'])
        
        # Symptom-related queries
        for symptom, response in self.responses['symptoms'].items():
            if symptom in message:
                return response
        
        # Doctor-related queries
        if any(word in message for word in ['doctor', 'appointment', 'specialist', 'consult']):
            return np.random.choice(self.responses['doctor'])
        
        # Emergency detection
        if any(word in message for word in ['emergency', 'urgent', 'help immediately', '911', 'hospital']):
            return np.random.choice(self.responses['emergency'])
        
        # General health queries
        if any(word in message for word in ['exercise', 'diet', 'sleep', 'healthy']):
            return np.random.choice(self.responses['general_health'])
        
        # Vitals queries
        if any(word in message for word in ['vital', 'blood pressure', 'heart rate', 'bmi', 'glucose']):
            return np.random.choice(self.responses['vitals'])
        
        # Immunization queries
        if any(word in message for word in ['vaccine', 'immunization', 'vaccination']):
            return np.random.choice(self.responses['immunization'])
        
        # Fallback response
        return np.random.choice(self.responses['fallback'])

# Initialize chatbot
health_bot = HealthChatbot()

# Password strength validation
def validate_password_strength(password):
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not any(char.isdigit() for char in password):
        return False, "Password must contain at least one digit"
    
    if not any(char.isupper() for char in password):
        return False, "Password must contain at least one uppercase letter"
    
    if not any(char.islower() for char in password):
        return False, "Password must contain at least one lowercase letter"
    
    return True, "Password is strong"

# Helper function to convert Vitals object to dictionary
def vital_to_dict(vital):
    """Convert Vitals SQLAlchemy object to dictionary"""
    return {
        'id': vital.id,
        'user_id': vital.user_id,
        'date_recorded': vital.date_recorded.isoformat() if vital.date_recorded else None,
        'height': vital.height,
        'weight': vital.weight,
        'bmi': vital.bmi,
        'bp_systolic': vital.bp_systolic,
        'bp_diastolic': vital.bp_diastolic,
        'heart_rate': vital.heart_rate,
        'glucose': vital.glucose,
        'temperature': vital.temperature,
        'oxygen_saturation': vital.oxygen_saturation
    }

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip().lower()
        password = request.form['password']
        
        # Validate password strength
        is_strong, message = validate_password_strength(password)
        if not is_strong:
            flash(message, 'danger')
            return redirect(url_for('register'))
        
        # Check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered. Please login.', 'danger')
            return redirect(url_for('login'))
        
        # Check username availability
        existing_username = User.query.filter_by(username=username).first()
        if existing_username:
            flash('Username already taken. Please choose another.', 'danger')
            return redirect(url_for('register'))
        
        # Create new user
        new_user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password']
        
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    user = User.query.get(user_id)
    
    # Get latest vitals
    latest_vitals = Vitals.query.filter_by(user_id=user_id).order_by(Vitals.date_recorded.desc()).first()
    
    # Get upcoming immunisations
    upcoming_immunisations = Immunisation.query.filter(
        Immunisation.user_id == user_id,
        Immunisation.date >= date.today()
    ).order_by(Immunisation.date.asc()).limit(5).all()
    
    # Get upcoming appointments
    upcoming_appointments = DoctorAppointment.query.filter(
        DoctorAppointment.user_id == user_id,
        DoctorAppointment.appointment_date >= datetime.now()
    ).order_by(DoctorAppointment.appointment_date.asc()).limit(3).all()
    
    return render_template('dashboard.html', 
                          user=user, 
                          latest_vitals=latest_vitals,
                          upcoming_immunisations=upcoming_immunisations,
                          upcoming_appointments=upcoming_appointments)

@app.route('/vitals', methods=['GET', 'POST'])
def vitals():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    
    if request.method == 'POST':
        try:
            # Get form data with validation
            height = float(request.form.get('height', 0))
            weight = float(request.form.get('weight', 0))
            bp_systolic = int(request.form.get('bp_systolic', 0))
            bp_diastolic = int(request.form.get('bp_diastolic', 0))
            heart_rate = int(request.form.get('heart_rate', 0))
            glucose = float(request.form.get('glucose', 0))
            temperature = float(request.form.get('temperature', 0))
            oxygen_saturation = int(request.form.get('oxygen_saturation', 0))
            
            # Validate input ranges
            if not (100 <= height <= 250):  # cm
                flash('Please enter a valid height (100-250 cm)', 'danger')
                return redirect(url_for('vitals'))
            
            if not (30 <= weight <= 300):  # kg
                flash('Please enter a valid weight (30-300 kg)', 'danger')
                return redirect(url_for('vitals'))
            
            if not (60 <= bp_systolic <= 250):
                flash('Please enter valid systolic blood pressure (60-250 mmHg)', 'danger')
                return redirect(url_for('vitals'))
            
            if not (40 <= bp_diastolic <= 150):
                flash('Please enter valid diastolic blood pressure (40-150 mmHg)', 'danger')
                return redirect(url_for('vitals'))
            
            # Calculate BMI
            bmi = weight / ((height/100) ** 2)
            
            # Create new vitals record
            new_vitals = Vitals(
                user_id=user_id,
                height=height,
                weight=weight,
                bmi=round(bmi, 2),
                bp_systolic=bp_systolic,
                bp_diastolic=bp_diastolic,
                heart_rate=heart_rate,
                glucose=glucose,
                temperature=temperature,
                oxygen_saturation=oxygen_saturation
            )
            
            db.session.add(new_vitals)
            db.session.commit()
            
            flash('Vitals recorded successfully!', 'success')
            return redirect(url_for('vitals'))
            
        except ValueError as e:
            flash('Please enter valid numeric values for all fields.', 'danger')
            return redirect(url_for('vitals'))
        except Exception as e:
            flash('An error occurred while saving vitals.', 'danger')
            return redirect(url_for('vitals'))
    
    # Get user's vitals history
    vitals_history = Vitals.query.filter_by(user_id=user_id).order_by(Vitals.date_recorded.desc()).all()
    
    # Convert to list of dictionaries for JSON serialization
    vitals_history_dict = [vital_to_dict(vital) for vital in vitals_history]
    
    return render_template('vitals.html', 
                         vitals_history=vitals_history,
                         vitals_history_json=vitals_history_dict)

@app.route('/immunisation', methods=['GET', 'POST'])
def immunisation():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    
    if request.method == 'POST':
        try:
            vaccine_name = request.form['vaccine_name'].strip()
            date_str = request.form['date']
            next_due_date_str = request.form.get('next_due_date')
            notes = request.form.get('notes', '')
            
            if not vaccine_name:
                flash('Please enter vaccine name.', 'danger')
                return redirect(url_for('immunisation'))
            
            # Convert date strings to date objects
            vaccine_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            
            if vaccine_date > date.today():
                flash('Vaccination date cannot be in the future.', 'danger')
                return redirect(url_for('immunisation'))
            
            next_due_date = datetime.strptime(next_due_date_str, '%Y-%m-%d').date() if next_due_date_str else None
            
            # Create new immunisation record
            new_immunisation = Immunisation(
                user_id=user_id,
                vaccine_name=vaccine_name,
                date=vaccine_date,
                next_due_date=next_due_date,
                notes=notes
            )
            
            db.session.add(new_immunisation)
            db.session.commit()
            
            flash('Immunisation record added successfully!', 'success')
            return redirect(url_for('immunisation'))
            
        except ValueError:
            flash('Invalid date format. Please use YYYY-MM-DD format.', 'danger')
            return redirect(url_for('immunisation'))
        except Exception as e:
            flash('An error occurred while saving immunisation record.', 'danger')
            return redirect(url_for('immunisation'))
    
    # Get user's immunisation records
    immunisation_records = Immunisation.query.filter_by(user_id=user_id).order_by(Immunisation.date.desc()).all()
    
    return render_template('immunisation.html', immunisation_records=immunisation_records)

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    
    # Get all user data for history
    vitals_history = Vitals.query.filter_by(user_id=user_id).order_by(Vitals.date_recorded.desc()).all()
    immunisation_history = Immunisation.query.filter_by(user_id=user_id).order_by(Immunisation.date.desc()).all()
    prediction_history = DiseasePrediction.query.filter_by(user_id=user_id).order_by(DiseasePrediction.prediction_date.desc()).all()
    
    return render_template('history.html', 
                          vitals_history=vitals_history,
                          immunisation_history=immunisation_history,
                          prediction_history=prediction_history)

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    
    if request.method == 'POST':
        try:
            # Get form data with validation
            age = int(request.form.get('age', 0))
            bmi = float(request.form.get('bmi', 0))
            bp_systolic = int(request.form.get('bp_systolic', 0))
            bp_diastolic = int(request.form.get('bp_diastolic', 0))
            glucose = float(request.form.get('glucose', 0))
            
            # Validate inputs
            if not (1 <= age <= 120):
                flash('Please enter a valid age (1-120 years)', 'danger')
                return redirect(url_for('prediction'))
            
            if not (10 <= bmi <= 50):
                flash('Please enter a valid BMI (10-50)', 'danger')
                return redirect(url_for('prediction'))
            
            # Make prediction
            features = np.array([[age, bmi, bp_systolic, bp_diastolic, glucose]])
            prediction = prediction_model.predict(features)[0]
            probability = prediction_model.predict_proba(features)[0]
            
            # Risk levels
            risk_levels = ['Low Risk', 'Medium Risk', 'High Risk']
            risk_description = [
                "You have a low risk of developing lifestyle diseases. Maintain your healthy habits!",
                "You have a medium risk of developing lifestyle diseases. Consider making some lifestyle improvements.",
                "You have a high risk of developing lifestyle diseases. Please consult with a healthcare professional."
            ]
            
            risk_level = risk_levels[prediction]
            risk_desc = risk_description[prediction]
            risk_prob = round(max(probability) * 100, 2)
            
            # Store prediction in database
            new_prediction = DiseasePrediction(
                user_id=user_id,
                age=age,
                bmi=bmi,
                bp_systolic=bp_systolic,
                bp_diastolic=bp_diastolic,
                glucose=glucose,
                risk_level=risk_level,
                risk_probability=risk_prob
            )
            
            db.session.add(new_prediction)
            db.session.commit()
            
            return render_template('prediction.html', 
                                  prediction=risk_level, 
                                  description=risk_desc,
                                  probability=risk_prob,
                                  show_result=True)
            
        except Exception as e:
            flash('An error occurred during prediction. Please check your inputs.', 'danger')
            return redirect(url_for('prediction'))
    
    # Get latest vitals for pre-filling the form
    latest_vitals = Vitals.query.filter_by(user_id=user_id).order_by(Vitals.date_recorded.desc()).first()
    
    return render_template('prediction.html', latest_vitals=latest_vitals, show_result=False)

# New Features Routes

@app.route('/doctors')
def doctors():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    specialization = request.args.get('specialization', '')
    
    if specialization:
        filtered_doctors = [doc for doc in DOCTORS if specialization.lower() in doc['specialization'].lower()]
    else:
        filtered_doctors = DOCTORS
    
    return render_template('doctors.html', doctors=filtered_doctors, specialization=specialization)

@app.route('/book_appointment', methods=['GET', 'POST'])
def book_appointment():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    
    if request.method == 'POST':
        try:
            doctor_name = request.form.get('doctor_name', '').strip()
            specialization = request.form.get('specialization', '').strip()
            appointment_date = request.form.get('appointment_date', '').strip()
            symptoms = request.form.get('symptoms', '').strip()
            
            # Validate required fields
            if not all([doctor_name, specialization, appointment_date]):
                flash('Please fill all required fields.', 'danger')
                return redirect(url_for('book_appointment'))
            
            # Convert date string to datetime
            appointment_datetime = datetime.strptime(appointment_date, '%Y-%m-%dT%H:%M')
            
            # Check if appointment is in the future
            if appointment_datetime <= datetime.now():
                flash('Please select a future date and time for appointment.', 'danger')
                return redirect(url_for('book_appointment'))
            
            # Create new appointment
            new_appointment = DoctorAppointment(
                user_id=user_id,
                doctor_name=doctor_name,
                specialization=specialization,
                appointment_date=appointment_datetime,
                symptoms=symptoms
            )
            
            db.session.add(new_appointment)
            db.session.commit()
            
            flash('Appointment booked successfully!', 'success')
            return redirect(url_for('appointments'))
            
        except ValueError:
            flash('Invalid date format. Please use the date picker.', 'danger')
            return redirect(url_for('book_appointment'))
        except Exception as e:
            flash('An error occurred while booking appointment.', 'danger')
            return redirect(url_for('book_appointment'))
    
    doctor_name = request.args.get('doctor', '')
    specialization = request.args.get('specialization', '')
    
    return render_template('book_appointment.html', 
                         doctor_name=doctor_name, 
                         specialization=specialization,
                         doctors=DOCTORS,
                         now=datetime.now())

@app.route('/appointments')
def appointments():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    appointments = DoctorAppointment.query.filter_by(user_id=user_id).order_by(DoctorAppointment.appointment_date.desc()).all()
    
    return render_template('appointments.html', appointments=appointments, now=datetime.now())

@app.route('/emergency')
def emergency():
    return render_template('emergency.html')

@app.route('/api/nearby_hospitals')
def nearby_hospitals():
    # In a real app, you would use Google Places API or similar
    # For demo, we'll return sample data
    lat = request.args.get('lat')
    lng = request.args.get('lng')
    
    # Sample hospital data (in real app, this would come from an API)
    sample_hospitals = [
        {
            'name': 'City General Hospital',
            'address': '123 Medical Center, Downtown',
            'distance': '1.2 km',
            'phone': '+1-234-567-8900',
            'emergency': True,
            'lat': float(lat) + 0.01 if lat else 28.6139,
            'lng': float(lng) + 0.01 if lng else 77.2090
        },
        {
            'name': 'Community Health Center',
            'address': '456 Health Street, Midtown',
            'distance': '2.5 km',
            'phone': '+1-234-567-8901',
            'emergency': True,
            'lat': float(lat) - 0.01 if lat else 28.6039,
            'lng': float(lng) - 0.01 if lng else 77.1990
        },
        {
            'name': 'Metropolitan Hospital',
            'address': '789 Care Avenue, Uptown',
            'distance': '3.1 km',
            'phone': '+1-234-567-8902',
            'emergency': True,
            'lat': float(lat) + 0.02 if lat else 28.6239,
            'lng': float(lng) - 0.02 if lng else 77.1890
        }
    ]
    
    return jsonify(sample_hospitals)

# Chatbot Routes
@app.route('/chatbot')
def chatbot():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    # Get chat history
    chat_history = ChatMessage.query.filter_by(user_id=user_id).order_by(ChatMessage.timestamp.asc()).limit(50).all()
    
    return render_template('chatbot.html', chat_history=chat_history)

@app.route('/api/chat/send', methods=['POST'])
def chat_send():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_id = session['user_id']
    data = request.get_json()
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({'error': 'Empty message'}), 400
    
    # Get bot response
    bot_response = health_bot.get_response(message)
    
    # Save message to database
    chat_message = ChatMessage(
        user_id=user_id,
        message=message,
        response=bot_response,
        is_user=True
    )
    db.session.add(chat_message)
    db.session.commit()
    
    return jsonify({
        'user_message': message,
        'bot_response': bot_response,
        'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/api/chat/history')
def chat_history():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_id = session['user_id']
    chat_messages = ChatMessage.query.filter_by(user_id=user_id).order_by(ChatMessage.timestamp.asc()).limit(50).all()
    
    history = []
    for msg in chat_messages:
        history.append({
            'is_user': msg.is_user,
            'message': msg.message if msg.is_user else msg.response,
            'timestamp': msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return jsonify(history)

@app.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_id = session['user_id']
    # Delete user's chat history
    ChatMessage.query.filter_by(user_id=user_id).delete()
    db.session.commit()
    
    return jsonify({'success': True})

@app.route('/api/vitals/chart')
def vitals_chart_data():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_id = session['user_id']
    
    # Get last 7 vitals records
    vitals = Vitals.query.filter_by(user_id=user_id).order_by(Vitals.date_recorded.desc()).limit(7).all()
    
    # Reverse to show chronological order
    vitals.reverse()
    
    dates = [v.date_recorded.strftime('%Y-%m-%d') for v in vitals]
    weights = [v.weight for v in vitals]
    bmis = [v.bmi for v in vitals]
    heart_rates = [v.heart_rate for v in vitals]
    glucose_levels = [v.glucose for v in vitals]
    
    return jsonify({
        'dates': dates,
        'weights': weights,
        'bmis': bmis,
        'heart_rates': heart_rates,
        'glucose_levels': glucose_levels
    })

@app.route('/send_email_report', methods=['POST'])
def send_email_report():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    # Implement actual email sending with SMTP
    # For now, return a placeholder response
    flash('Email report feature coming soon!', 'info')
    return jsonify({'success': True, 'message': 'Report will be sent via email'})

# Export Routes
@app.route('/api/vitals/<int:vital_id>')
def get_vital_details(vital_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_id = session['user_id']
    vital = Vitals.query.filter_by(id=vital_id, user_id=user_id).first()
    
    if not vital:
        return jsonify({'error': 'Vital record not found'}), 404
    
    return jsonify(vital_to_dict(vital))

@app.route('/api/vitals/clear', methods=['POST'])
def clear_vitals_history():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_id = session['user_id']
    
    try:
        # Delete all vitals for the user
        Vitals.query.filter_by(user_id=user_id).delete()
        db.session.commit()
        
        flash('Vitals history cleared successfully!', 'success')
        return jsonify({'success': True})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Failed to clear history'}), 500

@app.route('/export/vitals/pdf')
def export_vitals_pdf():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    vitals_history = Vitals.query.filter_by(user_id=user_id).order_by(Vitals.date_recorded.desc()).all()
    
    # For now, we'll return a simple HTML page that users can print as PDF
    from flask import render_template_string
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Health Vitals Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; }}
            .table {{ width: 100%; border-collapse: collapse; }}
            .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .table th {{ background-color: #f2f2f2; }}
            .summary {{ margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }}
            .footer {{ margin-top: 30px; text-align: center; font-size: 12px; color: #666; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Health Vitals Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>User: {session.get('username', 'Unknown')}</p>
        </div>
        
        <div class="summary">
            <h3>Report Summary</h3>
            <p>Total Records: {len(vitals_history)}</p>
            <p>Date Range: {vitals_history[-1].date_recorded.strftime('%Y-%m-%d') if vitals_history else 'N/A'} 
               to {vitals_history[0].date_recorded.strftime('%Y-%m-%d') if vitals_history else 'N/A'}</p>
        </div>
        
        <table class="table">
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Height (cm)</th>
                    <th>Weight (kg)</th>
                    <th>BMI</th>
                    <th>Blood Pressure</th>
                    <th>Heart Rate</th>
                    <th>Glucose</th>
                    <th>Temp (°C)</th>
                    <th>Oxygen (%)</th>
                </tr>
            </thead>
            <tbody>
                {"".join([f"""
                <tr>
                    <td>{vital.date_recorded.strftime('%Y-%m-%d')}</td>
                    <td>{vital.height}</td>
                    <td>{vital.weight}</td>
                    <td>{vital.bmi:.1f}</td>
                    <td>{vital.bp_systolic}/{vital.bp_diastolic}</td>
                    <td>{vital.heart_rate}</td>
                    <td>{vital.glucose}</td>
                    <td>{vital.temperature}</td>
                    <td>{vital.oxygen_saturation}</td>
                </tr>
                """ for vital in vitals_history])}
            </tbody>
        </table>
        
        <div class="footer">
            <p>This report was generated by Health World - Your Personal Health Tracker</p>
            <p>For medical advice, please consult with healthcare professionals.</p>
        </div>
        
        <script>
            window.onload = function() {{
                window.print();
            }};
        </script>
    </body>
    </html>
    """
    
    return render_template_string(html_content)

@app.route('/api/vitals/export/<format_type>')
def export_vitals_data(format_type):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_id = session['user_id']
    vitals_history = Vitals.query.filter_by(user_id=user_id).order_by(Vitals.date_recorded.desc()).all()
    
    # Convert to list of dictionaries
    vitals_data = [vital_to_dict(vital) for vital in vitals_history]
    
    if format_type == 'json':
        return jsonify(vitals_data)
    elif format_type == 'csv':
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Date', 'Height (cm)', 'Weight (kg)', 'BMI', 'Systolic BP', 'Diastolic BP', 
                        'Heart Rate (bpm)', 'Glucose (mg/dL)', 'Temperature (°C)', 'Oxygen Saturation (%)'])
        
        # Write data
        for vital in vitals_data:
            writer.writerow([
                vital['date_recorded'][:10],  # Just the date part
                vital['height'],
                vital['weight'],
                vital['bmi'],
                vital['bp_systolic'],
                vital['bp_diastolic'],
                vital['heart_rate'],
                vital['glucose'],
                vital['temperature'],
                vital['oxygen_saturation']
            ])
        
        response = app.response_class(
            response=output.getvalue(),
            status=200,
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename=health_vitals_{datetime.now().strftime("%Y%m%d")}.csv'}
        )
        return response
    
    return jsonify({'error': 'Invalid format'}), 400

# Profile Management Routes
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    user = User.query.get(user_id)
    profile = UserProfile.query.filter_by(user_id=user_id).first()
    
    if request.method == 'POST':
        try:
            # Get form data
            phone = request.form.get('phone', '').strip()
            address = request.form.get('address', '').strip()
            date_of_birth_str = request.form.get('date_of_birth')
            gender = request.form.get('gender', '').strip()
            emergency_contact = request.form.get('emergency_contact', '').strip()
            emergency_phone = request.form.get('emergency_phone', '').strip()
            blood_group = request.form.get('blood_group', '').strip()
            allergies = request.form.get('allergies', '').strip()
            medical_conditions = request.form.get('medical_conditions', '').strip()
            
            # Handle profile picture
            profile_picture = None
            if 'profile_picture' in request.files:
                file = request.files['profile_picture']
                if file and file.filename:
                    # Validate file type
                    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                        flash('Please upload a valid image file (PNG, JPG, JPEG, GIF)', 'danger')
                        return redirect(url_for('profile'))
                    
                    # Validate file size (max 5MB)
                    file.seek(0, os.SEEK_END)
                    file_length = file.tell()
                    file.seek(0)
                    if file_length > 5 * 1024 * 1024:
                        flash('Profile picture must be less than 5MB', 'danger')
                        return redirect(url_for('profile'))
                    
                    # Process image
                    try:
                        image = Image.open(file)
                        # Resize image to max 500x500
                        image.thumbnail((500, 500), Image.Resampling.LANCZOS)
                        
                        # Convert to base64
                        buffered = BytesIO()
                        image.save(buffered, format="PNG")
                        profile_picture = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    except Exception as e:
                        flash('Error processing image. Please try another image.', 'danger')
                        return redirect(url_for('profile'))
            
            # Convert date string to date object
            date_of_birth = None
            if date_of_birth_str:
                date_of_birth = datetime.strptime(date_of_birth_str, '%Y-%m-%d').date()
            
            if profile:
                # Update existing profile
                profile.phone = phone
                profile.address = address
                profile.date_of_birth = date_of_birth
                profile.gender = gender
                profile.emergency_contact = emergency_contact
                profile.emergency_phone = emergency_phone
                profile.blood_group = blood_group
                profile.allergies = allergies
                profile.medical_conditions = medical_conditions
                profile.updated_at = datetime.utcnow()
                
                if profile_picture:
                    profile.profile_picture = profile_picture
            else:
                # Create new profile
                profile = UserProfile(
                    user_id=user_id,
                    phone=phone,
                    address=address,
                    date_of_birth=date_of_birth,
                    gender=gender,
                    emergency_contact=emergency_contact,
                    emergency_phone=emergency_phone,
                    blood_group=blood_group,
                    allergies=allergies,
                    medical_conditions=medical_conditions,
                    profile_picture=profile_picture
                )
                db.session.add(profile)
            
            db.session.commit()
            flash('Profile updated successfully!', 'success')
            return redirect(url_for('profile'))
            
        except Exception as e:
            db.session.rollback()
            flash('An error occurred while updating profile.', 'danger')
            return redirect(url_for('profile'))
    
    return render_template('profile.html', user=user, profile=profile)

@app.route('/profile/remove_picture', methods=['POST'])
def remove_profile_picture():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_id = session['user_id']
    profile = UserProfile.query.filter_by(user_id=user_id).first()
    
    if profile:
        profile.profile_picture = None
        db.session.commit()
        flash('Profile picture removed successfully!', 'success')
    
    return redirect(url_for('profile'))

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)