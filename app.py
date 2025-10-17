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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'b00232f07c4572c4e0bc67b2a42bf661'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///health_tracker.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

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
                "I specialize in health-related questions. Could you ask about symptoms, vitals, or general health tips?",
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

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered. Please login.', 'danger')
            return redirect(url_for('login'))
        
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
        email = request.form['email']
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
        # Get form data
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        bp_systolic = int(request.form['bp_systolic'])
        bp_diastolic = int(request.form['bp_diastolic'])
        heart_rate = int(request.form['heart_rate'])
        glucose = float(request.form['glucose'])
        temperature = float(request.form['temperature'])
        oxygen_saturation = int(request.form['oxygen_saturation'])
        
        # Calculate BMI
        bmi = weight / ((height/100) ** 2)
        
        # Create new vitals record
        new_vitals = Vitals(
            user_id=user_id,
            height=height,
            weight=weight,
            bmi=bmi,
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
    
    # Get user's vitals history
    vitals_history = Vitals.query.filter_by(user_id=user_id).order_by(Vitals.date_recorded.desc()).all()
    
    return render_template('vitals.html', vitals_history=vitals_history)

@app.route('/immunisation', methods=['GET', 'POST'])
def immunisation():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    
    if request.method == 'POST':
        vaccine_name = request.form['vaccine_name']
        date_str = request.form['date']
        next_due_date_str = request.form.get('next_due_date')
        notes = request.form.get('notes', '')
        
        # Convert date strings to date objects
        vaccine_date = datetime.strptime(date_str, '%Y-%m-%d').date()
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
    user = User.query.get(user_id)
    
    if request.method == 'POST':
        # Get form data
        age = int(request.form['age'])
        bmi = float(request.form['bmi'])
        bp_systolic = int(request.form['bp_systolic'])
        bp_diastolic = int(request.form['bp_diastolic'])
        glucose = float(request.form['glucose'])
        
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
        
        # Store prediction in database
        new_prediction = DiseasePrediction(
            user_id=user_id,
            age=age,
            bmi=bmi,
            bp_systolic=bp_systolic,
            bp_diastolic=bp_diastolic,
            glucose=glucose,
            risk_level=risk_level,
            risk_probability=max(probability) * 100
        )
        
        db.session.add(new_prediction)
        db.session.commit()
        
        return render_template('prediction.html', 
                              prediction=risk_level, 
                              description=risk_desc,
                              probability=max(probability) * 100,
                              show_result=True)
    
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
        doctor_name = request.form['doctor_name']
        specialization = request.form['specialization']
        appointment_date = request.form['appointment_date']
        symptoms = request.form['symptoms']
        
        # Convert date string to datetime
        appointment_datetime = datetime.strptime(appointment_date, '%Y-%m-%dT%H:%M')
        
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
    
    return render_template('appointments.html', appointments=appointments,now=datetime.now())

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

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)