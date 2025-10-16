from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///health_tracker.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Models definition - app.py के अंदर ही
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
    
    return render_template('dashboard.html', 
                          user=user, 
                          latest_vitals=latest_vitals,
                          upcoming_immunisations=upcoming_immunisations)

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