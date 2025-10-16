from app import db
from datetime import datetime, date

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