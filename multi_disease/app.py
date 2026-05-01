import os
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
try:
    from .config import Config
    from .database import db, User, Prediction
    from .gemini_helper import GeminiHelper
    from .email_helper import EmailHelper
except (ImportError, ValueError):
    from config import Config
    from database import db, User, Prediction
    from gemini_helper import GeminiHelper
    from email_helper import EmailHelper
import pickle
import numpy as np
import json
from geopy.geocoders import Nominatim
import requests
from math import radians, sin, cos, sqrt, atan2

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

with app.app_context():
    db.create_all()

def load_model_and_scaler(disease_type):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, Config.MODELS_PATH, f"{disease_type}_model.pkl")
    scaler_path = os.path.join(base_dir, Config.MODELS_PATH, f"{disease_type}_scaler.pkl")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

def get_risk_category(prediction, probability):
    if prediction == 0:
        return "Low", probability[0] * 100
    else:
        prob = probability[1] * 100
        if prob < 50:
            return "Medium", prob
        else:
            return "High", prob

def find_nearby_hospitals(latitude, longitude, radius=5000):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      node["amenity"="hospital"](around:{radius},{latitude},{longitude});
      way["amenity"="hospital"](around:{radius},{latitude},{longitude});
      relation["amenity"="hospital"](around:{radius},{latitude},{longitude});
    );
    out center;
    """
    
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    
    hospitals = []
    for element in data.get('elements', []):
        if 'tags' in element:
            name = element['tags'].get('name', 'Unnamed Hospital')
            
            if 'lat' in element and 'lon' in element:
                hosp_lat = element['lat']
                hosp_lon = element['lon']
            elif 'center' in element:
                hosp_lat = element['center']['lat']
                hosp_lon = element['center']['lon']
            else:
                continue
            
            distance = calculate_distance(latitude, longitude, hosp_lat, hosp_lon)
            google_maps_link = f"https://www.google.com/maps/dir/{latitude},{longitude}/{hosp_lat},{hosp_lon}"
            
            hospitals.append({
                'name': name,
                'distance': round(distance, 2),
                'google_maps_link': google_maps_link
            })
    
    hospitals.sort(key=lambda x: x['distance'])
    return hospitals[:5]

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        parent_email = request.form.get('parent_email')
        latitude = float(request.form.get('latitude'))
        longitude = float(request.form.get('longitude'))
        
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already exists'}), 400
        
        user = User(email=email, parent_email=parent_email, latitude=latitude, longitude=longitude)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Account created successfully'})
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['user_email'] = user.email
            return jsonify({'success': True, 'redirect': url_for('dashboard')})
        
        return jsonify({'error': 'Invalid credentials'}), 401
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user = db.session.get(User, session['user_id'])
    predictions = Prediction.query.filter_by(user_id=user.id).order_by(Prediction.created_at.desc()).all()
    
    return render_template('dashboard.html', user=user, predictions=predictions)

@app.route('/asthma')
def asthma():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('asthma.html')

@app.route('/diabetes')
def diabetes():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('diabetes.html')

@app.route('/heart')
def heart():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('heart.html')

@app.route('/stroke')
def stroke():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('stroke.html')

@app.route('/predict/<disease_type>', methods=['POST'])
def predict(disease_type):
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    user = db.session.get(User, session['user_id'])
    features = request.json.get('features')

    if not isinstance(features, (list, tuple)):
        return jsonify({'error': 'Features must be a list'}), 400

    model, scaler = load_model_and_scaler(disease_type)
    expected_features = getattr(scaler, 'n_features_in_', None)
    if expected_features is not None and len(features) != expected_features:
        return jsonify({'error': f'Expected {expected_features} features, got {len(features)}'}), 400

    features_array = np.array(features, dtype=float).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    risk_category, risk_score = get_risk_category(prediction, probability)
    risk_score = float(risk_score)
    prediction_result = "Positive" if prediction == 1 else "Negative"
    
    hospitals = []
    gemini_recommendation = ""
    
    if prediction == 1:
        gemini_recommendation = GeminiHelper.get_treatment_recommendation(
            disease_type.capitalize(), 
            prediction_result, 
            risk_score, 
            json.dumps(features)
        )
        
        hospitals = find_nearby_hospitals(user.latitude, user.longitude)
        
        EmailHelper.send_prediction_email(
            user.email,
            user.parent_email,
            user.email.split('@')[0],
            disease_type.capitalize(),
            prediction_result,
            risk_score,
            gemini_recommendation,
            hospitals
        )
    
    new_prediction = Prediction(
        user_id=user.id,
        disease_type=disease_type,
        prediction_result=prediction_result,
        risk_score=risk_score,
        risk_category=risk_category,
        input_features=json.dumps(features),
        gemini_recommendation=gemini_recommendation,
        hospitals=json.dumps(hospitals)
    )
    db.session.add(new_prediction)
    db.session.commit()
    
    return jsonify({
        'prediction_id': new_prediction.id,  # ADD THIS LINE
        'prediction': prediction_result,
        'risk_score': risk_score,
        'risk_category': risk_category,
        'recommendation': gemini_recommendation,
        'hospitals': hospitals
    })

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        user_message = request.json.get('message')
        chat_history = request.json.get('history', [])
        
        response = GeminiHelper.get_chatbot_response(user_message, chat_history)
        
        return jsonify({'response': response})
    
    return render_template('chatbot.html')

@app.route('/results/<int:prediction_id>')
def results(prediction_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    prediction = Prediction.query.get_or_404(prediction_id)
    
    if prediction.user_id != session['user_id']:
        return redirect(url_for('dashboard'))
    
    hospitals = json.loads(prediction.hospitals) if prediction.hospitals else []
    
    return render_template('results.html', prediction=prediction, hospitals=hospitals)

@app.route('/hospital')
def hospital():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user = db.session.get(User, session['user_id'])
    hospitals = find_nearby_hospitals(user.latitude, user.longitude)
    
    return render_template('hospital.html', hospitals=hospitals, user=user)

if __name__ == '__main__':
    app.run(debug=True)