from flask import Flask, render_template, Response, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from flask_pymongo import PyMongo
import cv2
import numpy as np
from ultralytics import YOLO
import os

# Flask app setup
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Use a strong secret key for session management

# MongoDB setup
app.config['MONGO_URI'] = os.getenv("MONGO_URI")  # Use environment variable for security
mongo = PyMongo(app)

# Load YOLO model
model = YOLO('yolov10n.pt')

# Check if running on Render (headless server)
if "RENDER" in os.environ:
    cap = None  # Disable webcam access on Render
else:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Warning: Webcam not accessible. Falling back to sample video.")
        cap = cv2.VideoCapture("sample_video.mp4")  # Add a sample video file

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    try:
        file = request.files['frame']
        frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Perform object detection
        results = model(frame)
        detected_objects = []
        
        for result in results:
            labels = result.names
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for i, box in enumerate(boxes):
                detected_objects.append({
                    "label": labels[int(classes[i])],
                    "box": box.tolist()
                })

        return jsonify({"detections": detected_objects})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('index'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']

        existing_email = mongo.db.users.find_one({'email': email})
        existing_username = mongo.db.users.find_one({'username': username})

        if existing_email:
            flash('This email is already registered.', 'danger')
        elif existing_username:
            flash('This username is already taken.', 'danger')
        else:
            hashed_password = generate_password_hash(password)
            mongo.db.users.insert_one({'email': email, 'username': username, 'password': hashed_password})
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = mongo.db.users.find_one({'username': username})

        if user and check_password_hash(user['password'], password):
            session['username'] = username
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password!', 'danger')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

@app.route('/dashboard')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Ensure Render assigns a port
    app.run(host="0.0.0.0", port=port, debug=True)
