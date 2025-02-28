from flask import Flask, render_template, Response, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_pymongo import PyMongo
import cv2
from ultralytics import YOLO
import numpy as np

# Flask app setup
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Use a strong secret key for session management

# MongoDB setup
app.config['MONGO_URI'] = 'mongodb+srv://bhumijain:test123@cluster0.kzxid0j.mongodb.net/node-tuts?retryWrites=true&w=majority'  # Replace with your MongoDB URI
mongo = PyMongo(app)

# Load YOLO model
model = YOLO('yolov10n.pt')

# Initialize the webcam
import os

# Try opening webcam, else fallback to a sample video file
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Warning: Webcam not accessible. Falling back to sample video.")
    cap = cv2.VideoCapture("sample_video.mp4")  # Add a sample video file


# Helper function for classifying apples
def classify_apple(frame, boxes, labels, classes):
    ripe_count, overripe_count, unripe_count = 0, 0, 0
    # Iterate through detected objects
    for i, box in enumerate(boxes):
        # Sample classification (customize this logic for your use case)
        if labels[int(classes[i])] == 'apple':
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0)  # Default color for ripe (green)
            if np.random.random() > 0.66:
                color = (0, 0, 255)  # Overripe (red)
                overripe_count += 1
            elif np.random.random() > 0.33:
                color = (255, 0, 0)  # Unripe (blue)
                unripe_count += 1
            else:
                ripe_count += 1

            # Draw the bounding box around the detected object
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, labels[int(classes[i])], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame, ripe_count, overripe_count, unripe_count

# Function to generate frames for the video feed
def generate_frames():
    while True:
        try:
            # Capture frame from webcam
            success, frame = cap.read()
            if not success:
                print("Error: Unable to read from webcam")
                break

            # Perform object detection using YOLO model
            results = model(frame)

            for result in results:
                labels = result.names  # Object class labels
                boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
                classes = result.boxes.cls.cpu().numpy()  # Class predictions

                # Classify apples in the frame
                frame, ripe_count, overripe_count, unripe_count = classify_apple(frame, boxes, labels, classes)

            # Display classification counts on the frame
            cv2.putText(frame, f'Ripe: {ripe_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Overripe: {overripe_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f'Unripe: {unripe_count}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Encode the processed frame for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Error: Failed to encode frame")
                continue

            # Yield the frame as part of the video stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except Exception as e:
            print(f"Error in generate_frames: {e}")
            break

# Routes for your Flask app
@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('login'))
    return redirect(url_for('login'))  # Redirect to the login page if not logged in


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']

        # Check if email or username already exists
        existing_email = mongo.db.users.find_one({'email': email})
        existing_username = mongo.db.users.find_one({'username': username})

        if existing_email:
            flash('This email is already registered.', 'danger')
        elif existing_username:
            flash('This username is already taken.', 'danger')
        else:
            # Hash the password and save to MongoDB
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

        # Fetch user from MongoDB
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
    return render_template('index.html')  # Your existing index.html page

# Route for the video feed
@app.route('/video_feed')
def video_feed():
    if 'username' in session:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return redirect(url_for('login'))

if __name__ == '__main__':
    try:
        port = int(os.environ.get("PORT", 5000))  # Get Render's assigned port
        app.run(host="0.0.0.0", port=port, debug=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()

