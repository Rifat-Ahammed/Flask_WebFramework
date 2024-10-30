import cv2
import face_recognition
import numpy as np
from flask import Flask, Response, render_template

app = Flask(__name__, template_folder='templates_1')

# Load images and create face encodings
image_paths = ["Rifat/Rifat.jpg", "TomCruise/Tom_cruise.jpg"]
known_face_encodings = []
known_face_names = []

# Load each image and get its face encoding
for i, image_path in enumerate(image_paths):
    try:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:  # Check if at least one encoding is found
            known_face_encodings.append(encodings[0])
            known_face_names.append(f"Person {i + 1}")  # Customize names if needed
        else:
            print(f"No faces found in image: {image_path}")  # Debugging output
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")  # Debugging output

# Open the webcam (0 is usually the default camera)
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open video device.")  # Debugging output

def gen_frames():
    while True:
        success, frame = video_capture.read()
        if not success:
            print("Failed to capture frame")  # Debugging output
            break
        
        # Convert the image from BGR to RGB format
        rgb_frame = frame[:, :, ::-1]

        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through the faces found in the current frame
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match is found, use the known face's name
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Draw a box around the face and label it
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Could not encode frame")  # Debugging output
            continue
            
        frame = buffer.tobytes()

        # Yield the resulting frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('face_index.html')  # Make sure the correct template is referenced

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        video_capture.release()  # Ensure the capture is released on exit
