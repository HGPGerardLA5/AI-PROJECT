from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, make_response
import os
import openai
import requests
from PIL import Image
from io import BytesIO
from datetime import datetime
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import cv2  # Add this import at the top
import numpy as np

FACE_PROTO = "deploy.prototxt"
FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

load_dotenv()

print(f"[DEBUG] Loaded OpenAI Key: {os.getenv('OPENAI_API_KEY') is not None}")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

openai.api_key = os.getenv("OPENAI_API_KEY")

history = []

# Helper: Check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# DALL-E Image Generator
def generate_dalle_image(prompt):
    try:
        response = openai.images.generate(prompt=prompt, n=1, size="512x512")
        image_url = response.data[0].url
        img_data = requests.get(image_url).content
        image = Image.open(BytesIO(img_data))

        image = image.resize((900, 900))

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        output_filename = f"generated_{timestamp}.png"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        image.save(output_path)

        image_url_local = url_for('static', filename='outputs/' + output_filename)
        history.append({'type': 'generate', 'prompt': prompt, 'image_url': image_url_local})

        return output_filename
    except Exception as e:
        print(f"[ERROR] DALL-E generation failed: {e}")
        return None

# Dummy analyzer for uploaded images (replace with actual detection logic)
def analyze_uploaded_image(upload_path):
    image = cv2.imread(upload_path)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()

    print(f"[DEBUG] Detected {detections.shape[2]} objects")

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.3:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    image = cv2.resize(image, (900, 900))

    output_filename = f"analyzed_{os.path.basename(upload_path)}"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    cv2.imwrite(output_path, image)

    image_url_local = url_for('static', filename='outputs/' + output_filename)
    history.append({'type': 'upload', 'filename': os.path.basename(upload_path), 'image_url': image_url_local})

    return output_filename

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        tab = request.form.get('tab', 'generate')
        session['tab'] = tab

    generated_image_url = session.pop('latest_generated_image', None)
    uploaded_image_url = session.pop('latest_uploaded_image', None)

    if request.method == 'POST':
        print("[DEBUG] POST received")
        action = request.form.get('action')
        print(f"[DEBUG] Action received: {action}")

        if action == 'generate':
            prompt = request.form.get('prompt', '')
            output_filename = generate_dalle_image(prompt)
            if output_filename:
                session['latest_generated_image'] = url_for('static', filename='outputs/' + output_filename)
            tab = request.form.get('tab', 'generate')
            return redirect(url_for('index', tab=tab))

        elif action == 'upload':
            file = request.files.get('file')
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(upload_path)

                output_filename = analyze_uploaded_image(upload_path)
                if output_filename:
                    session['latest_uploaded_image'] = url_for('static', filename='outputs/' + output_filename)
            tab = request.form.get('tab', 'upload')
            return redirect(url_for('index', tab=tab))

    generate_history = [item for item in history if item["type"] == "generate"]
    upload_history = [item for item in history if item["type"] == "upload"]
    active_tab = request.args.get('tab') or session.get('tab') or request.cookies.get('tab') or 'generate'
    response = make_response(render_template(
        'combined.html',
        image_url=generated_image_url,
        upload_url=uploaded_image_url,
        generate_history=generate_history,
        upload_history=upload_history,
        active_tab=active_tab
    ))
    response.set_cookie('tab', active_tab)
    session['tab'] = active_tab
    return response

@app.route('/download/<filename>')
def download_image(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5050, debug=True)