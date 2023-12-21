from flask import Flask, render_template, request, redirect, url_for ,send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import uuid
import csv
from keras_facenet import FaceNet
import json
from flask import jsonify
import ast
import base64


app = Flask(__name__)
embedder = FaceNet()

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'

# Define directories
TEMP_FOLDER = 'temp_faces'

# # Set absolute paths for directories
# app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, UPLOAD_FOLDER)
app.config['TEMP_FOLDER'] = os.path.join(app.root_path, TEMP_FOLDER)




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/click')
def click():
    return render_template('camera_click.html')

TEMP_STORAGE_PATH = 'temp_storage'  # Replace with your desired temporary storage folder
os.makedirs(TEMP_STORAGE_PATH, exist_ok=True)




def detect_faces(image_path, detections):
    image = cv2.imread(image_path)
    for detection in detections:
        if detection['box']:
            startX, startY, w, h = detection['box']
            endX, endY = startX + w, startY + h
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 4)

    processed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_image.jpg')
    cv2.imwrite(processed_image_path, image)
    return processed_image_path

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        detections = embedder.extract(file_path, threshold=0.95)
        save_embeddings(image_name  = filename, 
                        detections  = detections,
                        csv_file    ='embeddings.csv')
        num_faces = len(detections)
        impr_features = ['box', 'confidence']  # Keys to be selected

        filtered_list = [{key: value for key, value in d.items() if key in impr_features} for d in detections]
        result = {
                'num_faces': num_faces,
                'detections': filtered_list }

        detections_json = json.dumps(result)
        processed_image_path = detect_faces(file_path, detections)

        return render_template('processed_image.html', processed_image=processed_image_path, detections=detections_json)
    
@app.route('/image_search')
def image_search():
    return render_template('image_search.html')

@app.route('/process_photo', methods=['POST'])
def process_photo():
    photo_data = request.json.get('photo')

    # Save the photo data to a temporary file (you may need to handle filenames uniquely)
    filename = 'captured_photo.jpg'
    photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(photo_path, 'wb') as file:
        file.write(base64.b64decode(photo_data.split(',')[1]))

    result = process_single_face(photo_path)  # Call the function with the photo path

    # You can return a response if needed
    return jsonify(result)

@app.route('/process_single_face', methods=['POST'])
def process_single_face():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    if file:
        print('file',file)
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print('file-path --',file_path)
        detections = embedder.extract(file_path, threshold=0.95)
        
        x = find_similar_faces(detections[0]['embedding'])

        
        num_faces = len(detections)
        impr_features = ['box', 'confidence']  # Keys to be selected

        filtered_list = [{key: value for key, value in d.items() if key in impr_features} for d in detections]
        detections_json = {
                'num_faces' : num_faces,
                'detections': filtered_list,
                'top 5 images' : x['selected_images'] ,
            }

        # Extract detections from JSON data
        detections = detections_json.get('detections', [])

        # Create a folder to temporarily store cropped faces
        temp_folder = os.path.join(app.config['TEMP_FOLDER'], str(uuid.uuid4()))
        os.makedirs(temp_folder)

        # Crop faces from the image using provided coordinates and save them temporarily
        cropped_faces_paths = []
        image = cv2.imread(file_path)
        for i, detection in enumerate(detections):
            box = detection.get('box', [])
            if len(box) == 4:  # Check if the box has four coordinates (x, y, w, h)
                startX, startY, w, h = box
                endX, endY = startX + w, startY + h
                face = image[startY:endY, startX:endX].copy()
                face_path = os.path.join(temp_folder, f"cropped_face_{i + 1}.jpg")
                cv2.imwrite(face_path, face)
                cropped_faces_paths.append(face_path)

        # Return paths of the cropped faces to the frontend
        return jsonify(detections_json)

def euclidean_distance(vec1, vec2):
    euclidean_distance = np.linalg.norm(vec1 - vec2)
    return euclidean_distance

def find_similar_faces(detected_face_embedding):
    # Load embeddings from the CSV file
    embeddings = []
    filenames = []
    with open('embeddings.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            filenames.append(row[0])
            # print('lenggth of row', len(row))
            string_formatted_array = row[3]
            array_values = string_formatted_array.replace('[', '').replace(']', '').split()
            converted_array = np.array([float(value) for value in array_values])
            embedding = np.array(converted_array, dtype=float)  # Assuming the embeddings start from index 1 in each row

            embeddings.append(embedding)

    # Calculate distances between the detected face's embedding and stored embeddings
    distances = []
    for i, stored_embedding in enumerate(embeddings):
        distance = euclidean_distance(detected_face_embedding, stored_embedding)
        if distance < 0.6:
            distances.append((filenames[i], float(distance)))

    # Sort the distances
    distances.sort(key=lambda x: x[1])

    dict_output = { 'selected_images' :  distances[:8] }
    return dict_output

@app.route('/temp_faces/<path:filename>')
def serve_cropped_faces(filename):
    return send_from_directory(app.config['TEMP_FOLDER'], filename)



@app.route('/processed_image/<path:processed_image>')
def show_processed_image(processed_image):
    return render_template('processed_image.html', processed_image=processed_image)


# Function to save face detections and embeddings to a CSV file
def save_embeddings(image_name, detections, csv_file):
    # Check if the image name already exists in the CSV file
    # with open(csv_file, 'r', newline='') as file:
    #     reader = csv.DictReader(file)
    #     for row in reader:
    #         if row[0] == image_name:
    #             return "Duplicate entry found. Image name already exists in the CSV file."

    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        
        for detection in detections:
            box = detection.get('box', [])
            confidence = detection.get('confidence', 0)
            embedding = detection.get('embedding', [])
            
            
            writer.writerow([image_name, box, confidence, embedding])
    return None




def read_csv_file(file_path):
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        data = [row for row in reader]
    return data

@app.route('/show_csv')
def show_csv():
    csv_file_path = 'embeddings.csv'
    csv_data = read_csv_file(csv_file_path)
    
    return render_template('csv_display.html', csv_data=csv_data)


@app.route('/gallery')
def image_gallery():
    images_folder = 'uploads'  # Change this to the folder containing your images
    images_folder = 'sigma_analytics_image_flix/EV001'
    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    return render_template('gallery.html', image_files=image_files)

@app.route('/display_image/<filename>')
def display_image(filename):
    images_folder = 'uploads'  # Change this to the folder containing your images
    return send_from_directory(images_folder, filename)

if __name__ == '__main__':
    app.run(debug=True)
