import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from base64 import b64decode
from PIL import Image, ImageEnhance, ImageFilter
import io
import imageio
from io import BytesIO
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)
CORS(app)

ASCII_CHARS = r'$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,"^`\'. '

def process_image(image, brightness=1.0, contrast=1.0, sharpness=1.0):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pil_image = ImageEnhance.Brightness(pil_image).enhance(brightness)
    pil_image = ImageEnhance.Contrast(pil_image).enhance(contrast)
    pil_image = ImageEnhance.Sharpness(pil_image).enhance(sharpness)
    pil_image = pil_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def image_to_ascii(image, width=200, brightness=1.0, contrast=1.0, sharpness=1.0, invert=False):
    image = process_image(image, brightness, contrast, sharpness)
    height = int(image.shape[0] * width / image.shape[1])
    image = cv2.resize(image, (width, height))
    
    if invert:
        image = 255 - image
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = (image > 128) * 255
    
    ascii_image = []
    for i in range(0, image.shape[0] - 1, 2):
        ascii_row = []
        for j in range(image.shape[1]):
            top_pixel = image[i, j]
            bottom_pixel = image[i+1, j]
            char = get_detailed_ascii_char(top_pixel, bottom_pixel)
            ascii_row.append(char)
        ascii_image.append(ascii_row)
    
    return '\n'.join([''.join(row) for row in ascii_image])

def get_detailed_ascii_char(top_pixel, bottom_pixel):
    if top_pixel == 255 and bottom_pixel == 255:
        return ' '
    elif top_pixel == 255 and bottom_pixel == 0:
        return '_'
    elif top_pixel == 0 and bottom_pixel == 255:
        return '¯'
    else:
        return '▄'

def process_frame(frame, width, brightness, contrast, sharpness, invert):
    return image_to_ascii(frame, width, brightness, contrast, sharpness, invert)

def process_input(input_data, width, brightness, contrast, sharpness, invert):
    try:
        gif = imageio.mimread(io.BytesIO(input_data))
        frames = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in gif]
        return frames
    except:
        try:
            temp_file = 'temp_video.mp4'
            with open(temp_file, 'wb') as f:
                f.write(input_data)
            cap = cv2.VideoCapture(temp_file)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            import os
            os.remove(temp_file)
            return frames
        except:
            nparr = np.frombuffer(input_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return [frame]

def process_frames_in_batches(frames, width, brightness, contrast, sharpness, invert, batch_size=10):
    ascii_frames = []
    
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            futures = [executor.submit(process_frame, frame, width, brightness, contrast, sharpness, invert) for frame in batch]
            ascii_frames.extend([future.result() for future in as_completed(futures)])
    
    return ascii_frames

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/convert', methods=['POST'])
def convert_input():
    data = request.get_json()
    input_data = data['input']
    width = data.get('width', 400)
    brightness = data.get('brightness', 1.0)
    contrast = data.get('contrast', 1.2)
    sharpness = data.get('sharpness', 1.5)
    invert = data.get('invert', False)
    
    input_data = b64decode(input_data.split(',')[1])
    
    frames = process_input(input_data, width, brightness, contrast, sharpness, invert)
    
    ascii_frames = process_frames_in_batches(frames, width, brightness, contrast, sharpness, invert)
    
    return jsonify({'ascii_art': ascii_frames})

if __name__ == '__main__':
    app.run(debug=True)