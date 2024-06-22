from flask import Flask, jsonify, request
import cv2
import os
from YOLO import computeCobb  # Assuming computeCobb is defined in YOLO.py

app = Flask(__name__)

# Load the Google Drive API key from the environment variable
google_drive_api_key = os.environ.get('AIzaSyD6ETpwgy0neL0jFJEWCBDyjV2Gih0hqcE')


@app.route('/compute-cobb', methods=['POST'])
def compute_cobb_api():
    try:
        file = request.files.get('image')
        if not file:
            return jsonify({'error': 'No image uploaded'}), 400

        # Save the uploaded image to a temporary file
        file.save('temp_image.jpg')

        # Read the uploaded image using OpenCV
        image = cv2.imread('temp_image.jpg')

        # Ensure the image is read correctly
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400

        # Perform Cobb angle computation
        cobb_up, cobb_low, img_cobb, result = computeCobb(image)

        # Check the results of the computation
        if cobb_up is None or cobb_low is None:
            return jsonify({'error': 'No vertebrae detected or wrong image'}), 400

        cobb_angle = max(abs(cobb_up), abs(cobb_low))

        # Return the results as JSON
        return jsonify({'cobb_angle': cobb_angle, 'classification': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 4000))
    app.run(host='0.0.0.0', port=port, debug=True)
