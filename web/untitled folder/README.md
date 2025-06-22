# Garbage Classifier React Frontend

This is a React (Vite) frontend for uploading images and sending them to a Flask backend for garbage classification predictions.

## Usage

1. Start your Flask backend on port 8888:
   ```sh
   source ../.venv310/bin/activate
   python web/server.py
   ```
   (Make sure your backend is running on port 8888 and accessible from this frontend.)

2. Start the React frontend:
   ```sh
   npm run dev
   ```
   Then open the provided local URL in your browser.

3. Upload an image and view the prediction result.

## Configuration
- The backend API URL is set to `http://localhost:8888/predict` in `ImageUploader.jsx`. Change this if your backend runs elsewhere.
