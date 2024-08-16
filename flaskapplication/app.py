from flask import Flask, request, render_template, jsonify
import speech_recognition as sr
import os
import ffmpeg

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/convert_audio', methods=['POST'])
def convert_audio():
    if 'audio_data' not in request.files:
        return jsonify({'error': 'No audio file uploaded'})

    audio_file = request.files['audio_data']
    input_file_path = os.path.join(UPLOAD_FOLDER, 'audio.webm')
    output_file_path = os.path.join(UPLOAD_FOLDER, 'audio.wav')
    audio_file.save(input_file_path)

    try:
        # Convert audio file to PCM WAV format using ffmpeg
        ffmpeg.input(input_file_path).output(output_file_path, format='wav').run()
    except ffmpeg.Error as e:
        return jsonify({'error': str(e)})

    recognizer = sr.Recognizer()
    with sr.AudioFile(output_file_path) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        text = "Could not understand audio"
    except sr.RequestError as e:
        text = f"Could not request results; {e}"

    # Optionally delete the files after processing
    os.remove(input_file_path)
    os.remove(output_file_path)

    return jsonify({'text': text})

if __name__ == '__main__':
    app.run(debug=True)
