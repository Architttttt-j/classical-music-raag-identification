import os
print("NEW VERSION LOADED")

import numpy as np
import librosa
import scipy.stats
import joblib

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'uploads'
)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024

ALLOWED_EXTENSIONS = {'mp3', 'wav'}

# Create uploads folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
try:
    yaman_model = joblib.load("yaman_combined.pkl")
    bhoopali_model = joblib.load("bhoopali_combined.pkl")
    bhairavi_model = joblib.load("bhairavi_combined.pkl")

    print("HMM Models loaded successfully.")

except Exception as e:

    print(f"Error loading models: {e}")

    yaman_model = None
    bhoopali_model = None
    bhairavi_model = None


# Metadata
RAAG_METADATA = {

    "Yaman": {
        "tagline": "The evening scale of twilight serenity.",
        "thaat": "Kalyan",
        "prahar": "1st Prahar of Night",
        "rasa": "Peace, Devotion",
        "vadi_samvadi": "Ga / Ni",
        "aaroh": "N R G M# P D N S'",
        "avroha": "S' N D P M# G R S",
        "description": "Yaman is a late evening Raag."
    },

    "Bhoopali": {
        "tagline": "The pentatonic harmony of early night.",
        "thaat": "Kalyan",
        "prahar": "Night",
        "rasa": "Peace, Devotion",
        "vadi_samvadi": "Ga / Dha",
        "aaroh": "S R G P D S'",
        "avroha": "S' D P G R S",
        "description": "Bhoopali is a pentatonic Raag."
    },

    "Bhairavi": {
        "tagline": "The soulful morning Raag.",
        "thaat": "Bhairavi",
        "prahar": "Morning",
        "rasa": "Bhakti, Karuna",
        "vadi_samvadi": "Ma / Sa",
        "aaroh": "S r g M P d n S'",
        "avroha": "S' n d P M g r S",
        "description": "Bhairavi uses komal swaras."
    }
}


def allowed_file(filename):

    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def process_audio(file_path):

    y, sr = librosa.load(file_path)

    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C5')
    )

    valid_f0 = f0[~np.isnan(f0)]

    if len(valid_f0) == 0:
        raise ValueError(
            "Could not detect clear pitch."
        )

    tonic_sa = float(
        scipy.stats.mode(
            np.round(valid_f0),
            keepdims=True
        ).mode[0]
    )

    swaras = librosa.hz_to_svara_h(
        valid_f0,
        Sa=tonic_sa
    )

    allowed = ['S', 'R', 'G', 'M', 'P', 'D', 'N']

    clean = []

    for s in swaras:

        if s == 'M#':
            clean.append('M')

        elif s in allowed:
            clean.append(s)

    compressed = []

    if clean:

        compressed.append(clean[0])

        for i in range(1, len(clean)):

            if clean[i] != clean[i - 1]:
                compressed.append(clean[i])

    if len(compressed) < 5:
        raise ValueError(
            "Audio too short or unclear."
        )

    return compressed


@app.route('/')
def index():

    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:

        return jsonify({
            "error": "No file uploaded"
        }), 400

    file = request.files['file']

    if file.filename == '':

        return jsonify({
            "error": "No file selected"
        }), 400

    if not allowed_file(file.filename):

        return jsonify({
            "error": "Only MP3 and WAV allowed"
        }), 400

    if (
        yaman_model is None or
        bhoopali_model is None or
        bhairavi_model is None
    ):

        return jsonify({
            "error": "Models not loaded"
        }), 500

    try:

        filename = secure_filename(file.filename)

        file_path = os.path.join(
            app.config['UPLOAD_FOLDER'],
            filename
        )

        file.save(file_path)

        try:

            # Extract swaras
            swaras = process_audio(file_path)
            print("\n====================")
            print("Extracted Swaras:")
            print(" ".join([str(s) for s in swaras]))
            print("====================")

            # Common mapping
            swara_map = {
                'S': 0,
                'R': 1,
                'G': 2,
                'M': 3,
                'P': 4,
                'D': 5,
                'N': 6
            }

            numeric_seq = np.array(
                [
                    swara_map[s]
                    for s in swaras
                    if s in swara_map
                ]
            ).reshape(-1, 1)

            if len(numeric_seq) < 5:

                raise ValueError(
                    "Audio too short."
                )

            # Safe score calculations
            try:
                yaman_score = float(
                    yaman_model.score(numeric_seq)
                ) / len(numeric_seq)

                if not np.isfinite(yaman_score):
                    yaman_score = -9999

            except:
                yaman_score = -9999

            try:
                bhoopali_score = float(
                    bhoopali_model.score(numeric_seq)
                ) / len(numeric_seq)

                if not np.isfinite(bhoopali_score):
                    bhoopali_score = -9999

            except:
                bhoopali_score = -9999

            try:
                bhairavi_score = float(
                    bhairavi_model.score(numeric_seq)
                ) / len(numeric_seq)

                if not np.isfinite(bhairavi_score):
                    bhairavi_score = -9999

            except:
                bhairavi_score = -9999

            print("Yaman Score:", yaman_score)
            print("Bhoopali Score:", bhoopali_score)
            print("Bhairavi Score:", bhairavi_score)

            scores = {
                "Yaman": yaman_score,
                "Bhoopali": bhoopali_score,
                "Bhairavi": bhairavi_score
            }

            predicted_raag = max(
                scores,
                key=scores.get
            )

            metadata = RAAG_METADATA[predicted_raag]

            response_data = {
                "raag": predicted_raag,
                "scores": {
                    "yaman": yaman_score,
                    "bhoopali": bhoopali_score,
                    "bhairavi": bhairavi_score
                },
                "metadata": metadata
            }

            return jsonify(response_data)

        finally:

            if os.path.exists(file_path):
                os.remove(file_path)

    except ValueError as e:

        return jsonify({
            "error": str(e)
        }), 422

    except Exception as e:

        print(f"Error: {e}")

        return jsonify({
            "error": "Internal server error"
        }), 500


if __name__ == '__main__':

    app.run(
        debug=True,
        port=5000
    )