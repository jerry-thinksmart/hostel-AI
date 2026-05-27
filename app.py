from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('room_model.pkl')
encoders = joblib.load('encoders.pkl')

@app.route('/')
def home():
    return "Hostel AI Running"

@app.route('/recommend', methods=['POST'])
def recommend():

    data = request.json

    encoded_student = [[

        encoders['temperature'].transform(
            [data['temperature']]
        )[0],

        encoders['cleaning'].transform(
            [data['cleaning']]
        )[0],

        encoders['disability'].transform(
            [data['disability']]
        )[0],

        encoders['cgpa_level'].transform(
            [data['cgpa_level']]
        )[0],

        encoders['floor_preference'].transform(
            [data['floor_preference']]
        )[0]
    ]]

    encoded_df = pd.DataFrame(
        encoded_student,
        columns=[
            'temperature',
            'cleaning',
            'disability',
            'cgpa_level',
            'floor_preference'
        ]
    )

    prediction = model.predict(encoded_df)

    room = encoders['assigned_room'].inverse_transform(
        prediction
    )

    return jsonify({
        "recommended_room": room[0]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)