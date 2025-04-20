import numpy as np
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from sklearn.preprocessing import StandardScaler
import io
import base64

app = Flask(__name__)

model = model = tf.keras.models.load_model('models/model02.h5')

def load_ecg_csv(csv_ecg):
  ecg_data = np.loadtxt(csv_ecg, delimiter=",", skiprows=1)
  Y_scaler = StandardScaler()
  Y_scaler.fit(ecg_data.reshape(-1, ecg_data.shape[-1]))
  ecg_data = Y_scaler.transform(ecg_data.reshape(-1, ecg_data.shape[-1])).reshape(ecg_data.shape)

  return ecg_data


import numpy as np
import matplotlib.pyplot as plt
import io
import base64

def plot_ecg(data, fs=500):
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle('12-Lead ECG', fontsize=16, fontweight='bold')

    time = np.arange(data.shape[0]) / fs  # Convert samples to time (assuming fs=500 Hz)

    for i, lead in enumerate(leads):
        ax = axes[i // 2, i % 2]
        ax.plot(time, data[:, i], color='black', linewidth=1.2)
        ax.set_title(lead, fontsize=12, fontweight='bold', pad=5)
        
        # Add grid for reference
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6, color='red')
        
        # X-axis: Show only on bottom row
        if i // 2 == 5:
            ax.set_xlabel("Time (seconds)", fontsize=10)
        else:
            ax.set_xticks([])

        # Y-axis: Voltage scale
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(["-1mV", "0", "1mV"], fontsize=8)

        # Formatting spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')

    # Save as base64-encoded image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    encoded_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return encoded_image


@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        your_csv_ecg = load_ecg_csv(csv_ecg=file)
        img_buf = plot_ecg(your_csv_ecg)
        your_csv_ecg = your_csv_ecg.reshape(1, your_csv_ecg.shape[0], your_csv_ecg.shape[1])
        prediction = model.predict(your_csv_ecg).round().astype(int)[0]
        print(prediction)
        return render_template('main.html', prediction=prediction, ecg_image=img_buf)


if __name__ == '__main__':
    if not os.path.exists('temp'):
        os.makedirs('temp')
    app.run(debug=True)
