import os
import cv2
import ast
import requests
import numpy as np
from time import time
from io import BytesIO
from flask import Flask, render_template, request, Response, jsonify, redirect, url_for
from flask_cors import CORS, cross_origin
from Preprocesamiento import Preprocesamiento
from Reconocimiento import Reconocimiento
from modelos.utiles import clean_predictions_folder, \
    REALTIME_PREDICTION_OUT_PRIVATE_FOLDER, \
    REALTIME_PREDICTION_OUT_PUBLIC_FOLDER

backend_app = Flask(__name__)
cors = CORS(backend_app)
backend_app.config['CORS_HEADERS'] = 'Content-Type'
preprocesamiento = Preprocesamiento()
reconocimiento = Reconocimiento()


@backend_app.route('/recon_results')
@cross_origin()
def get_recon_results():
    predicciones = None
    imagen = None
    response = requests.get('http://172.30.0.2:5000/get_frame')
    frame_bytes = BytesIO(ast.literal_eval(response.json()['frame']))
    frame = np.load(frame_bytes, allow_pickle=True)

    try:
        if frame is not None:
            predicciones = reconocimiento.ejecutar(frame)

            if not os.path.exists(REALTIME_PREDICTION_OUT_PRIVATE_FOLDER):
                os.mkdir(REALTIME_PREDICTION_OUT_PRIVATE_FOLDER)
            filename = str(int(time()))+'.jpg'
            imagen_save_dir = REALTIME_PREDICTION_OUT_PRIVATE_FOLDER+filename
            imagen = REALTIME_PREDICTION_OUT_PUBLIC_FOLDER+filename
            cv2.imwrite(imagen_save_dir, frame)
    except Exception as e:
        backend_app.logger.error(e)
        backend_app.logger.error("Error reconociendo la foto")

    return jsonify({
        'imagen': imagen,
        'predicciones': predicciones
    })


@backend_app.route('/process_image', methods=['GET', 'POST'])
@cross_origin()
def process_image():
    image = request.files.get('file')

    npimg = np.fromfile(image, np.uint8)
    file = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    predicciones = reconocimiento.ejecutar(file)

    return jsonify(predicciones)


if __name__ == '__main__':
    backend_app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
