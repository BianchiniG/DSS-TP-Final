import os
import cv2
import numpy as np
from time import time
from copy import copy
from flask import Flask, render_template, request, Response, jsonify, redirect, url_for
from Preprocesamiento import Preprocesamiento
from Reconocimiento import Reconocimiento
from modelos.utiles import clean_predictions_folder, \
    REALTIME_PREDICTION_OUT_PRIVATE_FOLDER, \
    REALTIME_PREDICTION_OUT_PUBLIC_FOLDER

app = Flask(__name__)

clean_predictions_folder()
cv2.ocl.setUseOpenCL(False)
camera = cv2.VideoCapture(-1)
preprocesamiento = Preprocesamiento()
reconocimiento = Reconocimiento()
frame = None


def gen_frames():
    global frame

    while True:
        success, frame = camera.read()
        if not success:
            app.logger.error("No se pudo leer de la cámara")
            frame = None
            break
        else:
            framed_image = copy(frame)
            framed_image = preprocesamiento.frame_detected_face(framed_image)
            ret, buffer = cv2.imencode('.jpg', framed_image)
            framed_image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + framed_image + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/visualizacion')
def visualizacion():
    modelo = request.args.get('modelo')
    return render_template('visualizacion.html', data=modelo)


@app.route('/prueba')
def prueba():
    return render_template('prueba.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/recon_results')
def get_recon_results():
    predicciones = None
    imagen = None

    # try:
    #     if frame is not None:
    #         predicciones = reconocimiento.ejecutar(frame)
    #
    #         if not os.path.exists(REALTIME_PREDICTION_OUT_PRIVATE_FOLDER):
    #             os.mkdir(REALTIME_PREDICTION_OUT_PRIVATE_FOLDER)
    #         filename = str(int(time()))+'.jpg'
    #         imagen_save_dir = REALTIME_PREDICTION_OUT_PRIVATE_FOLDER+filename
    #         imagen = REALTIME_PREDICTION_OUT_PUBLIC_FOLDER+filename
    #         cv2.imwrite(imagen_save_dir, frame)
    # except Exception:
    #     app.logger.error("Exception en el reconocimiento en tiempo real")

    return jsonify({
        'imagen': imagen,
        'predicciones': predicciones
    })


@app.route('/process_image', methods=['GET', 'POST'])
def process_image():
    image = request.files.get('file')

    npimg = np.fromfile(image, np.uint8)
    file = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    predicciones = reconocimiento.ejecutar(file)

    return jsonify(predicciones)


@app.route('/not_found')
def not_found():
    return render_template('404.html')


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', threaded=True)
