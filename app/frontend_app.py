import cv2
import numpy as np
from copy import copy
from io import BytesIO
from flask import Flask, render_template, request, Response, jsonify, redirect, url_for
from Preprocesamiento import Preprocesamiento
from Reconocimiento import Reconocimiento
from modelos.utiles import clean_predictions_folder, \
    REALTIME_PREDICTION_OUT_PRIVATE_FOLDER, \
    REALTIME_PREDICTION_OUT_PUBLIC_FOLDER

frontend_app = Flask(__name__)

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
            frontend_app.logger.error("No se pudo leer de la cámara")
            frame = None
            break
        else:
            framed_image = copy(frame)
            framed_image = preprocesamiento.frame_detected_face(framed_image)
            ret, buffer = cv2.imencode('.jpg', framed_image)
            framed_image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + framed_image + b'\r\n')


@frontend_app.route('/')
def index():
    return render_template('index.html')


@frontend_app.route('/visualizacion')
def visualizacion():
    modelo = request.args.get('modelo')
    return render_template('visualizacion.html', data=modelo)


@frontend_app.route('/prueba')
def prueba():
    return render_template('prueba.html')


@frontend_app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@frontend_app.route('/get_frame')
def get_frame():
    image = copy(frame)
    np_bytes = BytesIO()
    np.save(np_bytes, image, allow_pickle=True)

    return jsonify({
        'frame': str(np_bytes.getvalue())
    })


@frontend_app.route('/not_found')
def not_found():
    return render_template('404.html')


if __name__ == '__main__':
    frontend_app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
