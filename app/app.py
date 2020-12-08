import numpy as np
import cv2
from flask import Flask, render_template, request, Response, jsonify, redirect, url_for
from Reconocimiento import Reconocimiento


app = Flask(__name__)

cv2.ocl.setUseOpenCL(False)
camera = cv2.VideoCapture(-1)
reconocimiento = Reconocimiento()


def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            app.logger.error("No se pudo leer de la cámara")
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            # reconocimiento.ejecutar(buffer)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


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
    return jsonify({})


@app.route('/result/:model')
def view_result(model):
    if model == 'cnn':
        return render_template('cnn_train_results.html')
    elif model == 'rf':
        return render_template('rf_train_results.html')
    elif model == 'svm':
        return render_template('svm_train_results.html')
    else:
        return render_template('not_found.html')


@app.route('/process_image', methods=['GET', 'POST'])
def process_image():
    image = request.files.get('file')

    npimg = np.fromfile(image, np.uint8)
    file = cv2.imdecode(npimg, cv2.COLOR_BGR2GRAY)

    predicciones = reconocimiento.ejecutar(file)

    return jsonify(predicciones)


@app.route('/not_found')
def not_found():
    return render_template('404.html')


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', threaded=True)
