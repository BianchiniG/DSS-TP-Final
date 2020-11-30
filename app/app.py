from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

camera = cv2.VideoCapture('/dev/video0')
if not camera.isOpened():
    camera.open(0)


def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            app.logger.error("No se pudo leer de la cámara")
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            # Llamada a las predicciones
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', threaded=True)
