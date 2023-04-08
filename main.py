import os

from distributed.http.utils import redirect
from flask import Flask, render_template, Response, url_for
from camera import VideoCamera

app = Flask(__name__)

@app.route('/')
def home():
    """Renders the home page with a Get Started button."""
    return render_template('home.html')

@app.route('/detect')
def detect():
    """Renders the detect page."""
    return render_template('detect.html')

@app.route('/start_detection')
def start_detect():
    """Redirects to the detect page when the Get Started button is clicked."""
    return redirect(url_for('detect'))



def gen(camera):
    """A generator function that yields the frames from the camera."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    """Returns the response object that contains the frames from the camera."""
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', debug=False)