from flask import Flask, render_template
import subprocess

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run')
def run_camera():
    subprocess.Popen(["python", "save.py"])
    return """
        <h2>🎥 Camera Program Started!</h2>
        <p>Please check your Python window — your webcam will open shortly.</p>
        <a href="/">⬅ Back to Home</a>
    """

if __name__ == '__main__':
    app.run(debug=True)
