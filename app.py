from flask import Flask, request, render_template
import os
import uuid
import io
import sys
from werkzeug.utils import secure_filename
from pipeline import run_pipeline

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = None
    output_logs = None
    error = None

    if request.method == 'POST':
        filepath = None
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, str(uuid.uuid4()) + "_" + filename)
            file.save(filepath)
        elif 'filepath' in request.form and request.form['filepath'] != '':
            filepath = request.form['filepath']
            if not os.path.exists(filepath):
                error = "Файл за вказаним шляхом не існує."
                return render_template('index.html', predictions=predictions, error=error, output_logs=output_logs)
        else:
            error = "Не обрано файл і не вказано шлях."
            return render_template('index.html', predictions=predictions, error=error, output_logs=output_logs)
        try:
            old_stdout = sys.stdout
            sys.stdout = mystdout = io.StringIO()
            predictions = run_pipeline(filepath)
            output_logs = mystdout.getvalue()
            if not predictions:
                error = "Не вдалося визначити предикти."
        finally:
            sys.stdout = old_stdout
            if 'uploads' in filepath:
                os.remove(filepath)
    return render_template('index.html', predictions=predictions, error=error, output_logs=output_logs)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
