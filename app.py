from flask import Flask, render_template, request, redirect, url_for
from ocr import perform_ocr
from ner import perform_ner, translate_text
from htr import perform_htr
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']
        text_type = request.form['text_type']
        translate = 'translate' in request.form

        if file.filename == '':
            return redirect(request.url)

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            if text_type == 'ocr':
                text = perform_ocr(filepath)
            else:
                _, text = perform_htr(filepath)

            if translate:
                text = translate_text(text)

            annotated_text = perform_ner(text)

            image_url = url_for('static', filename=f'uploads/{file.filename}')

            return redirect(url_for('results', image_path=image_url, extracted_text=annotated_text, text_type=text_type, translate=translate))

    return render_template('index.html')

@app.route('/results')
def results():
    image_path = request.args.get('image_path')
    extracted_text = request.args.get('extracted_text')
    text_type = request.args.get('text_type', 'ocr')
    translate = request.args.get('translate', 'False') == 'True'

    if not image_path or not extracted_text:
        return redirect(url_for('index'))

    return render_template('results.html', image_path=image_path, extracted_text=extracted_text, text_type=text_type, translate=translate)


@app.route('/ner_check', methods=['GET', 'POST'])
def ner_check():
    extracted_text = None
    translate = False
    if request.method == 'POST':
        text = request.form.get('text', '')
        translate = 'translate' in request.form
        if text:
            if translate:
                text = translate_text(text)
            extracted_text = perform_ner(text)

    return render_template('ner_check.html', extracted_text=extracted_text, translate=translate)

if __name__ == '__main__':
    app.run(debug=True)