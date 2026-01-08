from flask import Flask, render_template, request, redirect, url_for
from flasgger import Swagger
from ocr import perform_ocr
from htr import perform_htr
from ner import perform_ner, translate_text
from relations import extract_relations
import ast
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Настройка Flasgger
template = {
    "swagger": "2.0",
    "info": {
        "title": "OCR + NER API",
        "description": "API для AI Archive with ner, ocr and htr models",
        "version": "1.0.0"
    },
    "consumes": [
        "multipart/form-data"
    ],
    "produces": [
        "application/json"
    ],
}

swagger = Swagger(app, template=template)


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Загрузка изображения и выбор типа обработки
    ---
    tags:
      - Main
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: image
        type: file
        required: true
        description: Изображение для обработки
      - in: formData
        name: text_type
        type: string
        enum: [ocr, htr]
        required: true
        description: Тип текста (машинный или рукописный)
      - in: formData
        name: translate
        type: boolean
        required: false
        description: Перевести текст с дореволюционного русского
    responses:
      302:
        description: Редирект на страницу результатов
    """
    if request.method == 'POST':
        if 'image' not in request.files or 'text_type' not in request.form:
            return redirect(request.url)

        file = request.files['image']
        text_type = request.form['text_type']
        translate = 'translate' in request.form

        if file.filename == '':
            return redirect(request.url)

        if file and text_type in ['ocr', 'htr']:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            if text_type == 'ocr':
                text = perform_ocr(filepath)
            else:  # htr
                _, text = perform_htr(filepath)

            if translate:
                text = translate_text(text)

            annotated_text_html = perform_ner(text)
            relations = extract_relations(text)

            image_url = url_for('static', filename=f'uploads/{file.filename}')

            return redirect(
                url_for('results',
                        image_path=image_url,
                        extracted_text=annotated_text_html,
                        text_type=text_type,
                        translate=translate,
                        relations=relations))

    return render_template('index.html')


@app.route('/results')
def results():
    """
    Отображение результатов обработки
    ---
    tags:
      - Results
    parameters:
      - in: query
        name: image_path
        type: string
        required: true
        description: Путь к изображению
      - in: query
        name: extracted_text
        type: string
        required: true
        description: Извлечённый и аннотированный текст
      - in: query
        name: text_type
        type: string
        enum: [ocr, htr]
        required: false
        default: ocr
        description: Тип текста (OCR или HTR)
      - in: query
        name: translate
        type: boolean
        required: false
        default: false
        description: Был ли применён перевод
      - in: query
        name: relations
        type: string
        required: false
        description: Извлечённые отношения
    responses:
      200:
        description: Страница с результатами
    """
    image_path = request.args.get('image_path')
    extracted_text = request.args.get('extracted_text')
    text_type = request.args.get('text_type', 'ocr')
    translate = request.args.get('translate', 'False') == 'True'

    relations_str = request.args.get('relations', '[]')
    try:
        relations = ast.literal_eval(relations_str)
    except (ValueError, SyntaxError):
        relations = []

    if not image_path or not extracted_text:
        return redirect(url_for('index'))

    return render_template('results.html',
                           image_path=image_path,
                           extracted_text=extracted_text,
                           text_type=text_type,
                           translate=translate,
                           relations=relations)


@app.route('/ner_check', methods=['GET', 'POST'])
def ner_check():
    """
    Проверка NER-модели на тексте
    ---
    tags:
      - NER Check
    parameters:
      - in: formData
        name: text
        type: string
        required: false
        description: Текст для анализа
      - in: formData
        name: translate
        type: boolean
        required: false
        description: Перевести текст с дореволюционного русского
    responses:
      200:
        description: Страница с результатами NER
    """
    extracted_text = None
    relations = []
    translate = False
    if request.method == 'POST':
        text = request.form.get('text', '')
        translate = 'translate' in request.form
        if text:
            if translate:
                text = translate_text(text)
            extracted_text = perform_ner(text)
            relations = extract_relations(text)

    return render_template('ner_check.html',
                           extracted_text=extracted_text,
                           relations=relations,
                           translate=translate)


@app.route('/api/process', methods=['POST'])
def api_process():
    """
    API-эндпоинт для обработки изображений
    ---
    tags:
      - API
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: image
        type: file
        required: true
        description: Изображение для обработки
      - in: formData
        name: text_type
        type: string
        enum: [ocr, htr]
        required: true
        description: Тип текста (машинный или рукописный)
      - in: formData
        name: translate
        type: boolean
        required: false
        description: Перевести текст с дореволюционного русского
    responses:
      200:
        description: Успешная обработка
        schema:
          type: object
          properties:
            text:
              type: string
              description: Извлечённый текст
            annotated_text:
              type: string
              description: Текст с NER-тегами
            relations:
              type: array
              items:
                type: array
                description: Список кортежей (сущность1, отношение, сущность2)
      400:
        description: Ошибка ввода
    """
    if 'image' not in request.files or 'text_type' not in request.form:
        return {'error': 'Missing required fields'}, 400

    file = request.files['image']
    text_type = request.form['text_type']
    translate = 'translate' in request.form

    if file.filename == '':
        return {'error': 'No file selected'}, 400

    if text_type not in ['ocr', 'htr']:
        return {'error': 'Invalid text_type'}, 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    if text_type == 'ocr':
        text = perform_ocr(filepath)
    else:
        _, text = perform_htr(filepath)

    if translate:
        text = translate_text(text)

    annotated_text = perform_ner(text)
    relations = extract_relations(text)

    return {
        'text': text,
        'annotated_text': annotated_text,
        'relations': relations
    }

@app.route('/test_page')
def test_page():
    """
    Тестовая страница для разработки и расширения тестового покрытия
    """
    return render_template('test_page.html')

if __name__ == '__main__':
    app.run(debug=True)
