from flask import Flask, render_template, request, redirect, url_for
from flasgger import Swagger
from ocr import perform_ocr
from tesseract_ocr import perform_tesseract_ocr
from htr import perform_htr
from ner import perform_ner, translate_text
from relations import extract_relations
from text_cleanup import clean_text
import ast
import os
import uuid

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def save_upload(file):
    _, extension = os.path.splitext(file.filename)
    filename = f"{uuid.uuid4().hex}{extension.lower()}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return filename, filepath


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
        name: ocr_model
        type: string
        enum: [easyocr, tesseract]
        required: false
        description: Модель OCR (только для text_type=ocr)
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
        ocr_model = request.form.get('ocr_model', 'easyocr')
        translate = 'translate' in request.form
        cleanup_enabled = 'clean_text' in request.form

        if file.filename == '':
            return redirect(request.url)

        if file and text_type in ['ocr', 'htr']:
            filename, filepath = save_upload(file)

            if text_type == 'ocr':
                if ocr_model == 'tesseract':
                    text = perform_tesseract_ocr(filepath)
                else:  # easyocr (по умолчанию)
                    text = perform_ocr(filepath)
            else:  # htr
                _, text = perform_htr(filepath)

            if translate:
                text = translate_text(text)

            raw_text = text
            processed_text = clean_text(text) if cleanup_enabled else text

            annotated_text_html = perform_ner(processed_text)
            relations = extract_relations(processed_text)

            image_url = url_for('static', filename=f'uploads/{filename}')

            return render_template('results.html',
                                   image_path=image_url,
                                   extracted_text=annotated_text_html,
                                   text_type=text_type,
                                   ocr_model=ocr_model,
                                   translate=translate,
                                   cleanup_enabled=cleanup_enabled,
                                   raw_text=raw_text,
                                   cleaned_text=processed_text,
                                   relations=relations)

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
        name: ocr_model
        type: string
        enum: [easyocr, tesseract]
        required: false
        default: easyocr
        description: Использованная модель OCR
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
    ocr_model = request.args.get('ocr_model', 'easyocr')
    translate = request.args.get('translate', 'False') == 'True'
    cleanup_enabled = request.args.get('cleanup_enabled', 'False') == 'True'
    raw_text = request.args.get('raw_text')
    cleaned_text = request.args.get('cleaned_text')

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
                           ocr_model=ocr_model,
                           translate=translate,
                           cleanup_enabled=cleanup_enabled,
                           raw_text=raw_text,
                           cleaned_text=cleaned_text,
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
        name: ocr_model
        type: string
        enum: [easyocr, tesseract]
        required: false
        description: Модель OCR (только для text_type=ocr)
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
    ocr_model = request.form.get('ocr_model', 'easyocr')
    translate = 'translate' in request.form
    cleanup_enabled = 'clean_text' in request.form

    if file.filename == '':
        return {'error': 'No file selected'}, 400

    if text_type not in ['ocr', 'htr']:
        return {'error': 'Invalid text_type'}, 400

    _, filepath = save_upload(file)

    if text_type == 'ocr':
        if ocr_model == 'tesseract':
            text = perform_tesseract_ocr(filepath)
        else:  # easyocr (по умолчанию)
            text = perform_ocr(filepath)
    else:
        _, text = perform_htr(filepath)

    if translate:
        text = translate_text(text)

    raw_text = text
    processed_text = clean_text(text) if cleanup_enabled else text

    annotated_text = perform_ner(processed_text)
    relations = extract_relations(processed_text)

    response = {
        'text': processed_text,
        'annotated_text': annotated_text,
        'relations': relations
    }

    if cleanup_enabled:
        response['raw_text'] = raw_text

    return response


@app.route('/api/clean-text', methods=['POST'])
def api_clean_text():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return {'error': 'JSON object is required'}, 400

    text = payload.get('text', '')

    if not text or not text.strip():
        return {'error': 'Text is required'}, 400

    return {'cleaned_text': clean_text(text)}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
