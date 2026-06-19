import json
import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flasgger import Swagger
from models import db, User, ProcessingResult
from ocr import perform_ocr
from tesseract_ocr import perform_tesseract_ocr
from htr import perform_htr
from ner import perform_ner, translate_text
from relations import extract_relations

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///archive.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Пожалуйста, войдите в систему'

template = {
    "swagger": "2.0",
    "info": {
        "title": "OCR + NER API",
        "description": "API для AI Archive with ner, ocr and htr models",
        "version": "1.0.0"
    },
}
swagger = Swagger(app, template=template)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


with app.app_context():
    db.create_all()


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Неверное имя пользователя или пароль')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if User.query.filter_by(username=username).first():
            flash('Пользователь с таким именем уже существует')
            return redirect(url_for('register'))

        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        login_user(user)
        return redirect(url_for('index'))

    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    """Logout user."""
    logout_user()
    return redirect(url_for('login'))


@app.route('/', methods=['GET'])
@login_required
def index():
    """Main page with upload form."""
    return render_template('index.html')


@app.route('/process', methods=['POST'])
@login_required
def process():
    """
    Process image asynchronously and save result.
    Returns immediately with task ID.
    """
    if 'image' not in request.files or 'text_type' not in request.form:
        return jsonify({'error': 'Missing required fields'}), 400

    file = request.files['image']
    text_type = request.form['text_type']
    ocr_model = request.form.get('ocr_model', 'easyocr')
    translate = 'translate' in request.form

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    result = ProcessingResult(
        user_id=current_user.id,
        image_filename=file.filename,
        text_type=text_type,
        ocr_model=ocr_model,
        translated=translate,
        status='processing'
    )
    db.session.add(result)
    db.session.commit()

    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    try:
        if text_type == 'ocr':
            if ocr_model == 'tesseract':
                text = perform_tesseract_ocr(filepath)
            else:
                text = perform_ocr(filepath)
        else:
            _, text = perform_htr(filepath)

        result.original_text = text

        if translate:
            text = translate_text(text)

        annotated_text_html = perform_ner(text)
        result.processed_text_html = annotated_text_html

        relations = extract_relations(text)
        result.relations_json = json.dumps(relations, ensure_ascii=False, indent=2)

        result.status = 'completed'
        db.session.commit()

        return jsonify({
            'success': True,
            'result_id': result.id,
            'redirect_url': url_for('view_result', result_id=result.id)
        })

    except Exception as e:
        result.status = 'failed'
        result.error_message = str(e)
        db.session.commit()

        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/result/<int:result_id>')
@login_required
def view_result(result_id):
    """View a specific processing result."""
    result = ProcessingResult.query.get_or_404(result_id)

    if result.user_id != current_user.id:
        flash('У вас нет доступа к этому результату')
        return redirect(url_for('my_results'))

    return render_template('result_detail.html', result=result)


@app.route('/my_results')
@login_required
def my_results():
    """View all user's processing results."""
    results = ProcessingResult.query.filter_by(user_id=current_user.id) \
        .order_by(ProcessingResult.created_at.desc()).all()

    return render_template('my_results.html', results=results)


@app.route('/api/result/<int:result_id>/status')
@login_required
def get_result_status(result_id):
    """API endpoint to check result status (for AJAX polling)."""
    result = ProcessingResult.query.get_or_404(result_id)

    if result.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403

    return jsonify({
        'status': result.status,
        'result_id': result.id
    })


@app.route('/ner_check', methods=['GET', 'POST'])
@login_required
def ner_check():
    """NER check page."""
    extracted_text = None
    relations = []
    relations_json = '[]'
    translate = False

    if request.method == 'POST':
        text = request.form.get('text', '')
        translate = 'translate' in request.form

        if text:
            if translate:
                text = translate_text(text)

            extracted_text = perform_ner(text)
            relations = extract_relations(text)
            relations_json = json.dumps(relations, ensure_ascii=False, indent=2)

    return render_template('ner_check.html',
                           extracted_text=extracted_text,
                           relations=relations,
                           relations_json=relations_json,
                           translate=translate)


if __name__ == '__main__':
    app.run(debug=True)
