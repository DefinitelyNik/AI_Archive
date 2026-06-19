import json
import os
import uuid
import threading
from datetime import timedelta
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_session import Session
from flasgger import Swagger
from models import db, User, ProcessingResult
from ocr import perform_ocr
from tesseract_ocr import perform_tesseract_ocr
from htr import perform_htr
from ner import perform_ner, translate_text
from relations import extract_relations

app = Flask(__name__)

# Секретный ключ для сессий
app.secret_key = os.environ.get('SECRET_KEY', 'ai-archive-secret-key-change-in-production')

# Максимальный размер загружаемого файла (16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Абсолютный путь к базе данных
basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, 'archive.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Конфигурация сессий
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(basedir, 'flask_session')
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_KEY_PREFIX'] = 'ai_archive_'

# Папка для загрузок
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

# Инициализация расширений
db.init_app(app)
Session(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Пожалуйста, войдите в систему'

# Swagger конфигурация
template = {
    "swagger": "2.0",
    "info": {
        "title": "OCR + NER API",
        "description": "API для AI Archive",
        "version": "1.0.0"
    },
}
swagger = Swagger(app, template=template)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Создание таблиц базы данных
with app.app_context():
    db.create_all()
    print(f"Database initialized at: {db_path}")
    print(f"Session storage: {app.config['SESSION_FILE_DIR']}")


# ============ Authentication Routes ============

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember') == 'on'

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user, remember=remember)
            if remember:
                session.permanent = True
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        flash('Неверное имя пользователя или пароль', 'danger')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if User.query.filter_by(username=username).first():
            flash('Пользователь с таким именем уже существует', 'danger')
            return redirect(url_for('register'))

        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        login_user(user, remember=True)
        session.permanent = True
        return redirect(url_for('index'))

    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# ============ Background Processing ============

def update_stage(result_id, stage, stage_data_update=None):
    """Helper to update processing stage in database."""
    with app.app_context():
        result = ProcessingResult.query.get(result_id)
        if result:
            result.current_stage = stage
            if stage_data_update:
                try:
                    current_data = json.loads(result.stage_data or '{}')
                    current_data.update(stage_data_update)
                    result.stage_data = json.dumps(current_data, ensure_ascii=False)
                except Exception:
                    result.stage_data = json.dumps(stage_data_update, ensure_ascii=False)
                db.session.commit()


def process_in_background(result_id, filepath, text_type, ocr_model, translate):
    """Background processing with stage-by-stage updates."""
    with app.app_context():
        result = ProcessingResult.query.get(result_id)
        try:
            # Stage 1: OCR/HTR Recognition
            update_stage(result_id, 'recognizing')

            if text_type == 'ocr':
                if ocr_model == 'tesseract':
                    text = perform_tesseract_ocr(filepath)
                else:
                    text = perform_ocr(filepath)
            else:
                _, text = perform_htr(filepath)

            result = ProcessingResult.query.get(result_id)
            result.original_text = text
            db.session.commit()

            update_stage(result_id, 'recognizing', {
                'recognizing': {'status': 'completed', 'text': text}
            })

            # Stage 2: Translation (optional)
            if translate:
                update_stage(result_id, 'translating')
                text = translate_text(text)
                update_stage(result_id, 'translating', {
                    'translating': {'status': 'completed', 'text': text}
                })

            # Stage 3: NER
            update_stage(result_id, 'ner')
            annotated_text_html = perform_ner(text)

            result = ProcessingResult.query.get(result_id)
            result.processed_text_html = annotated_text_html
            db.session.commit()

            update_stage(result_id, 'ner', {
                'ner': {'status': 'completed', 'html': annotated_text_html}
            })

            # Stage 4: Relations
            update_stage(result_id, 'relations')
            relations = extract_relations(text)
            relations_json = json.dumps(relations, ensure_ascii=False, indent=2)

            result = ProcessingResult.query.get(result_id)
            result.relations_json = relations_json
            db.session.commit()

            update_stage(result_id, 'relations', {
                'relations': {'status': 'completed', 'json': relations_json, 'count': len(relations)}
            })

            # Final: Completed
            result = ProcessingResult.query.get(result_id)
            result.current_stage = 'completed'
            result.status = 'completed'
            db.session.commit()

        except Exception as e:
            import traceback
            traceback.print_exc()
            result = ProcessingResult.query.get(result_id)
            if result:
                result.current_stage = 'failed'
                result.status = 'failed'
                result.error_message = str(e)
                db.session.commit()


# ============ Main Routes ============

@app.route('/', methods=['GET'])
@login_required
def index():
    """Main page with split-view layout."""
    return render_template('index.html')


@app.route('/process', methods=['POST'])
@login_required
def process():
    """Start background processing and return immediately."""
    if 'image' not in request.files or 'text_type' not in request.form:
        return jsonify({'error': 'Missing required fields'}), 400

    file = request.files['image']
    text_type = request.form['text_type']
    ocr_model = request.form.get('ocr_model', 'easyocr')
    translate = 'translate' in request.form

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save file with unique name
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    # Create result record
    result = ProcessingResult(
        user_id=current_user.id,
        image_filename=unique_filename,
        text_type=text_type,
        ocr_model=ocr_model,
        translated=translate,
        status='processing',
        current_stage='queued',
        stage_data='{}'
    )
    db.session.add(result)
    db.session.commit()

    # Start background processing
    thread = threading.Thread(
        target=process_in_background,
        args=(result.id, filepath, text_type, ocr_model, translate),
        daemon=True
    )
    thread.start()

    return jsonify({
        'success': True,
        'result_id': result.id
    })


@app.route('/api/result/<int:result_id>/progress')
@login_required
def get_progress(result_id):
    """API endpoint for polling processing progress."""
    result = ProcessingResult.query.get_or_404(result_id)

    if result.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403

    stage_data = {}
    try:
        stage_data = json.loads(result.stage_data or '{}')
    except Exception:
        pass

    return jsonify({
        'current_stage': result.current_stage,
        'stage_data': stage_data,
        'status': result.status,
        'error': result.error_message,
        'image_filename': result.image_filename,
        'translated': result.translated
    })


@app.route('/result/<int:result_id>')
@login_required
def view_result(result_id):
    """View a specific processing result."""
    result = ProcessingResult.query.get_or_404(result_id)
    if result.user_id != current_user.id:
        flash('У вас нет доступа к этому результату', 'danger')
        return redirect(url_for('my_results'))
    return render_template('result_detail.html', result=result)


@app.route('/my_results')
@login_required
def my_results():
    """View all user's processing results."""
    results = ProcessingResult.query.filter_by(user_id=current_user.id) \
        .order_by(ProcessingResult.created_at.desc()).all()
    return render_template('my_results.html', results=results)


# ============ NER Check Routes ============

@app.route('/ner_check', methods=['GET', 'POST'])
@login_required
def ner_check():
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
    app.run(debug=True, threaded=True)
