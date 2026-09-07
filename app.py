"""Flask application for archival OCR/HTR, NER and relation extraction."""

import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from urllib.parse import urlparse

from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from flask_login import LoginManager, current_user, login_required, login_user, logout_user
from flask_session import Session
from flasgger import Swagger
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from htr import perform_htr
from ml_worker import MLWorkerError, run_ml_operation
from models import ProcessingResult, User, db
from ner import perform_ner, translate_text
from ocr import perform_ocr
from relations import extract_relations
from tesseract_ocr import perform_tesseract_ocr

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "tif", "tiff", "bmp"}
MAX_IMAGE_PIXELS = int(os.environ.get("AI_ARCHIVE_MAX_IMAGE_PIXELS", "40000000"))
MAX_WORKERS = int(os.environ.get("AI_ARCHIVE_MAX_WORKERS", "1"))
KEEP_UPLOADS = os.environ.get("AI_ARCHIVE_KEEP_UPLOADS", "true").lower() == "true"
ML_TIMEOUTS = {
    "easyocr": int(os.environ.get("AI_ARCHIVE_OCR_TIMEOUT", "900")),
    "tesseract": int(os.environ.get("AI_ARCHIVE_OCR_TIMEOUT", "900")),
    "htr": int(os.environ.get("AI_ARCHIVE_HTR_TIMEOUT", "1200")),
    "ner": int(os.environ.get("AI_ARCHIVE_NER_TIMEOUT", "600")),
    "relations": int(os.environ.get("AI_ARCHIVE_RELATIONS_TIMEOUT", "1200")),
}

# One heavy operation at a time is intentional for a desktop CPU-only deployment.
# It prevents two simultaneous model loads from exhausting RAM or native runtimes.
processing_executor = ThreadPoolExecutor(
    max_workers=max(1, MAX_WORKERS), thread_name_prefix="ai-archive-job"
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "ai-archive-secret-key-change-in-production")
app.config.update(
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,
    SQLALCHEMY_DATABASE_URI=f"sqlite:///{os.path.join(BASE_DIR, 'archive.db')}",
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    SESSION_TYPE="filesystem",
    SESSION_FILE_DIR=os.path.join(BASE_DIR, "flask_session"),
    SESSION_PERMANENT=True,
    PERMANENT_SESSION_LIFETIME=timedelta(days=30),
    SESSION_USE_SIGNER=True,
    SESSION_KEY_PREFIX="ai_archive_",
    UPLOAD_FOLDER=UPLOAD_FOLDER,
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True)
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

if "SECRET_KEY" not in os.environ:
    logger.warning("SECRET_KEY is not set; configure it before exposing the application to a network.")

# Extensions
_db_path = app.config["SQLALCHEMY_DATABASE_URI"]
db.init_app(app)
Session(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message = "Пожалуйста, войдите в систему"

swagger = Swagger(
    app,
    template={
        "swagger": "2.0",
        "info": {"title": "OCR + NER API", "description": "API для AI Archive", "version": "1.0.0"},
    },
)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


with app.app_context():
    db.create_all()
    logger.info("Database initialized at: %s", _db_path)
    logger.info("Session storage: %s", app.config["SESSION_FILE_DIR"])


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(_error):
    """Return JSON for AJAX uploads and a usable message for ordinary requests."""
    message = "Размер файла превышает допустимые 16 МБ."
    if request.path == "/process":
        return jsonify({"success": False, "error": message}), 413
    flash(message, "danger")
    return redirect(url_for("index"))


# Authentication routes
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        remember = request.form.get("remember") == "on"

        if not username or not password:
            flash("Имя пользователя и пароль обязательны", "danger")
            return render_template("login.html")

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user, remember=remember)
            if remember:
                session.permanent = True
            next_page = request.args.get("next")
            if next_page and not urlparse(next_page).netloc:
                return redirect(next_page)
            return redirect(url_for("index"))

        flash("Неверное имя пользователя или пароль", "danger")
        return render_template("login.html")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            flash("Имя пользователя и пароль обязательны", "danger")
            return redirect(url_for("register"))
        if not 3 <= len(username) <= 80:
            flash("Имя пользователя должно содержать от 3 до 80 символов", "danger")
            return redirect(url_for("register"))
        if len(password) < 8:
            flash("Пароль должен быть не менее 8 символов", "danger")
            return redirect(url_for("register"))
        if User.query.filter_by(username=username).first():
            flash("Пользователь с таким именем уже существует", "danger")
            return redirect(url_for("register"))

        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        login_user(user, remember=True)
        session.permanent = True
        return redirect(url_for("index"))

    return render_template("register.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# Processing helpers
def allowed_file(filename: str) -> bool:
    """Check the extension after it has been normalized with secure_filename."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_uploaded_image(file_storage) -> str:
    """Validate the filename and image header without loading model runtimes."""
    filename = secure_filename(file_storage.filename or "")
    if not filename:
        raise ValueError("Не удалось прочитать имя загруженного файла.")
    if not allowed_file(filename):
        raise ValueError("Поддерживаются только JPG, PNG, TIFF и BMP.")

    try:
        with Image.open(file_storage.stream) as image:
            image.verify()
    except Image.DecompressionBombError as exc:
        raise ValueError("Изображение содержит слишком много пикселей для безопасной обработки.") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Файл не является корректным изображением.") from exc
    finally:
        file_storage.stream.seek(0)

    return filename


def update_stage(result_id, stage, stage_data_update=None):
    """Persist a processing stage and any user-visible metadata."""
    with app.app_context():
        try:
            result = db.session.get(ProcessingResult, result_id)
            if result is None:
                return
            result.current_stage = stage
            if stage_data_update:
                try:
                    current_data = json.loads(result.stage_data or "{}")
                except (TypeError, ValueError):
                    current_data = {}
                current_data.update(stage_data_update)
                result.stage_data = json.dumps(current_data, ensure_ascii=False)
            db.session.commit()
        except Exception:
            db.session.rollback()
            logger.exception("Unable to update stage for result %s", result_id)
            raise
        finally:
            db.session.remove()


def run_heavy_operation(operation: str, *args):
    """Run a native/model task outside Flask and apply its dedicated timeout."""
    # Existing unit tests mock these module-level functions. Avoid spawning a
    # subprocess in TESTING mode so those mocks continue to cover the pipeline.
    if app.config.get("TESTING"):
        testing_dispatch = {
            "easyocr": perform_ocr,
            "tesseract": perform_tesseract_ocr,
            "htr": perform_htr,
            "ner": perform_ner,
            "relations": extract_relations,
        }
        return testing_dispatch[operation](*args)
    return run_ml_operation(operation, *args, timeout_seconds=ML_TIMEOUTS[operation])


def process_in_background(result_id, filepath, text_type, ocr_model, translate):
    """Run one document pipeline without allowing native crashes to kill Flask."""
    with app.app_context():
        try:
            update_stage(result_id, "recognizing")
            if text_type == "ocr":
                operation = "tesseract" if ocr_model == "tesseract" else "easyocr"
                text = run_heavy_operation(operation, filepath)
            else:
                _lines, text = run_heavy_operation("htr", filepath)

            result = db.session.get(ProcessingResult, result_id)
            if result is None:
                return
            result.original_text = text
            db.session.commit()
            update_stage(result_id, "recognizing", {
                "recognizing": {"status": "completed", "text": text[:500]}
            })

            if translate:
                update_stage(result_id, "translating")
                # This operation is deterministic string normalization; keeping it
                # in-process avoids starting an unnecessary Python child process.
                text = translate_text(text)
                update_stage(result_id, "translating", {
                    "translating": {"status": "completed", "text": text[:500]}
                })

            update_stage(result_id, "ner")
            annotated_text_html = run_heavy_operation("ner", text)
            result = db.session.get(ProcessingResult, result_id)
            if result is None:
                return
            result.processed_text_html = annotated_text_html
            db.session.commit()
            update_stage(result_id, "ner", {
                "ner": {"status": "completed", "html": annotated_text_html[:5000]}
            })

            # Relation extraction uses a 3B local LLM and is therefore optional:
            # it must not discard useful OCR/NER output if the enhancement fails.
            update_stage(result_id, "relations")
            try:
                relations = run_heavy_operation("relations", text)
                relation_stage = {"status": "completed", "count": len(relations)}
            except MLWorkerError as relation_error:
                logger.warning("Relation extraction skipped for result %s: %s", result_id, relation_error)
                relations = []
                relation_stage = {
                    "status": "completed",
                    "count": 0,
                    "warning": "Извлечение связей недоступно для этой задачи.",
                }

            relations_json = json.dumps(relations, ensure_ascii=False, indent=2)
            relation_stage["json"] = relations_json[:5000]
            result = db.session.get(ProcessingResult, result_id)
            if result is None:
                return
            result.relations_json = relations_json
            db.session.commit()
            update_stage(result_id, "relations", {"relations": relation_stage})

            result = db.session.get(ProcessingResult, result_id)
            if result is None:
                return
            result.current_stage = "completed"
            result.status = "completed"
            db.session.commit()

        except Exception as exc:
            logger.exception("Processing failed for result %s", result_id)
            db.session.rollback()
            result = db.session.get(ProcessingResult, result_id)
            if result is not None:
                result.current_stage = "failed"
                result.status = "failed"
                result.error_message = str(exc)[:500]
                db.session.commit()
        finally:
            # Results pages display the original source scan. Retain it by
            # default; controlled cleanup can opt out through the environment.
            if not KEEP_UPLOADS:
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                except OSError:
                    logger.warning("Could not remove temporary upload: %s", filepath)
            db.session.remove()


# Main routes
@app.route("/", methods=["GET"])
@login_required
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
@login_required
def process():
    if "image" not in request.files or "text_type" not in request.form:
        return jsonify({"success": False, "error": "Не переданы обязательные поля."}), 400

    file = request.files["image"]
    text_type = request.form["text_type"]
    ocr_model = request.form.get("ocr_model", "easyocr")
    translate = "translate" in request.form

    if file.filename == "":
        return jsonify({"success": False, "error": "Выберите файл изображения."}), 400
    if text_type not in {"ocr", "htr"}:
        return jsonify({"success": False, "error": "Неизвестный режим обработки."}), 400
    if ocr_model not in {"easyocr", "tesseract"}:
        return jsonify({"success": False, "error": "Неизвестная OCR-модель."}), 400

    try:
        safe_filename = validate_uploaded_image(file)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    unique_filename = f"{uuid.uuid4().hex}_{safe_filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    file.save(filepath)

    result = ProcessingResult(
        user_id=current_user.id,
        image_filename=unique_filename,
        text_type=text_type,
        ocr_model=ocr_model if text_type == "ocr" else "htr",
        translated=translate,
        status="processing",
        current_stage="queued",
        stage_data="{}",
    )
    db.session.add(result)
    db.session.commit()

    # Flask's in-memory test database is intentionally request-local; starting
    # a background task there would race fixture teardown. Production always
    # submits the job to the bounded executor below.
    if not app.config.get("TESTING"):
        processing_executor.submit(
            process_in_background, result.id, filepath, text_type, ocr_model, translate
        )
    return jsonify({"success": True, "result_id": result.id})


@app.route("/api/result/<int:result_id>/progress")
@login_required
def get_progress(result_id):
    result = db.get_or_404(ProcessingResult, result_id)
    if result.user_id != current_user.id:
        return jsonify({"error": "Unauthorized"}), 403

    try:
        stage_data = json.loads(result.stage_data or "{}")
    except (TypeError, ValueError):
        stage_data = {}

    return jsonify({
        "current_stage": result.current_stage,
        "stage_data": stage_data,
        "status": result.status,
        "error": result.error_message,
        "image_filename": result.image_filename,
        "translated": result.translated,
    })


@app.route("/result/<int:result_id>")
@login_required
def view_result(result_id):
    result = db.get_or_404(ProcessingResult, result_id)
    if result.user_id != current_user.id:
        flash("У вас нет доступа к этому результату", "danger")
        return redirect(url_for("my_results"))
    return render_template("result_detail.html", result=result)


@app.route("/my_results")
@login_required
def my_results():
    results = (
        ProcessingResult.query.filter_by(user_id=current_user.id)
        .order_by(ProcessingResult.created_at.desc())
        .all()
    )
    return render_template("my_results.html", results=results)


@app.route("/ner_check", methods=["GET", "POST"])
@login_required
def ner_check():
    extracted_text = None
    relations = []
    relations_json = "[]"
    translate = False

    if request.method == "POST":
        text = request.form.get("text", "")
        translate = "translate" in request.form
        if text:
            try:
                if translate:
                    text = translate_text(text)
                extracted_text = run_heavy_operation("ner", text)
                try:
                    relations = run_heavy_operation("relations", text)
                except MLWorkerError as relation_error:
                    logger.warning("Manual relation extraction skipped: %s", relation_error)
                relations_json = json.dumps(relations, ensure_ascii=False, indent=2)
            except MLWorkerError as exc:
                flash(f"Не удалось выполнить анализ: {exc}", "danger")

    return render_template(
        "ner_check.html",
        extracted_text=extracted_text,
        relations=relations,
        relations_json=relations_json,
        translate=translate,
    )


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug, threaded=True)
