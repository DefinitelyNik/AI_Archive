"""Database models for users and processing results."""

from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """User model for authentication."""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    # FIX: Увеличили длину до 512 для надёжности
    password_hash = db.Column(db.String(512), nullable=False)
    # FIX: datetime.utcnow() deprecated → datetime.now(timezone.utc)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # FIX: Добавлен cascade для удаления результатов при удалении пользователя
    results = db.relationship(
        'ProcessingResult',
        backref='user',
        lazy=True,
        cascade='all, delete-orphan'
    )

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"<User {self.username}>"


class ProcessingResult(db.Model):
    """Model for storing processing results with stage tracking."""
    __tablename__ = 'processing_results'

    id = db.Column(db.Integer, primary_key=True)
    # FIX: Добавлены индексы для частых запросов
    user_id = db.Column(
        db.Integer,
        db.ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )

    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        index=True
    )
    image_filename = db.Column(db.String(255))
    text_type = db.Column(db.String(20))
    ocr_model = db.Column(db.String(50))
    translated = db.Column(db.Boolean, default=False)

    current_stage = db.Column(db.String(30), default='queued')
    # FIX: Храним только метаданные, а не полный текст на каждом этапе
    stage_data = db.Column(db.Text, default='{}')

    original_text = db.Column(db.Text)
    processed_text_html = db.Column(db.Text)
    relations_json = db.Column(db.Text)

    status = db.Column(db.String(20), default='processing', index=True)
    error_message = db.Column(db.Text)

    STAGES = ['queued', 'recognizing', 'translating', 'ner', 'relations', 'completed', 'failed']

    def to_dict(self):
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'image_filename': self.image_filename,
            'text_type': self.text_type,
            'ocr_model': self.ocr_model,
            'translated': self.translated,
            'current_stage': self.current_stage,
            'stage_data': self.stage_data,
            'original_text': self.original_text,
            'processed_text_html': self.processed_text_html,
            'relations_json': self.relations_json,
            'status': self.status
        }

    def __repr__(self):
        return f"<ProcessingResult {self.id} user={self.user_id} stage={self.current_stage}>"