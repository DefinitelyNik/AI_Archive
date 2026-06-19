"""Database models for users and processing results."""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """User model for authentication."""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    results = db.relationship('ProcessingResult', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class ProcessingResult(db.Model):
    """Model for storing processing results with stage tracking."""
    __tablename__ = 'processing_results'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    image_filename = db.Column(db.String(255))
    text_type = db.Column(db.String(20))
    ocr_model = db.Column(db.String(50))
    translated = db.Column(db.Boolean, default=False)

    current_stage = db.Column(db.String(30), default='queued')
    stage_data = db.Column(db.Text, default='{}')  # JSON with intermediate results

    original_text = db.Column(db.Text)
    processed_text_html = db.Column(db.Text)
    relations_json = db.Column(db.Text)

    status = db.Column(db.String(20), default='processing')
    error_message = db.Column(db.Text)

    STAGES = ['queued', 'recognizing', 'translating', 'ner', 'relations', 'completed', 'failed']

    def to_dict(self):
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat(),
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