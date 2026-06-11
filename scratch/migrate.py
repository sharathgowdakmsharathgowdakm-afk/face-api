import sys
import os

# Add parent directory to path to allow importing app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, db
from sqlalchemy import text, inspect

with app.app_context():
    print("Database URI:", app.config['SQLALCHEMY_DATABASE_URI'])
    
    # Check current columns
    inspector = inspect(db.engine)
    columns = [col['name'] for col in inspector.get_columns('face_encoding')]
    print("Columns before migration:", columns)
    
    if 'encoding_data' not in columns:
        print("Attempting to add column encoding_data...")
        try:
            db.session.execute(text("ALTER TABLE face_encoding ADD COLUMN encoding_data JSON"))
            db.session.commit()
            print("Successfully added encoding_data column!")
        except Exception as e:
            db.session.rollback()
            print("Error running migration:", e)
    else:
        print("encoding_data column already exists in schema.")
        
    # Check columns after migration
    columns_after = [col['name'] for col in inspector.get_columns('face_encoding')]
    print("Columns after migration:", columns_after)
