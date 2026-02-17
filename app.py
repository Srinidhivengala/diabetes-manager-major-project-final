import os
import io
import csv
import json
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any

from real_smtp import send_otp_email
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, send_file, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_wtf.csrf import CSRFProtect
from werkzeug.security import generate_password_hash, check_password_hash

from ml.model import ModelService
from ml.federated_model import FederatedModelService
from ml.explain import explain_prediction
from ml.forecast import forecast_glucose

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, 'app.db')

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Database models (from your models.py)
from models import db, User, PatientProfile, GlucoseReading, RiskPrediction, Alert, AuditLog, RegistrationOTP

db.init_app(app)

# CSRF protection
csrf = CSRFProtect()
csrf.init_app(app)

# Auth
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# Template filter for local time formatting
@app.template_filter('localtime')
def localtime_filter(dt):
    if dt is None:
        return ''
    return dt.strftime('%Y-%m-%d %H:%M:%S')


@app.context_processor
def inject_helpers():
    def get_user_display_name(user):
        """Return a human-friendly display name for a user for templates."""
        if not user:
            return ''
        # Prefer profile name if available
        try:
            profile = getattr(user, 'profile', None)
            if profile:
                name = getattr(profile, 'name', None) or ' '
                if name and name.strip():
                    return name
        except Exception:
            pass

        # Fallback to first/last name attributes on user, then email
        first = getattr(user, 'first_name', None)
        last = getattr(user, 'last_name', None)
        if first or last:
            return f"{first or ''} {last or ''}".strip()
        return getattr(user, 'email', '') or ''

    return {'get_user_display_name': get_user_display_name}


@app.route('/dashboard')
def dashboard():
    # Example dummy data for the dashboard page
    total_readings = 25
    avg_glucose = 118
    total_reminders = 3

    chart_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    chart_values = [110, 120, 115, 130, 125, 118, 122]

    return render_template(
        "dashboard.html",
        total_readings=total_readings,
        avg_glucose=avg_glucose,
        total_reminders=total_reminders,
        chart_labels=chart_labels,
        chart_values=chart_values,
    )

# RBAC helpers

def role_required(*roles):
    def decorator(fn):
        from functools import wraps
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not current_user.is_authenticated:
                return redirect(url_for('login'))
            if current_user.role not in roles:
                flash('Unauthorized', 'error')
                return redirect(url_for('index'))
            return fn(*args, **kwargs)
        return wrapper
    return decorator


@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/')
@login_required
def index():
    if current_user.role == 'admin':
        return redirect(url_for('admin_dashboard'))
    elif current_user.role == 'clinician':
        return redirect(url_for('clinician_dashboard'))
    return redirect(url_for('patient_dashboard'))


def log_audit(user_id, action, details=None, ip_address=None):
    """Helper function to log audit events"""
    try:
        log = AuditLog(
            user_id=user_id,
            action=action,
            details=details,
            ip_address=ip_address or request.remote_addr
        )
        db.session.add(log)
        db.session.commit()
    except Exception as e:
        print(f'Error logging audit: {e}')
        db.session.rollback()


def generate_otp_code(length: int = 6) -> str:
    """Generate a numeric OTP code with the given length."""
    upper = 10 ** length
    return f"{secrets.randbelow(upper):0{length}d}"


def send_registration_otp(recipient_email: str, otp_code: str) -> bool:
    """
    Send OTP via REAL email only (no console, no fallback)
    """
    success = send_otp_email(recipient_email, otp_code)

    if not success:
        # Stop registration immediately if email fails
        raise RuntimeError("Failed to send OTP email")

    return True



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            log_audit(user.id, 'user_login', f'User {email} logged in', request.remote_addr)
            return redirect(url_for('index'))
        log_audit(None, 'login_failed', f'Failed login attempt for {email}', request.remote_addr)
        flash('Invalid credentials', 'error')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        # Force role to patient - no role selection for new users
        role = 'patient'
        
        # Validation
        if not email or not password:
            flash('Email and password are required', 'error')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'error')
            return render_template('register.html')
        
        # Get name fields
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        
        # Validation for names
        if not first_name or not last_name:
            flash('First name and last name are required', 'error')
            return render_template('register.html')
        
        # Check if user already exists
        if User.query.filter_by(email=email).first():
            flash('Email already registered. Please login instead.', 'error')
            return redirect(url_for('login'))
        
        # Prepare OTP record
        otp_code = generate_otp_code()
        expires_at = datetime.now() + timedelta(minutes=int(os.environ.get('OTP_EXPIRY_MINUTES', '10')))
        payload = json.dumps({
            'email': email,
            'password_hash': generate_password_hash(password),
            'role': role,
            'first_name': first_name,
            'last_name': last_name
        })

        otp_record = RegistrationOTP.query.filter_by(email=email).first()
        if not otp_record:
            otp_record = RegistrationOTP(
                email=email,
                otp_hash=generate_password_hash(otp_code),
                payload=payload,
                expires_at=expires_at,
                attempts=0
            )
            db.session.add(otp_record)
        else:
            otp_record.otp_hash = generate_password_hash(otp_code)
            otp_record.payload = payload
            otp_record.expires_at = expires_at
            otp_record.attempts = 0
            otp_record.created_at = datetime.now()

        db.session.commit()

        send_registration_otp(email, otp_code)
        session['pending_registration_email'] = email
        log_audit(None, 'registration_otp_sent', f'OTP sent to {email}', request.remote_addr)
        flash('We sent a one-time password (OTP) to your email. Enter it below to complete registration.', 'success')
        return redirect(url_for('verify_registration', email=email))
    
    email_prefill = request.args.get('email') or session.get('pending_registration_email', '')
    return render_template('register.html', email=email_prefill)


@app.route('/register/verify', methods=['GET', 'POST'])
def verify_registration():
    pending_email = session.get('pending_registration_email', '')
    if request.method == 'POST':
        email = (request.form.get('email') or '').strip().lower()
        otp_input = (request.form.get('otp') or '').strip()

        if not email or not otp_input:
            flash('Email and OTP are required.', 'error')
            return render_template('otp_verify.html', email=email)

        otp_record = RegistrationOTP.query.filter_by(email=email).first()
        if not otp_record:
            flash('No pending registration found. Please start again.', 'error')
            return redirect(url_for('register', email=email))

        if otp_record.expires_at < datetime.now():
            db.session.delete(otp_record)
            db.session.commit()
            log_audit(None, 'registration_otp_expired', f'OTP expired for {email}', request.remote_addr)
            flash('OTP expired. Please request a new one.', 'error')
            return redirect(url_for('register', email=email))

        otp_record.attempts += 1
        if otp_record.attempts > 5:
            db.session.delete(otp_record)
            db.session.commit()
            log_audit(None, 'registration_otp_blocked', f'OTP attempts exceeded for {email}', request.remote_addr)
            flash('Too many invalid attempts. Please restart registration.', 'error')
            return redirect(url_for('register'))

        if not check_password_hash(otp_record.otp_hash, otp_input):
            db.session.commit()
            log_audit(None, 'registration_otp_invalid', f'Invalid OTP for {email}', request.remote_addr)
            flash('Invalid OTP. Please try again.', 'error')
            return render_template('otp_verify.html', email=email)

        # Prevent duplicate user creation
        if User.query.filter_by(email=email).first():
            db.session.delete(otp_record)
            db.session.commit()
            log_audit(None, 'registration_otp_duplicate', f'User already exists for {email}', request.remote_addr)
            flash('Email already registered. Please login.', 'info')
            return redirect(url_for('login'))

        # Create user from payload
        data = json.loads(otp_record.payload)
        user = User(
            email=data['email'],
            password_hash=data['password_hash'],
            role=data.get('role', 'patient')
        )
        db.session.add(user)
        db.session.flush()

        # Create patient profile
        first_name = data.get('first_name')
        last_name = data.get('last_name')
        profile = PatientProfile(
            user_id=user.id,
            first_name=first_name,
            last_name=last_name,
            name=f"{first_name} {last_name}".strip()
        )
        db.session.add(profile)

        db.session.delete(otp_record)
        db.session.commit()

        session.pop('pending_registration_email', None)
        log_audit(user.id, 'user_register', f'New patient registered via OTP: {email}', request.remote_addr)
        flash('Registration complete! You can now log in.', 'success')
        return redirect(url_for('login'))

    # GET request
    email_param = request.args.get('email') or pending_email or ''
    return render_template('otp_verify.html', email=email_param)


@app.route('/logout')
@login_required
def logout():
    log_audit(current_user.id, 'user_logout', f'User {current_user.email} logged out', request.remote_addr)
    logout_user()
    return redirect(url_for('login'))


@app.route('/patient')
@login_required
@role_required('patient')
def patient_dashboard():
    readings = GlucoseReading.query.filter_by(user_id=current_user.id).order_by(GlucoseReading.timestamp.desc()).limit(100).all()
    predictions = RiskPrediction.query.filter_by(user_id=current_user.id).order_by(RiskPrediction.created_at.desc()).limit(20).all()
    alerts = Alert.query.filter_by(user_id=current_user.id).order_by(Alert.created_at.desc()).limit(10).all()
    return render_template('patient_dashboard.html', readings=readings, predictions=predictions, alerts=alerts)


@app.route('/clinician')
@login_required
@role_required('clinician')
def clinician_dashboard():
    patients = User.query.filter_by(role='patient').all()
    latest_predictions = {p.id: RiskPrediction.query.filter_by(user_id=p.id).order_by(RiskPrediction.created_at.desc()).first() for p in patients}
    return render_template('clinician_dashboard.html', patients=patients, latest_predictions=latest_predictions)


@app.route('/clinician/export/<int:patient_id>/csv')
@login_required
@role_required('clinician')
def export_patient_csv(patient_id: int):
    readings = GlucoseReading.query.filter_by(user_id=patient_id).order_by(GlucoseReading.timestamp.asc()).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['timestamp', 'glucose'])
    for r in readings:
        writer.writerow([r.timestamp.isoformat(), r.glucose])
    mem = io.BytesIO(output.getvalue().encode('utf-8'))
    mem.seek(0)
    filename = f'patient_{patient_id}_glucose.csv'
    return send_file(mem, as_attachment=True, download_name=filename, mimetype='text/csv')


# API: Predict
@app.post('/api/predict')
@login_required
def api_predict():
    try:
        payload = request.get_json(force=True)
        if not payload:
            return jsonify({'error': 'Empty payload'}), 400
    except Exception as e:
        return jsonify({'error': f'Invalid JSON: {str(e)}'}), 400
    
    # Try federated model first, fallback to regular model
    federated_service = FederatedModelService()
    if federated_service.is_available():
        model_service = federated_service
    else:
        model_service = ModelService()
    
    try:
        pred, prob, features_order = model_service.predict(payload)
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 400

    rp = RiskPrediction(user_id=current_user.id, prediction=int(pred), probability=float(prob))
    db.session.add(rp)
    db.session.commit()

    # simple alert rule
    if prob >= 0.7:
        alert = Alert(user_id=current_user.id, message=f'High diabetes risk: {prob:.2f}', level='high')
        db.session.add(alert)
        db.session.commit()

    log_audit(current_user.id, 'prediction_made', f'Risk prediction: {pred}, probability: {prob:.2f}', request.remote_addr)
    
    # Return prediction with timestamp for immediate UI update
    return jsonify({
        'prediction': int(pred), 
        'probability': float(prob), 
        'features': features_order,
        'created_at': rp.created_at.isoformat(),
        'id': rp.id
    })


# API: Get Recent Predictions
# API: Get all predictions
@app.get('/api/predictions')
@login_required
def api_all_predictions():
    """Get all predictions for the current user (used for variant rotation)"""
    predictions = RiskPrediction.query.filter_by(user_id=current_user.id).all()
    return jsonify([{
        'id': p.id,
        'prediction': p.prediction,
        'probability': float(p.probability),
        'created_at': p.created_at.isoformat()
    } for p in predictions])


@app.get('/api/predictions/recent')
@login_required
def api_recent_predictions():
    """Get recent predictions for the current user"""
    predictions = RiskPrediction.query.filter_by(user_id=current_user.id).order_by(RiskPrediction.created_at.desc()).limit(20).all()
    return jsonify({
        'predictions': [{
            'id': p.id,
            'prediction': p.prediction,
            'probability': float(p.probability),
            'created_at': p.created_at.isoformat()
        } for p in predictions]
    })


# API: Explain
@app.post('/api/explain')
@login_required
def api_explain():
    try:
        payload = request.get_json(force=True)
        if not payload:
            return jsonify({'error': 'Empty payload'}), 400
    except Exception as e:
        return jsonify({'error': f'Invalid JSON: {str(e)}'}), 400
    
    try:
        model_service = ModelService()
        pred, prob, features_order = model_service.predict(payload)
        shap_values = explain_prediction(model_service.model, payload, features_order)
        return jsonify({'prediction': int(pred), 'probability': float(prob), 'shap_values': shap_values, 'features': features_order})
    except Exception as e:
        return jsonify({'error': f'Explanation error: {str(e)}'}), 400


# API: Forecast
@app.post('/api/forecast')
@login_required
def api_forecast():
    try:
        data: Dict[str, Any] = {}
        cgm_uploaded = False
        
        if request.is_json:
            data = request.get_json()
        elif 'file' in request.files:
            file = request.files['file']
            text = file.read().decode('utf-8')
            reader = csv.DictReader(io.StringIO(text))
            readings = []
            for row in reader:
                readings.append({
                    'timestamp': row.get('timestamp'),
                    'glucose': float(row.get('glucose', '0') or 0)
                })
            data = {'readings': readings}
            cgm_uploaded = True
            
            # Store CGM readings in database
            for reading in readings:
                try:
                    # Parse timestamp
                    ts_str = reading.get('timestamp')
                    if isinstance(ts_str, str):
                        # Try ISO format first
                        try:
                            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                        except:
                            # Try other common formats
                            try:
                                ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
                            except:
                                ts = datetime.strptime(ts_str, '%Y-%m-%dT%H:%M:%S')
                    else:
                        ts = datetime.now()
                    
                    # Check if reading already exists (avoid duplicates)
                    existing = db.session.query(GlucoseReading).filter_by(
                        user_id=current_user.id,
                        timestamp=ts,
                        source='cgm'
                    ).first()
                    
                    if not existing:
                        glucose_reading = GlucoseReading(
                            user_id=current_user.id,
                            timestamp=ts,
                            glucose=float(reading.get('glucose', 0)),
                            source='cgm'
                        )
                        db.session.add(glucose_reading)
                except Exception as e:
                    print(f'Error storing CGM reading: {e}')
                    continue
            
            db.session.commit()
            log_audit(current_user.id, 'cgm_uploaded', f'CGM data uploaded: {len(readings)} readings', request.remote_addr)
        else:
            return jsonify({'error': 'No data provided'}), 400

        readings = data.get('readings', [])
        if not readings:
            # fallback to last readings from DB
            db_readings = db.session.query(GlucoseReading).filter_by(user_id=current_user.id).order_by(GlucoseReading.timestamp.desc()).limit(50).all()
            readings = [{'timestamp': r.timestamp.isoformat(), 'glucose': r.glucose} for r in reversed(db_readings)]

        forecast_points = forecast_glucose(readings)
        # cache latest forecast points as readings
        now = datetime.now()
        for pt in forecast_points:
            ts = datetime.fromisoformat(pt['timestamp']) if isinstance(pt['timestamp'], str) else now
            db.session.add(GlucoseReading(user_id=current_user.id, timestamp=ts, glucose=float(pt['glucose']), source='forecast'))
        db.session.commit()

        return jsonify({'forecast': forecast_points, 'cgm_stored': cgm_uploaded})
    except Exception as e:
        print(f'Forecast error: {e}')
        return jsonify({'error': f'Forecast error: {str(e)}'}), 400


@app.route('/admin')
@login_required
@role_required('admin')
def admin_dashboard():
    # Get statistics
    total_users = User.query.count()
    total_patients = User.query.filter_by(role='patient').count()
    total_clinicians = User.query.filter_by(role='clinician').count()
    total_admins = User.query.filter_by(role='admin').count()
    
    # Get recent audit logs
    audit_logs = AuditLog.query.order_by(AuditLog.created_at.desc()).limit(100).all()
    
    # Get all users for management
    users = User.query.order_by(User.created_at.desc()).all()
    
    return render_template('admin_dashboard.html', 
                         total_users=total_users,
                         total_patients=total_patients,
                         total_clinicians=total_clinicians,
                         total_admins=total_admins,
                         audit_logs=audit_logs,
                         users=users)


@app.route('/admin/create-clinician', methods=['GET', 'POST'])
@login_required
@role_required('admin')
def create_clinician():
    if request.method == 'POST':
        try:
            email = request.form.get('email')
            password = request.form.get('password')
            first_name = request.form.get('first_name', '').strip()
            last_name = request.form.get('last_name', '').strip()
            
            # Validation
            if not email or not password:
                flash('Email and password are required', 'error')
                return redirect(url_for('admin_dashboard'))
            
            if len(password) < 6:
                flash('Password must be at least 6 characters long', 'error')
                return redirect(url_for('admin_dashboard'))
            
            # Check if user already exists
            if User.query.filter_by(email=email).first():
                flash('Email already registered', 'error')
                return redirect(url_for('admin_dashboard'))
            
            # Create clinician user
            clinician = User(
                email=email,
                password_hash=generate_password_hash(password),
                role='clinician'
            )
            db.session.add(clinician)
            db.session.commit()
            
            log_audit(current_user.id, 'clinician_created', f'Admin created clinician: {email}', request.remote_addr)
            flash(f'Clinician {email} created successfully', 'success')
            return redirect(url_for('admin_dashboard'))
        except Exception as e:
            print(f'Error creating clinician: {e}')
            flash(f'Error creating clinician: {str(e)}', 'error')
            return redirect(url_for('admin_dashboard'))
    
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/delete-user/<int:user_id>', methods=['POST'])
@login_required
@role_required('admin')
def delete_user(user_id):
    try:
        user = User.query.get_or_404(user_id)
        
        # Prevent deleting yourself
        if user.id == current_user.id:
            flash('You cannot delete your own account', 'error')
            return redirect(url_for('admin_dashboard'))
        
        email = user.email
        db.session.delete(user)
        db.session.commit()
        
        log_audit(current_user.id, 'user_deleted', f'Admin deleted user: {email}', request.remote_addr)
        flash(f'User {email} deleted successfully', 'success')
        return redirect(url_for('admin_dashboard'))
    except Exception as e:
        print(f'Error in delete_user: {e}')
        flash(f'Error deleting user: {str(e)}', 'error')
        return redirect(url_for('admin_dashboard'))


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


# Database migration helper
def migrate_database():
    """Add new columns to existing database if they don't exist"""
    from sqlalchemy import inspect, text
    
    try:
        inspector = inspect(db.engine)
        table_names = inspector.get_table_names()
        
        # Check if patient_profile table exists
        if 'patient_profile' in table_names:
            columns = [col['name'] for col in inspector.get_columns('patient_profile')]
            
            # Check if first_name and last_name columns exist
            if 'first_name' not in columns:
                try:
                    db.session.execute(text('ALTER TABLE patient_profile ADD COLUMN first_name VARCHAR(100)'))
                    db.session.commit()
                    print('Added first_name column to patient_profile')
                except Exception as e:
                    db.session.rollback()
                    print(f'Error adding first_name column: {e}')
            
            if 'last_name' not in columns:
                try:
                    db.session.execute(text('ALTER TABLE patient_profile ADD COLUMN last_name VARCHAR(100)'))
                    db.session.commit()
                    print('Added last_name column to patient_profile')
                except Exception as e:
                    db.session.rollback()
                    print(f'Error adding last_name column: {e}')
        
        # Note: audit_log table will be created by db.create_all() if it doesn't exist
    except Exception as e:
        print(f'Migration check failed (atabase is new): {e}')

# CLI helper to init DB and seed users
@app.cli.command('initdb')
def initdb():
    with app.app_context():
        db.create_all()
        migrate_database()
        seed_defaults()
        print('Initialized the database and seeded defaults.')


def seed_defaults():
    # Create admin user
    if not User.query.filter_by(email='admin@example.com').first():
        admin = User(email='admin@example.com', role='admin', password_hash=generate_password_hash('admin123'))
        db.session.add(admin)
        print('Created admin user: admin@example.com / admin123')
    
    # Create default clinician
    if not User.query.filter_by(email='clinician@example.com').first():
        clinician = User(email='clinician@example.com', role='clinician', password_hash=generate_password_hash('password123'))
        db.session.add(clinician)
    
    # Create default patient
    if not User.query.filter_by(email='patient@example.com').first():
        patient = User(email='patient@example.com', role='patient', password_hash=generate_password_hash('password123'))
        db.session.add(patient)
        db.session.flush()
        # Create profile for default patient if it doesn't exist
        if not patient.profile:
            profile = PatientProfile(
                user_id=patient.id,
                first_name='John',
                last_name='Doe',
                name='John Doe'
            )
            db.session.add(profile)
    db.session.commit()


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        migrate_database()
        seed_defaults()
    app.run(host='127.0.0.1', port=5000, debug=True)

