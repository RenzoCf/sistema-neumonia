from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import keras
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime
import os
import json # Necesario para manejar la serialización de síntomas

app = Flask(__name__)
CORS(app)

# Configuración de la base de datos
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'pneumodetect.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'pneumodetect-secret-key-2024'
db = SQLAlchemy(app)

# MODELOS DE BASE DE DATOS

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(200), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    dni = db.Column(db.String(20), unique=True, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.String(20), nullable=False)
    phone = db.Column(db.String(20))
    address = db.Column(db.String(300))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    analyses = db.relationship('Analysis', backref='patient', lazy=True, cascade='all, delete-orphan')

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False) # Ahora es OBLIGATORIO
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True) 
    result = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    severity = db.Column(db.String(50))
    image_data = db.Column(db.Text, nullable=False)
    symptoms = db.Column(db.Text) # Agregamos síntomas
    temperature = db.Column(db.Float) # Agregamos temperatura
    notes = db.Column(db.Text) # Agregamos notas
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Crear tablas
with app.app_context():
    # ¡IMPORTANTE! Descomentamos esta línea para forzar la creación de la DB con las nuevas columnas
    db.create_all() 
    print("✅ Base de datos verificada/creada")

# CONFIGURACIÓN DEL MODELO Y PREPROCESAMIENTO
MODEL_PATH = 'model/modelo_neumonia_final.h5' 
model = None
try:
    model = keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    if model.input_shape:
        print(f"Forma de entrada esperada por el modelo: {model.input_shape}")
    print("✅ Modelo cargado exitosamente")
except Exception as e:
    print(f"❌ Error al cargar modelo desde '{MODEL_PATH}': {e}")
    model = None

# TAMAÑO CORREGIDO A 150x150
IMG_SIZE = (150, 150) 
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

# --- LÓGICA DE PACIENTES ---

def get_or_create_patient(patient_data):
    """ Busca un paciente por DNI; si no existe, lo crea. """
    patient = Patient.query.filter_by(dni=patient_data['dni']).first()
    
    if not patient:
        patient = Patient(
            name=patient_data['name'],
            dni=patient_data['dni'],
            age=int(patient_data['age']),
            sex=patient_data['sex'],
            phone=patient_data.get('phone'),
            address=patient_data.get('address')
        )
        db.session.add(patient)
        db.session.commit()
    return patient

# ENDPOINTS DE AUTENTICACIÓN (sin cambios)

@app.route('/api/register', methods=['POST'])
def register():
    # ... (código register sin cambios)
    data = request.json
    
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'El email ya está registrado'}), 400
    
    user = User(
        name=data['name'],
        email=data['email']
    )
    user.set_password(data['password'])
    
    db.session.add(user)
    db.session.commit()
    
    return jsonify({
        'message': 'Usuario registrado exitosamente',
        'user': {
            'id': user.id,
            'name': user.name,
            'email': user.email
        }
    })

@app.route('/api/login', methods=['POST'])
def login():
    # ... (código login sin cambios)
    data = request.json
    user = User.query.filter_by(email=data['email']).first()
    
    if not user or not user.check_password(data['password']):
        return jsonify({'error': 'Email o contraseña incorrectos'}), 401 
    
    return jsonify({
        'message': 'Login exitoso',
        'user': {
            'id': user.id,
            'name': user.name,
            'email': user.email
        }
    })

# ENDPOINTS DE PREDICCIÓN Y ESTADÍSTICAS

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo no cargado'}), 500
    
    try:
        # 1. Obtener datos del paciente (JSON) y la imagen (File)
        image_file = request.files.get('image')
        patient_data_json = request.form.get('patient_data')

        if not image_file:
            return jsonify({'error': 'No se encontró archivo de imagen'}), 400
        if not patient_data_json:
            return jsonify({'error': 'No se encontraron datos del paciente'}), 400

        patient_data = json.loads(patient_data_json)

        # 2. Buscar/Crear Paciente
        patient = get_or_create_patient(patient_data)
        
        # 3. Preprocesar Imagen
        image = Image.open(io.BytesIO(image_file.read()))
        processed_image = preprocess_image(image)
        
        # 4. Realizar Predicción
        predictions = model.predict(processed_image)
        
        if predictions.shape[-1] == 1: 
            confidence = float(predictions[0][0])
            predicted_class = 1 if confidence > 0.5 else 0
            confidence_percentage = confidence * 100 if predicted_class == 1 else (1 - confidence) * 100
        else:
            predicted_class = np.argmax(predictions[0])
            confidence_percentage = float(predictions[0][predicted_class]) * 100
        
        result_text = CLASS_NAMES[predicted_class]
        
        if result_text == 'PNEUMONIA':
            severity = 'Alta' if confidence_percentage > 85 else 'Moderada' if confidence_percentage > 70 else 'Baja'
        else:
            severity = 'N/A' 
        
        # 5. Convertir imagen a base64 (para guardar)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # 6. Guardar Análisis con Patient ID y datos clínicos
        analysis = Analysis(
            patient_id=patient.id, # <--- ¡USAMOS EL ID REAL!
            user_id=None, # Puedes obtener el ID del usuario si implementas JWT/Sesiones
            result=result_text,
            confidence=confidence_percentage,
            severity=severity,
            image_data=f"data:image/png;base64,{img_str}",
            symptoms=json.dumps(patient_data.get('symptoms', [])), # Guardamos como JSON string
            temperature=patient_data.get('temperature'),
            notes=patient_data.get('notes')
        )
        db.session.add(analysis)
        db.session.commit()
        
        # 7. Respuesta
        result = {
            'prediction': result_text,
            'confidence': round(confidence_percentage, 2),
            'severity': severity,
            'is_pneumonia': predicted_class == 1,
            'analysis_id': analysis.id
        }
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        print("ERROR en /api/predict:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    # ... (código stats sin cambios)
    total = Analysis.query.count()
    normal = Analysis.query.filter_by(result='NORMAL').count()
    pneumonia = Analysis.query.filter_by(result='PNEUMONIA').count()
    avg_confidence = db.session.query(db.func.avg(Analysis.confidence)).scalar() or 0
    
    return jsonify({
        'total': total,
        'normal': normal,
        'pneumonia': pneumonia,
        'avg_confidence': round(avg_confidence, 2)
    })

@app.route('/api/history', methods=['GET'])
def get_history():
    """
    Obtiene el historial, incluyendo los datos del paciente asociados.
    """
    # Consulta que carga también los datos del paciente
    analyses = db.session.query(Analysis, Patient).join(Patient).order_by(Analysis.created_at.desc()).limit(50).all()
    
    history_list = []
    for analysis, patient in analyses:
        history_list.append({
            'id': analysis.id,
            'patient_id': patient.id,
            'result': analysis.result,
            'confidence': analysis.confidence,
            'severity': analysis.severity,
            'image_data': analysis.image_data,
            'created_at': analysis.created_at.isoformat(),
            'patient': { # Incluir todos los datos del paciente y clínicos
                'name': patient.name,
                'dni': patient.dni,
                'age': patient.age,
                'sex': patient.sex,
                'phone': patient.phone,
                'address': patient.address,
                'symptoms': json.loads(analysis.symptoms) if analysis.symptoms else [], # Deserializar
                'temperature': analysis.temperature,
                'notes': analysis.notes
            }
        })
    return jsonify(history_list)


@app.route('/api/patients', methods=['GET'])
def get_patients():
    """
    Obtiene la lista de todos los pacientes.
    """
    patients_list = Patient.query.order_by(Patient.name).all()
    return jsonify([{
        'id': p.id,
        'name': p.name,
        'dni': p.dni,
        'age': p.age,
        'sex': p.sex,
        'phone': p.phone,
        'address': p.address
    } for p in patients_list])


if __name__ == '__main__':
    print("🚀 Iniciando servidor Flask con autenticación y modelo...")
    print(f"📊 Modelo: {'Cargado' if model else 'No disponible'}")
    app.run(debug=True, host='0.0.0.0', port=5000)