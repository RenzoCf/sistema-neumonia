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
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    result = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    severity = db.Column(db.String(50))
    image_data = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Crear tablas
with app.app_context():
    db.create_all()
    print("✅ Base de datos creada")

# Cargar modelo
MODEL_PATH = 'model/mejor_modelo.h5'
try:
    model = keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("✅ Modelo cargado exitosamente")
except Exception as e:
    print(f"❌ Error al cargar modelo: {e}")
    model = None

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

# ENDPOINTS DE AUTENTICACIÓN

@app.route('/api/register', methods=['POST'])
def register():
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

# ENDPOINTS DE PREDICCIÓN

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo no cargado'}), 500
    
    try:
        file = request.files.get('image')
        
        if not file:
            return jsonify({'error': 'No se encontró imagen'}), 400
        
        # Procesar imagen
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)
        
        # Realizar predicción
        predictions = model.predict(processed_image)
        
        if predictions.shape[-1] == 1:
            confidence = float(predictions[0][0])
            predicted_class = 1 if confidence > 0.5 else 0
            confidence_percentage = confidence * 100 if predicted_class == 1 else (1 - confidence) * 100
        else:
            predicted_class = np.argmax(predictions[0])
            confidence_percentage = float(predictions[0][predicted_class]) * 100
        
        result_text = CLASS_NAMES[predicted_class]
        severity = 'Alta' if confidence_percentage > 85 else 'Moderada' if confidence_percentage > 70 else 'Baja'
        
        # Convertir imagen a base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Guardar análisis
        analysis = Analysis(
            result=result_text,
            confidence=confidence_percentage,
            severity=severity,
            image_data=f"data:image/png;base64,{img_str}"
        )
        db.session.add(analysis)
        db.session.commit()
        
        result = {
            'prediction': result_text,
            'confidence': round(confidence_percentage, 2),
            'severity': severity,
            'is_pneumonia': predicted_class == 1
        }
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        print("ERROR:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
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
    analyses = Analysis.query.order_by(Analysis.created_at.desc()).limit(50).all()
    return jsonify([{
        'id': a.id,
        'result': a.result,
        'confidence': a.confidence,
        'severity': a.severity,
        'image_data': a.image_data,
        'created_at': a.created_at.isoformat()
    } for a in analyses])

if __name__ == '__main__':
    print("🚀 Iniciando servidor Flask con autenticación...")
    print(f"📊 Modelo: {'Cargado' if model else 'No disponible'}")
    app.run(debug=True, host='0.0.0.0', port=5000)

# Configuración del modelo
IMG_SIZE = (224, 224)  # Ajusta según tu modelo
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

def preprocess_image(image):
    """
    Preprocesa la imagen para el modelo
    """
    # Convertir a RGB si es necesario
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Redimensionar
    image = image.resize(IMG_SIZE)
    
    # Convertir a array numpy
    img_array = np.array(image)
    
    # Normalizar (ajusta según tu entrenamiento)
    img_array = img_array / 255.0
    
    # Agregar dimensión batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Verificar que el servidor está activo
    """
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint principal para predicción
    """
    if model is None:
        return jsonify({
            'error': 'Modelo no cargado'
        }), 500
    
    try:
        # Obtener imagen del request
        if 'image' not in request.files:
            return jsonify({
                'error': 'No se encontró imagen'
            }), 400
        
        file = request.files['image']
        
        # Leer imagen
        image = Image.open(io.BytesIO(file.read()))
        
        # Preprocesar
        processed_image = preprocess_image(image)
        
        # Realizar predicción
        predictions = model.predict(processed_image)
        
        # Interpretar resultados
        # Si tu modelo devuelve 1 valor (sigmoid): 
        if predictions.shape[-1] == 1:
            confidence = float(predictions[0][0])
            predicted_class = 1 if confidence > 0.5 else 0
            confidence_percentage = confidence * 100 if predicted_class == 1 else (1 - confidence) * 100
        
        # Si devuelve 2 valores (softmax):
        else:
            predicted_class = np.argmax(predictions[0])
            confidence_percentage = float(predictions[0][predicted_class]) * 100
        
        result = {
            'prediction': CLASS_NAMES[predicted_class],
            'confidence': round(confidence_percentage, 2),
            'is_pneumonia': predicted_class == 1,
            'raw_predictions': predictions.tolist()
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """
    Información del modelo
    """
    if model is None:
        return jsonify({'error': 'Modelo no disponible'}), 500
    
    try:
        return jsonify({
            'architecture': str(model.summary()),
            'input_shape': str(model.input_shape),
            'output_shape': str(model.output_shape),
            'total_params': model.count_params()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Iniciando servidor Flask...")
    print(f"📊 Modelo: {'Cargado' if model else 'No disponible'}")
    app.run(debug=True, host='0.0.0.0', port=5000)