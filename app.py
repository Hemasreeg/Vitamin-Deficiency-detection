from flask import Flask, request, render_template, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from vitamin_classifier import VitaminDeficiencyClassifier
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret_key_here_12345'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize classifier with the new approach
try:
    classifier = VitaminDeficiencyClassifier(
        'vitamin_deficiency_model_weights.pth',  # weights file
        'class_info.json',
        'preprocessing_info.json',
        'model_config.json'  # new config file
    )
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    classifier = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        flash('No file selected')
        return redirect(url_for('upload_page'))
    
    file = request.files['image']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('upload_page'))
    
    if file and classifier:
        try:
            # Generate unique filename
            filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction
            result = classifier.predict(filepath)
            
            # Return result page
            return render_template('result.html', result=result, filename=filename)
            
        except Exception as e:
            print(f"Error processing image: {e}")
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('upload_page'))
    else:
        flash('Error: Model not loaded properly')
        return redirect(url_for('upload_page'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)