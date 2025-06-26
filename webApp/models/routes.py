from flask import Flask, render_template, request
from webApp.models.model_utils import classify_resume, match_jobs

app = Flask(__name__,
            template_folder='../templates',
            static_folder='../static')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    resume_text = request.form.get('resume_text')
    if not resume_text:
        return "No resume text provided", 400

    category = classify_resume(resume_text)
    matched_jobs = match_jobs(resume_text, category)

    return render_template('results.html', prediction=category, resume_text=resume_text, jobs=matched_jobs)
