import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


RESUME_PATH = os.path.join("Cleaned Data", "Resumes", "Resume", "Resume_clean.csv")
MODEL_PATH = os.path.join("webApp", "models", "resume_classifier.pkl")


"""
Here I'm training a basic classifier that tries to guess what job category a resume belongs to.
I kept it simple using TF-IDF and Naive Bayes because this is just our starting point
This helps us get a baseline and make sure our resume data actually contains enough signal to predict categories like IT, HR, or Finance.
"""
def train_resume_classifier():
    df = pd.read_csv(RESUME_PATH)
    df = df.dropna(subset=['clean_text', 'Category'])
    if 'Category' not in df.columns or 'clean_text' not in df.columns:
        raise ValueError("Resume dataset must include 'Category' and 'clean_text' columns.")
    X = df['clean_text']
    y = df['Category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('nb', MultinomialNB())
    ])
    clf.fit(X_train, y_train)

    print("\nClassification Report on Validation Set:")
    print(classification_report(y_test, clf.predict(X_test)))

    return clf

"""
Just saving the trained model to a file so I don't have to retrain it every time I run the code.
"""
def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

"""
Loading the saved model if it exists — and if not, I just train a new one on the fly. This way, the system always has a model to work with
"""
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        print("Model file not found. Training a new model...")
        model = train_resume_classifier()
        save_model(model, path)
    return joblib.load(path)


def predict_category(text, model):
    return model.predict([text])[0]


def classify_resume(resume_text):
    model = load_model()
    return predict_category(resume_text, model)


def match_jobs(resume_text, category):
    job_list = {
        "Finance": ["Financial Analyst", "Accountant", "Investment Banker"],
        "Information Technology": ["Software Developer", "Data Scientist", "Systems Engineer"],
        "Human Resources": ["HR Coordinator", "Recruiter", "Talent Acquisition"],
        "Other": ["General Admin", "Support Assistant", "Operations Executive"]
    }
    return job_list.get(category, [])


if __name__ == "__main__":
    model = train_resume_classifier()
    save_model(model)
