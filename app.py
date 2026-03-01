from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# --- Load and Prepare Data (Run once on startup) ---
df = pd.read_excel("Gen_AI Dataset.xlsx")
df = df[['Query', 'Assessment_url']].dropna()
df['Query'] = df['Query'].str.strip()
df['Assessment_url'] = df['Assessment_url'].str.strip()
df = df.drop_duplicates()

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Query'])

def get_recommendations(new_query, top_k=3):
    new_vector = vectorizer.transform([new_query])
    similarities = cosine_similarity(new_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    # Return list of dictionaries for easy rendering in HTML
    results = df.iloc[top_indices].copy()
    results['similarity'] = similarities[top_indices]
    return results.to_dict(orient='records')

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    query = ""
    if request.method == 'POST':
        query = request.form.get('user_query')
        if query:
            results = get_recommendations(query)
            
    return render_template('index.html', results=results, query=query)

if __name__ == '__main__':
    app.run(debug=True)