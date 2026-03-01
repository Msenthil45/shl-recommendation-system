import pandas as pd
from  sklearn.feature_extraction.text import TfidfVectorizer
from  sklearn.metrics.pairwise import cosine_similarity

# Load Excel
df = pd.read_excel("Gen_AI Dataset.xlsx")

df = df[['Query', 'Assessment_url']]
df = df.dropna()

df['Query'] = df['Query'].str.strip()
df['Assessment_url'] = df['Assessment_url'].str.strip()

# Remove duplicates
df = df.drop_duplicates()

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit on job descriptions
tfidf_matrix = vectorizer.fit_transform(df['Query'])

def recommend(new_query, top_k=3):
    new_vector = vectorizer.transform([new_query])
    similarities = cosine_similarity(new_vector, tfidf_matrix).flatten()
    
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    results = df.iloc[top_indices][['Assessment_url']]
    results = results.assign(similarity=similarities[top_indices])
    
    return results

# Test
test_query = """
Hiring Java developer who can collaborate with business teams.
Looking for 40-minute assessment.
"""

print(recommend(test_query))