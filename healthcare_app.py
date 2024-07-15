import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('healthcare.db')
cursor = conn.cursor()

# Drop the table if it already exists
cursor.execute('DROP TABLE IF EXISTS MedicalProcedure')

# Create the MedicalProcedure table
cursor.execute('''
    CREATE TABLE MedicalProcedure (
        ProcedureID INTEGER PRIMARY KEY,
        ProcedureName TEXT
    )
''')

# Insert sample data into the MedicalProcedure table
sample_data = [
    (1, 'Knee Surgery'),
    (2, 'Hip Replacement'),
    (3, 'Cardiac Bypass'),
    (4, 'Kidney Transplant'),
    (5, 'Appendectomy')
]

cursor.executemany('INSERT INTO MedicalProcedure (ProcedureID, ProcedureName) VALUES (?, ?)', sample_data)

# Commit the changes and close the connection
conn.commit()
conn.close()

print("MedicalProcedure table created and sample data inserted.")
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Connect to SQLite database
conn = sqlite3.connect('healthcare.db')

# Query the MedicalProcedure table
query = '''
    SELECT ProcedureID, ProcedureName
    FROM MedicalProcedure
'''
procedures_df = pd.read_sql_query(query, conn)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the TF-IDF vectorizer on procedure names
tfidf_matrix = tfidf_vectorizer.fit_transform(procedures_df['ProcedureName'])

# Calculate cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Save the TF-IDF Vectorizer and cosine similarity matrix to a pickle file
with open('procedure_recommendation_model.pkl', 'wb') as file:
    pickle.dump((tfidf_vectorizer, cosine_sim), file)

print("Model saved to 'procedure_recommendation_model.pkl'.")
import streamlit as st
import pandas as pd
import pickle
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity

# Load the model
with open('procedure_recommendation_model.pkl', 'rb') as file:
    tfidf_vectorizer, cosine_sim = pickle.load(file)

# Connect to SQLite database
conn = sqlite3.connect('healthcare.db')

# Query the MedicalProcedure table
query = '''
    SELECT ProcedureID, ProcedureName
    FROM MedicalProcedure
'''
procedures_df = pd.read_sql_query(query, conn)

# Function to recommend procedures based on similarity
def recommend_procedures(procedure_name, top_n=5):
    # Find index of procedure_name if it exists
    if procedure_name in procedures_df['ProcedureName'].values:
        idx = procedures_df.index[procedures_df['ProcedureName'] == procedure_name].tolist()[0]

        # Get pairwise similarity scores with all procedures
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort procedures based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get top-n similar procedures
        top_sim_scores = sim_scores[1:top_n+1]  # Exclude self and take top-n

        # Get procedure indices
        procedure_indices = [i[0] for i in top_sim_scores]

        # Return top-n similar procedures
        return procedures_df.iloc[procedure_indices]
    else:
        st.write(f"No procedure found with name '{procedure_name}'.")
        return None

# Streamlit App
st.title('Medical Procedure Recommendation System')

procedure_name = st.text_input('Enter the name of a medical procedure:')
top_n = st.number_input('Number of recommendations:', min_value=1, max_value=10, value=5)

if st.button('Recommend'):
    recommendations = recommend_procedures(procedure_name, top_n)
    if recommendations is not None:
        st.write('Top recommendations:')
        for index, row in recommendations.iterrows():
            st.write(f"{row['ProcedureName']} (Procedure ID: {row['ProcedureID']})")
