
import streamlit as st
import pandas as pd
import sqlite3
import pickle

# Load the pickled objects
pickle_file = '/mnt/data/procedure_recommender.pkl'
with open(pickle_file, 'rb') as f:
    tfidf_vectorizer, cosine_sim, procedures_df = pickle.load(f)

def recommend_procedures(procedure_name, cosine_sim=cosine_sim, procedures_df=procedures_df, top_n=5):
    if procedure_name in procedures_df['ProcedureName'].values:
        idx = procedures_df.index[procedures_df['ProcedureName'] == procedure_name].tolist()[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_sim_scores = sim_scores[1:top_n+1]
        procedure_indices = [i[0] for i in top_sim_scores]
        return procedures_df.iloc[procedure_indices]
    else:
        st.write(f"No procedure found with name '{procedure_name}'.")
        return None

st.title("Healthcare Management System")

procedure_name = st.text_input("Enter a procedure name:", "Kidney transplant")
if procedure_name:
    recommendations = recommend_procedures(procedure_name)
    if recommendations is not None:
        st.write("Recommended procedures:")
        st.write(recommendations[['ProcedureID', 'ProcedureName']])
