import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('Patient.csv')

# Remove duplicates based on the PatientID column
df_cleaned = df.drop_duplicates(subset=['PatientID'])

# Save the cleaned DataFrame to a new CSV file
df_cleaned.to_csv('Patient_cleaned.csv', index=False)

print("Duplicates removed and cleaned data saved to 'Patient_cleaned.csv'.")

import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('Doctor.csv')

# Remove duplicates based on the PatientID column
df_cleaned = df.drop_duplicates(subset=['DoctorID'])

# Save the cleaned DataFrame to a new CSV file
df_cleaned.to_csv('Doctor_cleaned.csv', index=False)

print("Duplicates removed and cleaned data saved to 'Doctor_cleaned.csv'.")

import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('Appointment.csv')

# Remove duplicates based on the PatientID column
df_cleaned = df.drop_duplicates(subset=['AppointmentID'])

# Save the cleaned DataFrame to a new CSV file
df_cleaned.to_csv('Appointment_cleaned.csv', index=False)

print("Duplicates removed and cleaned data saved to 'Appointment_cleaned.csv'.")


import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('Medical Procedure.csv')

# Remove duplicates based on the PatientID column
df_cleaned = df.drop_duplicates(subset=['ProcedureID'])

# Save the cleaned DataFrame to a new CSV file
df_cleaned.to_csv('Procedure_cleaned.csv', index=False)

print("Duplicates removed and cleaned data saved to 'Procedure_cleaned.csv'.")

import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('Billing.csv')

# Remove duplicates based on the PatientID column
df_cleaned = df.drop_duplicates(subset=['InvoiceID'])

# Save the cleaned DataFrame to a new CSV file
df_cleaned.to_csv('Billing_cleaned.csv', index=False)

print("Duplicates removed and cleaned data saved to 'Billing_cleaned.csv'.")

import sqlite3

# Connect to SQLite database
with sqlite3.connect('healthcare.db') as conn:
    cursor = conn.cursor()

    # Drop existing tables (if they exist)
    cursor.execute('DROP TABLE IF EXISTS Patients')
    cursor.execute('DROP TABLE IF EXISTS Doctors')
    cursor.execute('DROP TABLE IF EXISTS Appointments')
    cursor.execute('DROP TABLE IF EXISTS MedicalProcedure')
    cursor.execute('DROP TABLE IF EXISTS Billing')

    # Create Patients table
    cursor.execute('''
        CREATE TABLE Patients (
            PatientID INTEGER PRIMARY KEY,
            firstname TEXT,
            lastname TEXT,
            email TEXT
        )
    ''')

    # Create Doctors table
    cursor.execute('''
        CREATE TABLE Doctors (
            DoctorID INTEGER PRIMARY KEY,
            DoctorName TEXT,
            Specialization TEXT,
            DoctorContact TEXT
        )
    ''')

    # Create Appointments table
    cursor.execute('''
        CREATE TABLE Appointments (
            AppointmentID INTEGER PRIMARY KEY,
            Date TEXT,
            Time TEXT,
            PatientID INTEGER,
            DoctorID INTEGER,
            FOREIGN KEY (PatientID) REFERENCES Patients(PatientID),
            FOREIGN KEY (DoctorID) REFERENCES Doctors(DoctorID)
        )
    ''')

    # Create MedicalProcedure table
    cursor.execute('''
        CREATE TABLE MedicalProcedure (
            ProcedureID INTEGER PRIMARY KEY,
            ProcedureName TEXT,
            AppointmentID INTEGER,
            FOREIGN KEY (AppointmentID) REFERENCES Appointments(AppointmentID)
        )
    ''')

    # Create Billing table
    cursor.execute('''
        CREATE TABLE Billing (
            InvoiceID TEXT PRIMARY KEY,
            PatientID INTEGER,
            Items TEXT,
            Amount REAL,
            FOREIGN KEY (PatientID) REFERENCES Patients(PatientID)
        )
    ''')

print("SQLite database 'healthcare.db' created successfully.")

import sqlite3
import csv

# Connect to SQLite database
conn = sqlite3.connect('healthcare.db')
cursor = conn.cursor()

# Function to insert data from CSV into SQLite table
def insert_data_from_csv(csv_file, table_name):
    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row if present
        for row in reader:
            cursor.execute(f'INSERT INTO {table_name} VALUES ({", ".join("?" * len(row))})', row)

# Insert data into Patients table from CSV
insert_data_from_csv('Patient_cleaned.csv', 'Patients')

# Insert data into Doctors table from CSV
insert_data_from_csv('Doctor_cleaned.csv', 'Doctors')

# Insert data into Appointments table from CSV
insert_data_from_csv('Appointment_cleaned.csv', 'Appointments')

# Insert data into MedicalProcedure table from CSV
insert_data_from_csv('Procedure_cleaned.csv', 'MedicalProcedure')

# Insert data into Billing table from CSV
insert_data_from_csv('Billing_cleaned.csv', 'Billing')

# Commit changes and close connection
conn.commit()

print("Data inserted into SQLite database successfully.")


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

# Function to recommend procedures based on similarity
def recommend_procedures(procedure_name, cosine_sim=cosine_sim, procedures_df=procedures_df, top_n=5):
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
        print(f"No procedure found with name '{procedure_name}'.")
        return None

# Example usage: Recommend procedures similar to 'Knee Surgery'
recommendations = recommend_procedures('Kidney transplant')
if recommendations is not None:
    print(recommendations[['ProcedureID', 'ProcedureName']])

# Save the TF-IDF vectorizer and cosine similarity matrix to a pickle file
with open('tfidf_cosine_sim.pkl', 'wb') as f:
    pickle.dump((tfidf_vectorizer, cosine_sim), f)

print("TF-IDF vectorizer and cosine similarity matrix saved to 'tfidf_cosine_sim.pkl'.")


##################################


# You can now use tfidf_vectorizer and cosine_sim as needed
import streamlit as st
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# Load the TF-IDF vectorizer and cosine similarity matrix from the pickle file
with open('tfidf_cosine_sim.pkl', 'rb') as f:
    tfidf_vectorizer, cosine_sim = pickle.load(f)

# Function to connect to the SQLite database
def get_connection(db_name='healthcare.db'):
    return sqlite3.connect(db_name)

# Function to load the data from the SQLite database
def load_data():
    conn = get_connection()
    query = '''
        SELECT ProcedureID, ProcedureName
        FROM MedicalProcedure
    '''
    return pd.read_sql_query(query, conn)

# Function to initialize the TF-IDF Vectorizer and calculate cosine similarity matrix
def calculate_similarity(procedures_df):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(procedures_df['ProcedureName'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# Function to recommend procedures based on similarity
def recommend_procedures(procedure_name, cosine_sim, procedures_df, top_n=5):
    if procedure_name in procedures_df['ProcedureName'].values:
        idx = procedures_df.index[procedures_df['ProcedureName'] == procedure_name].tolist()[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_sim_scores = sim_scores[1:top_n+1]
        procedure_indices = [i[0] for i in top_sim_scores]
        return procedures_df.iloc[procedure_indices]
    else:
        return None

# Streamlit app
def main():
    st.title("Medical Procedure Recommender System")

    # Load data and calculate similarity matrix
    procedures_df = load_data()
    cosine_sim = calculate_similarity(procedures_df)

    # User input for procedure name
    procedure_name = st.text_input("Enter a medical procedure name:")

    if st.button("Recommend"):
        recommendations = recommend_procedures(procedure_name, cosine_sim, procedures_df)

        if recommendations is not None:
            st.write("Recommended procedures similar to '{}':".format(procedure_name))
            st.dataframe(recommendations[['ProcedureID', 'ProcedureName']])
        else:
            st.write("No procedure found with name '{}'.".format(procedure_name))

if __name__ == "__main__":
    main()
