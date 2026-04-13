import sqlite3
import pandas as pd

def create_connection():
    conn = sqlite3.connect("patients.db", check_same_thread=False)
    return conn

def create_table():
    conn = create_connection()
    cursor = conn.cursor()

    # PATIENT TABLE
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age INTEGER,
            pregnancies INTEGER,
            glucose REAL,
            bp REAL,
            skin REAL,
            insulin REAL,
            bmi REAL,
            dpf REAL,
            risk TEXT,
            probability REAL
        )
    """)

    # AUDIT LOG TABLE
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            age INTEGER,
            pregnancies INTEGER,
            glucose REAL,
            bp REAL,
            skin REAL,
            insulin REAL,
            bmi REAL,
            dpf REAL,
            risk TEXT,
            probability REAL,
            timestamp TEXT
        )
    """)

    conn.commit()
    conn.close()


def insert_patient(data):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO patients (
            age, pregnancies, glucose, bp, skin, insulin, bmi, dpf, risk, probability
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)
    conn.commit()
    conn.close()


def insert_audit(data):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO audit_log (
            username, age, pregnancies, glucose, bp, skin, insulin, bmi, dpf, risk, probability, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)
    conn.commit()
    conn.close()


def get_all_patients():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients")
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_audit_logs():
    conn = create_connection()
    df = pd.read_sql_query("SELECT * FROM audit_log ORDER BY id DESC", conn)
    conn.close()
    return df