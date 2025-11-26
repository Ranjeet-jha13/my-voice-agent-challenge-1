import sqlite3

conn = sqlite3.connect("bank_fraud.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM transactions")
print(cursor.fetchall())
conn.close()