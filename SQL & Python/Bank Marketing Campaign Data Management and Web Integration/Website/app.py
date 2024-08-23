from flask import Flask, render_template
import psycopg2

app = Flask(__name__)

# Connect to PostgreSQL database
conn = psycopg2.connect(
    host="localhost",
    database="bankmarketing",
    user="postgres",
    password="p!nkyP@ndu7"
)

# Define routes for each table
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/client_campaign_details')
def client_campaign_details():
    cur = conn.cursor()
    cur.execute("SELECT * FROM client_campaign_details")
    rows = cur.fetchall()
    print(rows)
    return render_template('client_campaign_details.html', campaigns=rows)

@app.route('/client_lastcontact_details')
def client_lastcontact_details():
    cur = conn.cursor()
    cur.execute("SELECT * FROM client_lastcontact_details")
    rows = cur.fetchall()
    return render_template('client_lastcontact_details.html', rows=rows)

@app.route('/client_job_details')
def client_job_details():
    cur = conn.cursor()
    cur.execute("SELECT * FROM client_job_details")
    rows = cur.fetchall()
    return render_template('client_job_details.html', rows=rows)

@app.route('/client_details')
def client_details():
    cur = conn.cursor()
    cur.execute("SELECT * FROM client_details")
    rows = cur.fetchall()
    return render_template('client_details.html', rows=rows)

@app.route('/client_loan_details')
def client_loan_details():
    cur = conn.cursor()
    cur.execute("SELECT * FROM client_loan_details")
    rows = cur.fetchall()
    return render_template('client_loan_details.html', rows=rows)

@app.route('/client_socioeconomic_details')
def client_socioeconomic_details():
    cur = conn.cursor()
    cur.execute("SELECT * FROM client_socioeconomic_details")
    rows = cur.fetchall()
    return render_template('client_socioeconomic_details.html', rows=rows)

# # Close database connection
# conn.close()

if __name__ == '__main__':
    app.run(debug=True,port=5001)
