from flask import Flask
from flask import Flask, flash, redirect, render_template, request, session, abort, Response, jsonify  # NOQA
import os
import mysql.connector
from frontend import Gen_frame


mydb = mysql.connector.connect(
      host="localhost",
      user="touchdown",
      passwd="safer1234",
      database="FR"
)
current = []
mycursor = mydb.cursor(buffered=True, dictionary=True)
mydb.autocommit = True
image_gen = Gen_frame(mydb)

app = Flask(__name__)
app.secret_key = os.urandom(12)

IMG_path = os.path.join('static', 'images')


@app.route('/')
def home():
    if not session.get('logged_in'):
        return render_template('login.html')


@app.route('/add_student')
def add_student():
    return render_template('add_students.html')


@app.route('/submit_details', methods=['POST'])
def submit_details():
    print(request.form)
    return render_template('/display')


@app.route('/login', methods=['POST'])
def do_admin_login():
    if ((request.form['password'] == 'password') and
       (request.form['username'] == 'admin')):
        session['logged_in'] = True
        return redirect('/display')
    else:
        flash('wrong password!')
    return home()


@app.route('/video_feed')
def video_feed():
    return Response(image_gen.gen_image(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/update_details')
def details():
    mycursor.reset()
    mycursor.execute("Select * from current")
    t = mycursor.fetchone()
    if t:
        mycursor.execute("delete from current")
        mycursor.execute("Select * from `user` where `img_path`='" + t['path']
                         + "'")
        text = dict(mycursor.next())
        if text is not None:
            return jsonify(text)
    return {"name": "Not Found", "reg_id": "Not Found",
            "School": "Not Found", "Mobile_number": "Not Found",
            "Block": "Not Found", "img_path": ""}

@app.route("/display")
def display(text=None):
    return render_template('display.html', text=[1, 1, 1, 1, 1])


if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True, host='0.0.0.0', port=4002)