from flask import Flask, render_template, Response, request, redirect, url_for, flash,session
import cv2
import numpy as np
import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import pickle
import pandas as pd
from gtts import gTTS
import os
import pyttsx3
from playsound import playsound
from translate import Translator
import sqlite3
from pygame import mixer  
from pydub import AudioSegment
from pydub.playback import play



app = Flask(__name__)
app.secret_key = "123"

body_language_class = ""
language_codes = {'tamil': 'ta', 'english': 'en', 'hindi': 'hi', 'malayalam':'ml', 'telugu': 'te'}


database="database.db"
con=sqlite3.connect(database)

con.execute("create table if not exists custom(pid integer primary key,name text,mail text)")
con.execute("create table if not exists result(pid integer primary key, user_name text, comment text)")


con.close()
values = []



@app.route('/')
def loginpage():
    return render_template('loginpage.html')

l=[]
@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == 'POST':
        try:
            name = request.form['name']
            l.append(name)
            password = request.form['password']
            con = sqlite3.connect("database.db")
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute("select * from custom where name=? and mail=?", (name, password))
            data = cur.fetchone()

            if data:
                session["name"] = data["name"]
                session["mail"] = data["mail"]
                return redirect("index")
            else:
                flash("Username and password Mismatch", "danger")

        except Exception as e:
            print(f"Error: {str(e)}")
            flash("Check Your Name And Password")

    return redirect(url_for("loginpage"))

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method=='POST':
        try:
            name=request.form['name']
            mail=request.form['mail']
            con=sqlite3.connect("database.db")
            cur=con.cursor()
            cur.execute("insert into custom(name,mail)values(?,?)",(name,mail))
            con.commit()
            flash("Record Added Successfully","success")
        except:
            flash("Error in Insert Operation","danger")
        finally:
            return redirect(url_for("loginpage"))
            con.close()

    return render_template('register.html')

@app.route('/save_text', methods=['POST'])
def save_text():
    global saved_text
    if request.method == 'POST':
        # Get the text entered by the user
        user_text = request.form['user_text']
        con=sqlite3.connect("database.db")
        cur=con.cursor()
        # Check if there's at least one element in the list l
        if l:
            # Access the last element of the list l
            user_name = l[-1]
            # Insert the user's name and the entered text into the database
            cur.execute("insert into result(user_name, comment) values (?, ?)", (user_name, user_text))
    
            con.commit()
            con.close()
        else:
            flash("No user logged in", "danger")

    return redirect(url_for('index'))



@app.route("/admin", methods=["GET","POST"])
def admin():
    if request.method == "POST":
        a = "admin"
        b = "admin"
        user = request.form.get("name")
        password = request.form.get("pass")
        if user == a and password == b:
            conn = sqlite3.connect(database)
            cur = conn.cursor()
            cur.execute("SELECT * FROM result")
            result = cur.fetchall()
            return render_template("result.html", result=result)
        else:
            return "password mismatch"
    else:
        # If the request method is not POST, render the admin login page
        return render_template("admin.html")




mp_drawing = mp.solutions.drawing_utils 
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh
holistic = mp_holistic.Holistic()
face_mesh = mp_face_mesh.FaceMesh()
holistic = mp_holistic.Holistic()


mp_drawing = mp.solutions.drawing_utils 
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
holistic = mp_holistic.Holistic()

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

with open('hand_sign.pkl', 'rb') as f:
    model = pickle.load(f)

detected_letters = []
current_letter_duration = 0
min_continuous_duration = 30  

    
@app.route('/index', methods=['GET', 'POST'])
def index():
   
        return render_template('index.html')

@app.route('/generate_frames', methods=['GET', 'POST'])
def generate_frames():
    global body_language_class
    
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            last_detected_letter = None
            while cap.isOpened():
                ret, frame = cap.read()

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = holistic.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                         mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                         mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                         )

                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                         mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                         mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                         )

                try:
                    pose = results.left_hand_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                    row = pose_row

                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]
                    print(body_language_class)
                    body_language_prob = model.predict_proba(X)[0]


                    
                    cv2.putText(image, 'CLASS'
                                , (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, body_language_class
                                , (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                    cv2.putText(image, 'PROB', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2)), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
          
            
                except Exception as e:
                    print(e)
            
                ret, jpeg = cv2.imencode('.jpg', image)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                            
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()



@app.route('/play_language', methods=['POST'])
def play_language():
    global body_language_class
    language = request.form['language_text']
    print(language)

    if language == 'tamil':
        translator = Translator(to_lang='ta')
    elif language == 'english':
        translator = Translator(to_lang='en')
    elif language == 'hindi':
        translator = Translator(to_lang='hi')
    elif language == 'malayalam':
        translator = Translator(to_lang='ml')
    elif language == 'telugu':
        translator = Translator(to_lang='te')
    else:
        return render_template("index.html")
    

    body_language_class_translated = translator.translate(body_language_class)
    if body_language_class_translated:
        speak = gTTS(text=body_language_class_translated, lang=language_codes[language])
        speak.save("translated_show.mp3")
        playsound('translated_show.mp3')
        print('name')
        if os.path.exists('translated_show.mp3'):
            os.remove('translated_show.mp3')
        else:
            print("The file does not exist.")

    else:
        print("Translated text is empty.")

    return render_template("index.html")
    



  
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(port=600)
