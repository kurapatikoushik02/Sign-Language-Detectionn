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
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



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

from tensorflow.keras.models import load_model

# Load the model
#model = load_model("model.h5")

detected_letters = []
current_letter_duration = 0
min_continuous_duration = 30  

    
@app.route('/index', methods=['GET', 'POST'])
def index():
   
        return render_template('index.html')

@app.route('/generate_frames', methods=['GET', 'POST'])
def generate_frames():
    global text
    
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

                    X_text = pd.DataFrame([row])
                    df = pd.read_csv('hand.csv')

                    X = df.drop('class', axis=1)  
                    y = df['class']  

                    label_encoder = LabelEncoder()
                    y = label_encoder.fit_transform(y)

                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=1234)

                    model = Sequential([
                        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                        Dense(64, activation='relu'),
                        Dense(32, activation='relu'),
                        Dense(4, activation='softmax')  
                    ])

                    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

                    model.fit(X_train, y_train, epochs=3, batch_size=32, validation_split=0.2)
                    body_language_class = model.predict(X_text)
                    print(body_language_class)
                    class_labels = ["happy", "sad", "super", "fine"]

                    # Get the predicted class for each sample
                    predicted_classes = np.argmax(body_language_class, axis=1)

                    # Display the predicted class
                    cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    for i, predicted_class in enumerate(predicted_classes):
                        text = f"Predicted class: {class_labels[predicted_class]}"
                        cv2.putText(image, text, (10, 30*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)          
            
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
    global text
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
    

    body_language_class_translated = translator.translate(text)
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
    app.run(port=700)
