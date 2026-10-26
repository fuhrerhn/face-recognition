import cv2

recognizer = cv2.face.LBPHFaceRecognizer.create()
recognizer.read("face-model.yml")
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_COMPLEX

# DAFTAR NAMA ANGGOTA KELOMPOK
# Indeks 0 diisi 'None' karena ID diawali dari angka 1
names = ['None', 'Farhan', 'Yoga', 'Vina'] 
colors = [
    (255, 255, 255), # Index 0: Putih (None / Unknown)
    (0, 255, 0),     # Index 1: Hijau (Farhan)
    (255, 0, 255),   # Index 2: Magenta/Pink (Yoga)
    (255, 255, 0)    # Index 3: Cyan/Kuning Kebiruan (Vina)
]

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:        
        # Prediksi ID dan tingkat jarak/confidence
        pred_id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # LBPH: Semakin kecil confidence, semakin mirip
        if confidence < 100 and pred_id < len(names):
            display_name = names[pred_id]
            box_color = colors[pred_id]
        else:
            display_name = "Unknown"
            box_color = (0, 0, 255)

        # Hitung persentase akurasi
        accuracy_str = f"{round(100 - confidence)}%"
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        # Tampilkan Nama di atas kotak
        cv2.putText(frame, str(display_name), (x + 5, y - 10), font, 0.8, box_color, 2)
        
        # Tampilkan Akurasi di bawah kotak
        cv2.putText(frame, accuracy_str, (x + 5, y + h + 25), font, 0.7, box_color, 2)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()