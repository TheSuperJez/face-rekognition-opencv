import face_recognition
import cv2

video_capture = cv2.VideoCapture(0)

#Obama
obama_image = face_recognition.load_image_file("data/obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

#Trump
trump_image = face_recognition.load_image_file("data/trump.jpeg")
trump_face_encoding = face_recognition.face_encodings(trump_image)[0]

# Esteban
esteban_image = face_recognition.load_image_file("data/esteban.jpg")
esteban_face_encoding = face_recognition.face_encodings(esteban_image)[0]


known_face_encodings = [
    obama_face_encoding,
    trump_face_encoding,
    esteban_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Trump",
    "Jorge Esteban"

]


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Obtener video
    ret, frame = video_capture.read()

    # Redimensionar video
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convertir imagen a RGB
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        # Encontrar todas las caras
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # buscar matches
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # si hay un match mostrar nombre
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # mostrar nombre
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # calcular dibujado
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # dibujar caja al rededor de cara
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # dibujar texto
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # mostrar imagen con caja y texto
    cv2.imshow('Video', frame)

    # q para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# liberar camara
video_capture.release()
cv2.destroyAllWindows()