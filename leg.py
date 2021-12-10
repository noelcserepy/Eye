def find_face(img, eye_pos):
    face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, 1.2, 8)

    faces_focus = []
    faces_area = []

    for (x, y, w, h) in faces:
        focus_x = x + w // 2
        focus_y = y + h // 3
        area = w * h
        faces_focus.append([focus_x // w, focus_y // h])
        faces_area.append(area)

    # If face is detected: track face
    if len(faces_area) != 0:
        i = faces_area.index(max(faces_area))
        img, current_eye_pos = draw_iris(img, faces_focus[i][0], faces_focus[i][1])
        return img, current_eye_pos
    # If face is not detected: stay put
    else:
        img, current_eye_pos = draw_iris(img, eye_pos[0], eye_pos[1])
        return img, current_eye_pos
