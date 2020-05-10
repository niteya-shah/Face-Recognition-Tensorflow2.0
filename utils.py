import cv2

def take_photo():
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 32:
            cam.release()

            cv2.destroyAllWindows()
            return frame

def show_photo(img):
    cv2.imshow("photo",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
