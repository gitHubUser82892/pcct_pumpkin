from cvzone.FaceDetectionModule import FaceDetector
import cv2

# Initialize camera and detector
cap = cv2.VideoCapture(0)
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)  # 0 = short-range model

while True:
    success, img = cap.read()
    if not success:
        break

    # Detect faces
    img, bboxs = detector.findFaces(img)

    # Draw bounding boxes and center points
    if bboxs:
        for bbox in bboxs:
            x, y, w, h = bbox['bbox']
            cv2.putText(img, "Face", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display
    cv2.imshow("Pumpkin FaceDetector", img)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

