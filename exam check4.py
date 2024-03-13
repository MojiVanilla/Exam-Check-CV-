import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
import cv2
import imutils
from imutils import contours

answer_key = ['E', 'A', 'C', 'B', 'C', 'B', 'A', 'C', 'E', 'B', 'C', 'A', 'D', 'B', 'A', 'A', 'B', 'C', 'B', 'E', 'C', 'D', 'D', 'D', 'A', 'E', 'B', 'A', 'B', 'E', 'D', 'C', 'D', 'D', 'E', 'B', 'D', 'E', 'C', 'D']

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    maxWidth = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    maxHeight = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def process_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    docCnt = None

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                docCnt = approx
                break

    paper = four_point_transform(image, docCnt.reshape(4, 2))
    warped = four_point_transform(gray, docCnt.reshape(4, 2))
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if w >= 10 and h >= 10 and 0.9 <= ar <= 1.1:
            questionCnts.append(c)

    questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
    mid_x = sum([cv2.boundingRect(c)[0] for c in questionCnts]) / len(questionCnts)
    left_column = [c for c in questionCnts if cv2.boundingRect(c)[0] < mid_x]
    right_column = [c for c in questionCnts if cv2.boundingRect(c)[0] >= mid_x]

    filled_threshold = 50
    question_counter = 1
    correct_count = 0

#Divide into 2 columns
    for column in [left_column, right_column]:
        for i in range(0, len(column), 5):
            cnts = contours.sort_contours(column[i:i+5])[0]
            bubbled = None

            for (j, c) in enumerate(cnts):
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(mask)

                if bubbled is None or total > bubbled[0]:
                    bubbled = (total, j)

            correct_option = answer_key[question_counter - 1]
            if bubbled is not None:
                (x, y, w, h) = cv2.boundingRect(cnts[bubbled[1]])
                center = (int(x + w / 2), int(y + h / 2))
                if chr(65 + bubbled[1]) == correct_option:
                    cv2.circle(paper, center, max(int(h/2), 20), (0, 255, 0), 4)  # Green circle for correct answers
                    correct_count += 1
                else:
                    cv2.circle(paper, center, max(int(h/2), 20), (0, 0, 255), 4)  # Red circle for incorrect answers

            question_counter += 1

    total_questions = len(answer_key)
    score = (correct_count / total_questions) * 100

    # Calculate the position for the text
    text = f"Score: {score:.2f}% | Correct: {correct_count}/{total_questions}"
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_org = (10, paper.shape[0] - 10 - baseline)

    # Annotate the score summary on the image
    cv2.putText(paper, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display the processed image with annotations
    cv2.imshow("Processed Image", paper)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            process_image(file_path)
        except Exception as e:
            messagebox.showerror("Error", "Failed to process the image\n" + str(e))
    else:
        messagebox.showinfo("Information", "No file selected")

root = tk.Tk()
root.title("Bubble Sheet Scanner")

upload_btn = tk.Button(root, text="Upload Image", command=upload_image)
upload_btn.pack(pady=20)

root.mainloop()
