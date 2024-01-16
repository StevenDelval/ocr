import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re

# Adding custom options
custom_config = r'--oem 2 --psm 12 -l fra+eng'

def biggestRectangle(contours):
    max_area = 0
    indexReturn = -1
    for index in range(len(contours)):
        i = contours[index]
        area = cv2.contourArea(i)
        if area > 100:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.1 * peri, True)
            if area > max_area:
                max_area = area
                indexReturn = index
    return indexReturn

def process_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    invGamma = 1.0 / 0.3
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gray = cv2.LUT(gray, table)
    ret, thresh1 = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    indexReturn = biggestRectangle(contours)
    hull = cv2.convexHull(contours[indexReturn])

    mask = np.zeros_like(img)
    cv2.drawContours(mask, contours, indexReturn, 255, -1)
    out = np.zeros_like(img)
    out[mask == 255] = img[mask == 255]

    (y, x, _) = np.where(mask == 255)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out = img[topy:bottomy + 1, topx:bottomx + 1, :]

    return out

def extract_chars(regex, boxes, text):
    """
    Extract the first text based on the provided regex and coordinates.

    Parameters:
    - regex: Regular expression pattern to match.
    - boxes: The total number of OCR results.
    - text: The OCR result dictionary containing 'left', 'top', 'width', 'height', and 'text'.

    Returns:
    - The first text that matches the specified criteria.
    """
    following_chars = []
    liste_d2 = []
    is_following_d2 = False
    target_y_d2 = 0  # Set an initial value for target_y

    # Process OCR results
    for i in range(boxes):
        (x, y, w, h, chara) = (text['left'][i], text['top'][i], text['width'][i], text['height'][i], text["text"][i])

        # Check if the coordinates match the region of interest ('D.2')
        if re.search(regex, chara):
            target_y_d2 = y
            is_following_d2 = True
            liste_d2.append((x, y, w, h, chara))
        elif is_following_d2 and target_y_d2 - 10 <= y <= target_y_d2 + 10:
            if chara.strip():  # Check if chara is not empty after stripping whitespace
                following_chars.append(chara)
                is_following_d2 = False  # Stop capturing characters after the first non-null character

    # Extract the first text from the following_chars list
    first_text_d2 = following_chars[0] if following_chars else None

    return first_text_d2


def main():
    st.title("Image OCR Streamlit App")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        img_array = np.array(image)
        processed_img = process_image(img_array)

        st.subheader("Processed Image")
        st.image(processed_img, caption="Processed Image.", use_column_width=True)

        text = pytesseract.image_to_data(Image.fromarray(processed_img), output_type='dict',config=custom_config)
        liste_d2 = []
        # following_chars = []
        target_y_d2 = None
        is_following_d2 = False
        N_immatriculation = None
        dates = []
        E_chara = []
        print(text)
        boxes = len(text['level'])
        # Process OCR results
        for i in range(boxes):
            (x, y, w, h, chara) = (text['left'][i], text['top'][i], text['width'][i], text['height'][i], text["text"][i])

            # Check if the coordinates match the region of interest ('N. d'immatriculation')
            if N_immatriculation is None and re.search(r'[A-Z]{2}-\d{3}-[A-Z]{2}', chara):
                N_immatriculation = chara

            # Check if the coordinates match the region of interest (dates)
            if re.search(r'\b\d{2}/\d{2}/\d{4}\b', chara):
                dates.append(chara)


        result_d2 = extract_chars(r'^[dD0]\.2', boxes, text)
        result_E = extract_chars(r'\bE\b', boxes, text)
        result_D_3 = extract_chars(r'^[dD0]', boxes, text)


        # Print the results
        st.write(f"D.2 : {result_d2}")
        st.write(f"E. : {result_E}")
        st.write(f"D.3 : {result_D_3}")
        st.write(f"N. d'immatriculation  : {N_immatriculation}")
        st.write(f"Date de 1er immarticulation : {dates[0]}")
        st.write(f"Visite avant le  : {dates[1]}")
        st.write(f"Date d'immarticulation actuelle  : {dates[2]}")
        st.write(f"D.2 : {result_d2[-1]}")

        st.text_input("D2",result_d2,key="result_d2")

if __name__ == "__main__":
    main()