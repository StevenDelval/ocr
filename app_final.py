import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re
from PyPDF2 import PdfWriter, PdfReader
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import os

# Specify the directory path
directory_path = "./img"

# Check if the directory exists
if not os.path.exists(directory_path):
    # Create the directory if it does not exist
    os.makedirs(directory_path)


# Adding custom options
custom_config = r'--oem 2 --psm 12 -l fra+eng'

def image2text(image):
    text = ''
    

    text = pytesseract.image_to_string(image, config=custom_config)

    if len(text)!=0:
        return text
    else:
        return "no text detected"

def biggest_contour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest

def image2text(image):
    text = ''
    

    text = pytesseract.image_to_string(image, config=custom_config)

    if len(text)!=0:
        return text
    else:
        return "no text detected"

def biggest_contour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest
def order_points(pts):
    '''Rearrange coordinates to order:
       top-left, top-right, bottom-right, bottom-left'''
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect.astype('int').tolist()


def find_dest(pts):
    (tl, tr, br, bl) = pts
    # Finding the maximum width.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Finding the maximum height.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # Final destination co-ordinates.
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]

    return order_points(destination_corners)



def scan(img):
    # Resize image to workable size
    # dim_limit = 1080
    # max_dim = max(img.shape)
    # if max_dim > dim_limit:
    #     resize_scale = dim_limit / max_dim
    #     img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)
    
    # img = cv2.resize(img, (1400, 1800))
    # Create a copy of resized original image for later use

    orig_img = img.copy()
    # Repeated Closing operation to remove text from the document.
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)
    # GrabCut
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (20, 20, img.shape[1] - 20, img.shape[0] - 20)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    # Edge Detection.
    canny = cv2.Canny(gray, 0, 200)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Finding contours for the detected edges.
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Keeping only the largest detected contour.
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # Detecting Edges through Contour approximation.
    # Loop over the contours.
    if len(page) == 0:
        return orig_img
    for c in page:
        # Approximate the contour.
        epsilon = 0.02 * cv2.arcLength(c, True)
        corners = cv2.approxPolyDP(c, epsilon, True)
        # If our approximated contour has four points.
        if len(corners) == 4:
            break
    # Sorting the corners and converting them to desired shape.
    corners = sorted(np.concatenate(corners).tolist())
    # For 4 corner points being detected.
    corners = order_points(corners)

    destination_corners = find_dest(corners)

    # Getting the homography.
    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
    # Perspective transform using homography.
    final = cv2.warpPerspective(orig_img, M, (destination_corners[2][0], destination_corners[2][1]),
                                flags=cv2.INTER_LINEAR)
    final = cv2.resize(final, (1200, 1600))
    return final

regions_dict = {
    'points_N_immatriculation':[(31, 35), (31, 46), (100, 49), (101, 35)],
    'points_titre_document': [(113, 0), (114, 15), (339, 15), (336, 0)],
    'points_d_2': [(38, 242), (38, 262), (359, 262), (359, 242)],
    'points_E': [(288, 276), (289, 293), (412, 295), (412, 275)],
    'points_Date_de_1er_immarticulation': [(145, 37), (146, 50), (228, 51), (226, 36)],
    'points_d_1' : [(38, 232), (39, 246), (203, 245), (203, 229)],
    'points_d_3' : [(36, 279), (37, 293), (209, 290), (210, 278)],
    'points_j_1' : [(135, 320), (136, 332), (203, 331), (204, 319)],
    'points_Date_I' : [(31, 468), (32, 482), (117, 482), (116, 465)]
}



def main():
    st.title("Image OCR Streamlit App")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        
        
        # Print the results
        new_dict_var = {}
        new_dict_var['points_N_immatriculation'] = st.text_input("N°immatriculation",dict_var['points_N_immatriculation'].upper(),key="'points_N_immatriculation'")
        new_dict_var['points_d_1'] = st.text_input("D.1",dict_var['points_d_1'].upper(),key="points_d_1")
        new_dict_var['points_d_2'] = st.text_input("D.2",dict_var['points_d_2'].upper(),key="points_d_2")
        new_dict_var['points_d_3'] = st.text_input("D.3",dict_var['points_d_3'].upper(),key="points_d_3")
        new_dict_var['points_E'] = st.text_input("E",dict_var['points_E'].upper(),key="points_E")
        new_dict_var['points_j_1'] = st.text_input("J.1",dict_var['points_j_1'].upper(),key="points_j_1")
        new_dict_var['points_Date_de_1er_immarticulation'] = st.text_input("Date de 1er immarticulation",dict_var['points_Date_de_1er_immarticulation'].upper(),key="points_Date_de_1er_immarticulation")
        new_dict_var['points_Date_I'] = st.text_input("Date immarticulation actuelle",dict_var['points_Date_I'].upper(),key="points_Date_I")
        button = st.button("Confirmer")
        if button:
            buffer = BytesIO()
            p = canvas.Canvas(buffer, pagesize=A4)
            dict_pos = {
                'points_N_immatriculation':[50, 705],
                'points_d_1': [50, 652],
                'points_d_2': [50, 630],
                'points_d_3': [225, 652],
                'points_E': [50, 610],
                'points_j_1': [227, 610],
                'points_Date_de_1er_immarticulation': [445, 705],
                'points_Date_I' : [307, 705],
            }
            for region_key in dict_pos.keys():
                if "Date" not in region_key:
                    p.drawString(*dict_pos[region_key], new_dict_var[region_key].upper(),charSpace=2)
                else:
                    p.drawString(*dict_pos[region_key], new_dict_var[region_key].upper().replace("/",""),charSpace=8.5)
            p.showPage()
            p.save()

            #move to the beginning of the StringIO buffer
            buffer.seek(0)
            newPdf = PdfReader(buffer)

            # #######DEBUG NEW PDF created#############
            # pdf1 = buffer.getvalue()
            # open('pdf1.pdf', 'wb').write(pdf1)
            #########################################
            # read your existing PDF
            existingPdf = PdfReader(open('cerfa_13750-07.pdf', 'rb'))
            output = PdfWriter()
            # add the "watermark" (which is the new pdf) on the existing page
            page = existingPdf.pages[0]
            page.merge_page(newPdf.pages[0])
            output.add_page(page)
            output_bytes = BytesIO()
            output.write(output_bytes)
            

            st.download_button(label="Certificat",
                    data=output_bytes,
                    file_name="Certificat.pdf")
            




if __name__ == "__main__":
    main()