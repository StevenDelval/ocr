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

def resize_image(image):
    """
    Resize the input image to a fixed size of (1946, 1080).

    Parameters:
    - image: Input image (OpenCV format)

    Returns:
    - Resized image
    """
    # Resize the image to the fixed size
    resized_image = cv2.resize(image, (460, 800))

    return resized_image




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


def image_brightness(img):
    """
    Process the input image by converting color space, applying gamma correction,
    and applying binary thresholding.

    Parameters:
    - img: Input image (OpenCV format)

    Returns:
    - Processed image
    """
    # Convert color space from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Apply gamma correction
    invGamma = 1.0 / 0.3
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gray = cv2.LUT(gray, table)

    # Apply binary thresholding
    _, thresh1 = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)

    return thresh1


def process_image(img):
    
    thresh1 = image_brightness(img)
    
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
    out = resize_image(out)
    

    return out




# Define a function to trim and save an image
def trim_and_save(region_key, region_points, image):
    x, y, w, h = cv2.boundingRect(np.array(region_points))
    cropped_image = image[y:y+h, x:x+w]
    cv2.imwrite(f"./img/{region_key}_cropped.jpg", cropped_image)



def image2text(cropped_image_path):
    text = ''
    
    cropped_image = cv2.imread(cropped_image_path)

    text = pytesseract.image_to_string(cropped_image, config=custom_config)

    if len(text)!=0:
        return text
    else:
        print("no text detected")

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

        img_array = np.array(image)
        processed_img = process_image(img_array)
        for region_key, region_points in regions_dict.items():
            trim_and_save(region_key, region_points, processed_img)
        dict_var = {}
        for region_key, value in regions_dict.items():
            cropped_image_path = f"./img/{region_key}_cropped.jpg"
            # cropped_image = cv2.imread(cropped_image_path)
            word_found = image2text(cropped_image_path)
            dict_var[region_key]=word_found.strip()
        
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