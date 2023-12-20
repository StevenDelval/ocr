
___
## Usage
Install the dependencies:

```consol
pip install -r requirements_legacy.txt
```


## How to install tesseract-ocr on Windows
1. Run the installer (tesseract-ocr-w64-setup-5.3.3.20231005.exe) from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).
2. Choose the .traineddata file for the language you are interested in from this [GitHub repository](https://github.com/tesseract-ocr/tessdata/tree/main).
3. Move the .traineddata file to the path where you have installed Tesseract-OCR, more precisely, to '**yourpath**\Tesseract-OCR\tessdata'.

## How to install tesseract-ocr on Ubuntu
1. Run 
```consol
sudo apt install tesseract-ocr 
```
2. Choose the .traineddata file for the language you are interested in from this [GitHub repository](https://github.com/tesseract-ocr/tessdata/tree/main).
3. Move the .traineddata file to the path where you have installed Tesseract-OCR, 
```consol
mv ~/Téléchargements/fra.traineddata /usr/share/tesseract-ocr/4.00/tessdata
```

___
