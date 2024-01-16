FROM python:3.10-slim

ENV PYTHONUNBUFFERED 1
# Set the working directory to /app
WORKDIR /app

# Update the package lists and install dependencies
RUN apt-get update && \
    apt-get install -y libtesseract5 tesseract-ocr curl python3-opencv

RUN curl -LJO https://github.com/tesseract-ocr/tessdata/raw/main/fra.traineddata
RUN mv fra.traineddata /usr/share/tesseract-ocr/5/tessdata/

COPY . /app

RUN pip install --upgrade pip && \
    pip install -r requirements_legacy.txt

# Expose the port streamlit will run on
EXPOSE 8501

# Command to run on container start
CMD streamlit run app.py --server.address 0.0.0.0