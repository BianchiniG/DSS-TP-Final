FROM germanb/dss-tp-final:latest

EXPOSE 5000

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

CMD python app.py