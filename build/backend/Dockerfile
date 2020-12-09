FROM germanb/dss-tp-final:latest

EXPOSE 5001

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

CMD python backend_app.py