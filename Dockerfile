FROM ubuntu

RUN apt-get update

RUN apt-get install -y python3

RUN apt install -y python3-pip

RUN pip3 install pandas

RUN pip3 install scikit-learn

RUN pip3 install catboost

RUN pip3 install streamlit==1.20.0

COPY . /apps

WORKDIR /apps

CMD ["streamlit", "run", "main.py"]
