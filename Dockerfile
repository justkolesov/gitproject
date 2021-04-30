FROM python:3.8-slim-buster

WORKDIR usr/src/regression

 
#COPY ./data/regression.csv .
#COPY ./data/model_path .
#COPY ./data/model_save .
#COPY ./data/test_path.csv .
#COPY requirements.txt .
COPY . .

RUN pip install  -r requirements.txt


#COPY ./code/module.py .
#COPY ./code/dataloader.py .
#COPY ./code/test.py .
#COPY ./code/train.py .
#COPY ./code/test_functions.py .
 
RUN python ./code/train.py
#RUN python ./code/test.py  
#ENTRYPOINT ["python"]
#CMD ["dataloader.py" ]