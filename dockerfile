From python:3.7-slim

WORKDIR /Test
COPY ./Test
RUN pip install ==trusted-host pypi.python.org -r requirements.txt
EXPOSE 8080
ENV NAME TestWorld

CMD ["pytho", "Fashion.py"]
