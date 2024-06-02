# 
FROM python:3.9

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install -r /code/requirements.txt --use-deprecated=legacy-resolver

# 
COPY . /code/app

# 
CMD ["fastapi", "run", "app/main.py", "--port", "80"]