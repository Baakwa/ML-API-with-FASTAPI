#Create a base image for Docker container
FROM python:3.8.10

# Create a working directory named 'app'
WORKDIR /app

#Copy requirements.txt files into temporary folder in container
COPY requirements.txt /tmp/requirements.txt

#Pip install packages in requirements.txt file
RUN python -m pip install -r /tmp/requirements.txt

#Copy everything(all files and folders) into the container working directory
COPY . /app

#Indicate this port to expose outside the container
EXPOSE 8000

#Run FastAPI app
CMD ['uvicorn','main:app','--host','0.0.0.0','--port','8000']

