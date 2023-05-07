# Must use a Cuda version 11+
FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git
RUN apt install -y python3-packaging

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# We add the banana boilerplate here
ADD server.py .

# Add your model weight files 
# (in this case we have a python script)
ADD download.py .
RUN python3 download.py


# Add your custom app code, init() and inference()
ADD app.py .
ADD instruct_pipeline.py .

EXPOSE 8000

CMD python3 -u server.py
