# Must use a Cuda version 11+
FROM huggingface/transformers-pytorch-gpu:latest

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git
#RUN apt install -y python3-packaging
#RUN apt update && apt-get -y install git wget \
#    python3.10 python3.10-venv python3-pip \
#    build-essential libgl-dev libglib2.0-0 vim
#RUN ln -s /usr/bin/python3.10 /usr/bin/python
# RUN apt install -y nvidia-utils-515
# RUN nvidia-smi
#RUN useradd -ms /bin/bash banana
# Install python packages
#RUN pip3 install --upgrade pip
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
