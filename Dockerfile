FROM ubuntu:oracular
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -yq install python3-pip
COPY requirements.txt /model/
RUN pip3 install -r /model/requirements.txt
COPY model.py /model/
COPY tester.py /model/
COPY trainer.py /model/
COPY trained_model.pt /model/
CMD /model/model.py --input=/data/test.csv --output=/data/aki.csv
