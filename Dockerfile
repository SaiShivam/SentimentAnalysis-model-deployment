FROM python:3.7-slim

# copy local code to container image
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install dependencies
RUN pip install tensorflow==2.1.0 tensorflow-datasets Flask gunicorn healthcheck

# Run the flask service on container startup
#CMD exec gunicorn --bind  :$PORT --workers 1 --threads 8 SAgunicorn:app
EXPOSE 5002
ENTRYPOINT ["python3"]

CMD ["SAFlask.py"]