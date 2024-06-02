# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
ADD requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Add the current directory contents into the container at /app
ADD . /app

# Make port 80 available to the world outside this container
EXPOSE 7860

# Run app.py when the container launches
CMD ["python", "daoc_casting_optimizer.py"]