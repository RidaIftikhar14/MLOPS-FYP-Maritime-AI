# Use the official Python base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY ./app/requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app directory into the container
COPY ./app .

# Expose the desired port
EXPOSE 8000

# Define the command to run the Flask application
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=8000"]