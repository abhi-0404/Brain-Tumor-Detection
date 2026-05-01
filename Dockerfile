# Use the official Python 3.10 image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Hugging Face runs containers as a non-root user (user ID 1000). 
# We need to create this user and give them ownership of the app directory.
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

# Expose the specific port Hugging Face looks for
EXPOSE 7860

# The command to start your Flask application
CMD ["python", "web_app/app.py"]