# Use an official Python image as the base
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire Django project into the container
COPY . /app/

# Change to the correct directory where `manage.py` is located
WORKDIR /app/text_sentiment_analyzer

# Collect static files
RUN python manage.py collectstatic --noinput

# Expose port 8000 for Django
EXPOSE 8000

# Start the application using Gunicorn
CMD ["gunicorn", "text_sentiment_analyzer.wsgi:application", "--bind", "0.0.0.0:8000"]
