FROM python:3.10-slim

# Set up a non-root user (Hugging Face requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory
WORKDIR $HOME/app

# Copy requirements and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY --chown=user . .

# Expose Hugging Face default port
EXPOSE 7860

# Run the application using gunicorn for production
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app", "--timeout", "120"]
