# Gunicorn configuration file
# This file is automatically detected by Gunicorn when starting the app.

# Increase worker timeout to prevent Render from killing the worker
# during cold starts (TensorFlow initialization, Numba JIT compilation)
timeout = 180

# Use a single sync worker to minimize memory usage on the 512MB free tier
workers = 1

# Limit memory usage via threads
threads = 2

# Keep alive to handle connections gracefully
keepalive = 5
