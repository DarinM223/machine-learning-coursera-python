from flask import Flask

# Set up web app
app = Flask(__name__)

# Import routes
from ex1 import routes
from ex2 import routes

