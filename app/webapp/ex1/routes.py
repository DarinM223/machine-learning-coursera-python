import scipy as sp

from app.webapp import app
from flask import jsonify, request
from app.webapp.ex1 import theta, normal_theta, mu, sigma

@app.route('/ex1/gradient_descent', methods=['GET'])
def gradient_descent():
    """
    Predicts the price of a house using gradient descent
    Params:
    area: Square Feet
    bedrooms: Number of bedrooms
    """
    area = float(request.args.get('area'))
    bedrooms = float(request.args.get('bedrooms'))
    house = sp.array([[1.0, area, bedrooms]])
    # Normalize the features
    house[0, 1:] = (house[0, 1:] - mu) / sigma
    price = house.dot(theta)
    return jsonify({
        'predicted_price': price[0,0]
    })


@app.route('/ex1/normal_function', methods=['GET'])
def normal_function():
    """
    Predicts the price of a house using the normal equation
    area: Square Feet
    bedrooms: Number of bedrooms
    """
    area = float(request.args.get('area'))
    bedrooms = float(request.args.get('bedrooms'))
    house = sp.array([[1.0, area, bedrooms]])
    # Normalize the features
    house[0, 1:] = (house[0, 1:] - mu) / sigma
    price = house.dot(normal_theta)
    return jsonify({
        'predicted_price': price[0,0]
    })

