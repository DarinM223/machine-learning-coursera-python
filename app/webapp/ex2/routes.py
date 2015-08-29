import scipy as sp

from app.webapp import app
from flask import jsonify, request
from app.webapp.ex2 import theta
from app.ex2.logistic_regression import sigmoid

@app.route('/ex2/logistic_regression', methods=['GET'])
def logistic_regression():
    """
    Predicts the probability that a student will be admitted
    to a university based on how well he did on two exams
    Params:
    exam1: Integer score
    exam2: Integer score
    """
    exam1 = int(request.args.get('exam1'))
    exam2 = int(request.args.get('exam2'))
    prob = sigmoid(sp.asmatrix([1, exam1, exam2]).dot(theta))
    return jsonify({
        'probability_accepted': prob[0,0]
    })

