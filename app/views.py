from flask import Flask, jsonify, render_template, request
from app import app

@app.route('/_add_numbers')
def add_numbers():
    a = request.args.get('a', 0, type=int)
    b = request.args.get('b', 0, type=int)
    return jsonify(result=a + b)

@app.route('/')
@app.route('/index')
def index():
    # user = {'nickname': 'Jens'}
    return render_template('index.html',
                            title='Project Movie')

