from flask import render_template
from app import app

@app.route('/')
@app.route('/index')
def index():
    # user = {'nickname': 'Jens'}
    return render_template('index.html',
                            title='Project Movie')