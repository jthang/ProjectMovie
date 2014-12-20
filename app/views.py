from flask import render_template
from app import app

@app.route('/')
@app.route('/index')
def index():
    user = {'nickname': 'Jens'}
    posts = [
        {
        'author': {'nickname': 'Wing'},
        'body': 'Beautiful day in Matrix today!'
        },
        {
        'author': {'nickname': 'Jens'},
        'body': 'Yes, I agree. I love data!'
        }
    ]
    return render_template('index.html',
                            title='Home',
                            user=user,
                            posts=posts)