from flask import Flask

# WSGI Application
app = Flask(__name__)


@app.route('/')  # Parameter is url. Now it is homepage
def welcome():
    return "Any Message."


@app.route('/members')  # Parameter is url. Now it is /members
def members():
    return "Any Message. Members"


if __name__ == '__main__':
    app.run(debug=True)
