from flask import Flask, redirect, url_for

# WSGI Application
app = Flask(__name__)


# Variable Rules and URL Building

@app.route('/')  # Parameter is url. Now it is homepage
def welcome():
    return "Any Message."


@app.route('/success/<int:score>')
def success(score):
    return "Person Passed. Score is " + str(score)


@app.route('/fail/<int:score>')
def fail(score):
    return "Person Failed. Score is " + str(score)


# Result checker
@app.route('/results/<int:marks>')
def results(marks):
    result = ''
    if marks < 50:
        result = 'fail'
    else:
        result = 'success'
    return redirect(url_for(result, score=marks))


if __name__ == '__main__':
    app.run(debug=True)
