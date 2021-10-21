# Integrate HTML with Flask
# HTTP get and post

from flask import Flask, redirect, url_for, render_template, request

# WSGI Application
app = Flask(__name__)


# Variable Rules and URL Building

@app.route('/')  # Parameter is url. Now it is homepage
def welcome():
    return render_template('index.html')


@app.route('/success/<int:score>')
def success(score):
    return render_template('result.html', result='PASS', marks=score)


@app.route('/fail/<int:score>')
def fail(score):
    return render_template('result.html', result='Fail', marks=score)


# Result checker
@app.route('/results/<int:marks>')
def results(marks):
    result = ''
    if marks < 50:
        result = 'fail'
    else:
        result = 'success'
    return redirect(url_for(result, score=marks))


@app.route('/submit', methods=['POST', 'GET'])
def submit():
    total_score = 0
    if request.method == 'POST':
        science = float(request.form['science'])
        maths = float(request.form['maths'])
        c = float(request.form['c'])
        data_science = float(request.form['datascience'])
        total_score = (science + maths + c + data_science) / 4
    res = ''
    if total_score >= 50:
        res = 'success'
    else:
        res = 'fail'
    return redirect(url_for(res, score=total_score))


if __name__ == '__main__':
    app.run(debug=True)
