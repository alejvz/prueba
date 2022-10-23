#crear calculadora con flask

from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sumar', methods=['POST'])
def sumar():
    if request.method == 'POST':
        num1 = request.form['num1']
        num2 = request.form['num2']
        resultado = int(num1) + int(num2)
        return render_template('index.html', resultado=resultado)

if __name__ == '__main__':
    app.run(debug=True)
