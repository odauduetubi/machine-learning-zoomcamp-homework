
from flask import Flask

app = Flask('ping')

#Next, we add a decorator to our function. This decorator tells Flask what URL should trigger our function.

@app.route('/ping', methods=['GET'])
def ping():
    return "pong!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)