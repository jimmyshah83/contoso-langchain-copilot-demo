from flask import Flask
# pip install python-dotenv
# pip install openai==1.2.0

app = Flask(__name__)

@app.route('/')
def hello_world():
    print("Hello, World!")
    return 'Hello, World!'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
