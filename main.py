from flask import Flask, render_template, request
from werkzeug.utils import secure_filename


from assignment import predict_audio, classes

app = Flask(__name__)

@app.route('/')
def home():
   return render_template('index.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      result = predict_audio(f.filename, classes)

      return render_template('output.html', result = result)
		
if __name__ == '__main__':
   app.run(debug = True)