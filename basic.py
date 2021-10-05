from flask import Flask, render_template
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix, accuracy_score
import io



app = Flask(__name__)


@app.route('/home')
def home():
	dataset_test=pd.read_csv('Google_Stock_Price_Test.csv')
	X_test=dataset_test.iloc[:-1,1:2].values 
	y_test=dataset_test.iloc[1:,1:2].values
	ann = tf.keras.models.load_model('ann.h5')
	y_pred = ann.predict(X_test)
	fig = Figure()
	plt = fig.add_subplot(1, 1, 1)
	plt.plot(y_test, color = 'red', label = 'Real Google Stock Price')
	plt.plot(y_pred, color = 'blue', label = 'Predicted Google Stock Price')
	plt.set_title('Google Stock Price Prediction')
	plt.set_xlabel('Time')
	plt.set_ylabel('Google Stock Price')
	# plt.legend()
	# plt.show()
	# Convert plot to PNG image
	pngImage = io.BytesIO()
	FigureCanvas(fig).print_png(pngImage)
	# Encode PNG image to base64 string
	pngImageB64String = "data:image/png;base64,"
	pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
	return render_template('after.html', image=pngImageB64String)
    

if __name__ == "__main__":
    app.run(debug=True)
