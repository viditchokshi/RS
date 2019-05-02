from flask import Flask, render_template, request, redirect, url_for
import rs_fun as rsf 
import pandas as pd

app = Flask(__name__)

@app.route('/search', methods=['POST'])
def run_script():
	temp = []
	recommended_data = []
	data = pd.DataFrame()
	strr = request.form['sc']
	mydata = rsf.prep()
	indexes = rsf.get_index(mydata, strr)
	cs = rsf.compute(subset = mydata)
	for index in indexes:
		temp_data = rsf.get_recommendations(index=index ,cosine_sim = cs, subset = mydata)
		data = data.append(rsf.pproc(temp_data), ignore_index = True)
	data = data.drop_duplicates()
	# for i in data:
	# 	for j in i:
	# 		recommended_data.append(j)
	# product_names = []
	# final_data = []
	# for data in recommended_data:
	# 	if not data['name'] in product_names:
	# 		product_names.append(data['name'])
	# 		final_data.append(data)
	return render_template('rs.html', ds1 = data, ds2 = data)
		#.sort_values('suppliertotalreview', ascending= False))



@app.route('/')
def index():
	return render_template('rs_index.html')


if __name__ == '__main__':
   app.run(debug = True)

