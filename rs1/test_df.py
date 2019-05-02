from flask import Flask, redirect, url_for
import pandas as pd
app = Flask(__name__)

@app.route('/')
def hello_world():
   df = pd.read_csv(r'C:\Users\ASUS\Desktop\ML\ScrapedSets\electro.csv')
   print df.iloc[100]
   return 'Ended'


if __name__ == '__main__':
   app.run(debug = True)