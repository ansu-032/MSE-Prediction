from flask  import Flask,render_template,request, redirect, url_for
import pandas as pd
import joblib


app=Flask(__name__)

# Load the encoders
country_encoder = joblib.load('country_encoder.pkl')
balance_encoder = joblib.load('balance_encoder.pkl')
product_encoder = joblib.load('product_encoder.pkl')
month_encoder = joblib.load('month_encoder.pkl')

# Load the trained model
model = joblib.load('model.pkl')

data=pd.read_csv("Cleaned1.csv")

#----------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html') 

# About section
@app.route('/about')
def home():
        return render_template('descrption.html')  

# Render the home page on successful login
@app.route('/demandforecasting')
def demandforecasting():
    return render_template('demandforecasting.html')

@app.route('/data_analysis')
def data_analysis():
    return render_template('data_analysis.html')

@app.route('/Customizable_Inputs')
def Customizable_Inputs():
    return render_template('Customizable_Inputs.html')

#contact page
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        message = request.form['message']
        return redirect(url_for('index'))
    return render_template('contact.html')

#Signup page
@app.route('/signupform',methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        company = request.form['company']
        phone = request.form['phone']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        return redirect(url_for('index'))
    return render_template('signupform.html') 

#log in page
@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        return redirect(url_for('index'))
    return render_template('login.html')
    
# FAQ page
@app.route('/containt')
def containt():
    return render_template('containt.html') 

#------------------------------------------------------------------------------------------------

#Prediction
@app.route('/predict',methods=['GET','POST'])
def predict():
    Country=sorted(data['Country'].unique())
    Balance=sorted(data['Balance'].unique())
    Product=sorted(data['Product'].unique())
    Month=sorted(data['Month'].unique())
    if request.method == 'POST':
        predict=request.form['predict']
        return render_template('predict.html',predict=predict,Country=Country,Balance=Balance,Product=Product,Month=Month)
    return render_template('predict.html',Country=Country,Balance=Balance,Product=Product,Month=Month)    

@app.route('/price',methods=['GET','POST'])
def price_pred():
    Country=request.form.get('Country')
    Balance=request.form.get('Balance')
    Product=request.form.get('Product')
    Month=request.form.get('Month')
    Date=request.form.get('Date')

    #Create dataframe
    input_df = pd.DataFrame([[Country,Balance,Product,Month,Date]], columns=['Country', 'Balance', 'Product', 'Month', 'Date'])

    # Handle unseen labels
    def handle_unseen_labels(encoder, value, default_value):
        if value not in encoder.classes_:
            return default_value
        else:
            return value

    input_df['Country'] = input_df['Country'].apply(lambda x: handle_unseen_labels(country_encoder, x, country_encoder.classes_[0]))
    input_df['Balance'] = input_df['Balance'].apply(lambda x: handle_unseen_labels(balance_encoder, x, balance_encoder.classes_[0]))
    input_df['Product'] = input_df['Product'].apply(lambda x: handle_unseen_labels(product_encoder, x, product_encoder.classes_[0]))
    input_df['Month'] = input_df['Month'].apply(lambda x: handle_unseen_labels(month_encoder, x, month_encoder.classes_[0]))
    
    # Encode the input data
    input_df['Country'] = country_encoder.transform(input_df['Country'])
    input_df['Balance'] = balance_encoder.transform(input_df['Balance'])
    input_df['Product'] = product_encoder.transform(input_df['Product'])
    input_df['Month'] = month_encoder.transform(input_df['Month'])

    # Extract the features
    input_features = input_df[['Country', 'Balance', 'Product', 'Month', 'Date']].values
    predicted_value = model.predict(input_features)
    return str(predicted_value[0].round())


if __name__ =='__main__':
    app.run(debug=True,port=7000)