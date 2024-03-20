from django.shortcuts import render, redirect
from . models import UserPersonalModel
from . forms import UserPersonalForm, UserRegisterForm
from django.contrib.auth import authenticate, login,logout
from django.contrib import messages
import numpy as np
import joblib


def Landing_1(request):
    return render(request, '1_Landing.html')

def Register_2(request):
    form = UserRegisterForm()
    if request.method =='POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            user = form.cleaned_data.get('username')
            messages.success(request, 'Account was successfully created. ' + user)
            return redirect('Login_3')

    context = {'form':form}
    return render(request, '2_Register.html', context)


def Login_3(request):
    if request.method =='POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('Home_4')
        else:
            messages.info(request, 'Username OR Password incorrect')

    context = {}
    return render(request,'3_Login.html', context)

def Home_4(request):
    return render(request, '4_Home.html')

def Teamates_5(request):
    return render(request,'5_Teamates.html')

def Domain_Result_6(request):
    return render(request,'6_Domain_Result.html')

def Problem_Statement_7(request):
    return render(request,'7_Problem_Statement.html')
    

def Per_Info_8(request):
    if request.method == 'POST':
        fieldss = ['firstname','lastname','age','address','phone','city','state','country']
        form = UserPersonalForm(request.POST)
        if form.is_valid():
            print('Saving data in Form')
            form.save()
        return render(request, '4_Home.html', {'form':form})
    else:
        print('Else working')
        form = UserPersonalForm(request.POST)    
        return render(request, '8_Per_Info.html', {'form':form})
    
    
Model1 = joblib.load('C:/Users/lenovo/Music/MAIN_PROJECT/CODE/DEPLOYMENT/PROJECT/APP/GENERATION.pkl')
Model2 = joblib.load('C:/Users/lenovo/Music/MAIN_PROJECT/CODE/DEPLOYMENT/PROJECT/APP/RADIATION.pkl')    
  
def Deploy_9(request): 
    if request.method == "POST":
        int_features = [x for x in request.POST.values()]
        int_features = int_features[1:]
        print(int_features)
        final_features = [np.array(int_features, dtype=float)]
        print(final_features)
        prediction = Model1.predict(final_features)
        print(prediction)
        output = prediction[0]
        print(output)        
        return render(request, '9_Deploy.html', {'prediction_text': F"THE GENERATION OF POWER IS {output} KW"})
    else:
        return render(request, '9_Deploy.html')


def Deploy_10(request): 
    if request.method == "POST":
        int_features = [x for x in request.POST.values()]
        int_features = int_features[1:]
        print(int_features)
        final_features = [np.array(int_features, dtype=float)]
        print(final_features)
        prediction = Model2.predict(final_features)
        print(prediction)
        output = prediction[0]
        print(output)        
        return render(request, '10_Deploy.html', {'prediction_text1': F"THE RADIATIONS IS {output}" })
    else:
        return render(request, '10_Deploy.html')    
    
def Per_Database_10(request):
    models = UserPersonalModel.objects.all()
    return render(request, '10_Per_Database.html', {'models':models})

def Logout(request):
    logout(request)
    return redirect('Login_3')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from django.shortcuts import render
import io
import base64

def forcast(request):
    if request.method == "POST":
        username = request.POST.get('START')
        password = request.POST.get('END')
        data = pd.read_csv("C:/Users/lenovo/Music/MAIN_PROJECT/CODE/DEPLOYMENT/PROJECT/APP/energy.csv")
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        daily_data = data.resample('D').sum()
        train_size = int(0.8 * len(daily_data))
        train_data, test_data = daily_data.iloc[:train_size], daily_data.iloc[train_size:]
        order = (1, 1, 1)  
        seasonal_order = (1, 1, 1, 7)  
        model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
        fit_model = model.fit(disp=False)
        forecast_start = pd.to_datetime(username)
        forecast_end = pd.to_datetime(password)
        forecast_index = pd.date_range(start=forecast_start, end=forecast_end, freq='D')
        forecast_steps = len(forecast_index)
        forecast = fit_model.get_forecast(steps=forecast_steps)
        forecast_values = forecast.predicted_mean
        forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecasted_Sales': forecast_values})
        forecast_df.set_index('Date', inplace=True)  
        print(forecast_df)
        forecast_df.to_csv("SARIMAX_FILE.csv", index=True)
        plt.figure(figsize=(12, 6))
        """plt.plot(daily_data.index, daily_data['Sales'], label='Actual')"""
        plt.plot(forecast_index, forecast_values, label='Forecasted Energy', linestyle='dashed')
        plt.xlabel('Date')
        plt.ylabel('Energyconsumption')
        plt.title('ENERGY CONSUMPTION FORCAST IN SARIMAX')
        plt.legend()
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png')
        img_bytes.seek(0)
        img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
        plt.close()

        return render(request, 'forcast.html', {'prediction_image_base64': img_base64})
    else:
        return render(request, 'forcast.html')
