# Electricity Consumption Forecasting Using LSTM

## Dataset
UCI Individual Household Electric Power Consumption:
https://dev.uci-ics-mlr-prod.aws.uci.edu/dataset/235/individual%2Bhousehold%2Belectric%2Bpower%2Bconsumption

Place `household_power_consumption.txt` inside `data/`.

## Run
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Model
LSTM with 64 and 32 units, dropout 0.2, Adam optimizer, MSE loss.
The previous 24 hourly observations predict the next hour.
Evaluation uses MAE and RMSE.

## Deployment
Upload the project to GitHub and deploy `app.py` using Streamlit Community Cloud.
