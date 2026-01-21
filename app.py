import gradio as gr
import pandas as pd
import pickle

with open("insurance_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_charges(age, sex, bmi, children, smoker, region):
    input_data = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }])
    predicted_charge = model.predict(input_data)[0]
    return f"Predicted Insurance Charges: ${predicted_charge:,.2f}"

inputs = [
    gr.Number(label="Age", value=30),
    gr.Radio(choices=["male", "female"], label="Sex"),
    gr.Slider(minimum=10, maximum=50, step=0.1, label="BMI", value=25.0),
    gr.Slider(minimum=0, maximum=5, step=1, label="Children", value=0),
    gr.Radio(choices=["yes", "no"], label="Smoker"),
    gr.Dropdown(
        choices=["southeast", "southwest", "northeast", "northwest"],
        label="Region"
    )
]

app = gr.Interface(
    fn=predict_charges,
    inputs=inputs,
    outputs="text",
    title="Medical Insurance Cost Predictor",
    description="Predict annual insurance charges based on personal information."
)

if __name__ == "__main__":
    app.launch( share = True)