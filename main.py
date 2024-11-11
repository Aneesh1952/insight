from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Load the model and encoders
with open("user_preference_model2.pkl", "rb") as f:
    model_data = pickle.load(f)
model = model_data['model']
label_encoders = model_data['encoders']

# Define input data model for prediction
class PredictionInput(BaseModel):
    age: int
    gender: str
    region: str
    interest_tags: str
    avg_session_dur: float
    ctr: float
    pages_viewed: int

# Define the required columns for the dataset
REQUIRED_COLUMNS = {'Subscription_Status', 'Age', 'Interest_Tags'}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Ensure the uploads directory exists
    os.makedirs("uploads", exist_ok=True)

    # Save the uploaded file
    file_location = f"uploads/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    try:
        # Load dataset
        data = pd.read_csv(file_location)
    except Exception:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid CSV.")

    # Check for required columns
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"The uploaded file is missing required columns: {', '.join(missing_columns)}"
        )

    # Ensure 'Age' is numeric
    if not pd.api.types.is_numeric_dtype(data['Age']):
        data['Age'] = pd.to_numeric(data['Age'], errors='coerce')

    # Drop rows with NaN values in required columns
    data = data.dropna(subset=REQUIRED_COLUMNS)

    # Generate visualizations
    visualizations = generate_visualizations(data)

    return {"visualizations": visualizations}

def generate_visualizations(data):
    # Ensure the visualizations directory exists
    os.makedirs("visualizations", exist_ok=True)

    visualizations = []

    # Subscription Status Distribution (Bar Plot)
    plt.figure(figsize=(10, 5))
    subscription_counts = data['Subscription_Status'].value_counts()
    sns.barplot(x=subscription_counts.index, y=subscription_counts.values)
    plt.title("Subscription Status Distribution")
    plt.xlabel("Subscription Status")
    plt.ylabel("Count")
    plt.savefig("visualizations/subscription_status.png")
    visualizations.append("/visualizations/subscription_status.png")
    plt.close()

    # Subscription Status Distribution (Pie Chart)
    plt.figure(figsize=(8, 8))
    subscription_counts.plot.pie(autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set3"))
    plt.title("Subscription Status Proportion")
    plt.ylabel("")
    plt.savefig("visualizations/subscription_status_pie.png")
    visualizations.append("/visualizations/subscription_status_pie.png")
    plt.close()

    # Age Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(data['Age'], kde=True)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.savefig("visualizations/age_distribution.png")
    visualizations.append("/visualizations/age_distribution.png")
    plt.close()

    # Interest Tags Word Cloud
    interest_text = " ".join(data['Interest_Tags'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(interest_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("visualizations/interest_tags_wordcloud.png")
    visualizations.append("/visualizations/interest_tags_wordcloud.png")
    plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(12, 8))
    correlation = data.corr()
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Correlation Heatmap")
    plt.savefig("visualizations/correlation_heatmap.png")
    visualizations.append("/visualizations/correlation_heatmap.png")
    plt.close()

    return visualizations

# Static file serving
app.mount("/visualizations", StaticFiles(directory="visualizations"), name="visualizations")

# Helper function to safely encode categorical inputs
def safe_transform(encoder, value, default=0):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        return default  # Use a default value for unseen categories

# Prediction endpoint
@app.post("/predict/")
async def predict_subscription_status(input_data: PredictionInput):
    # Encode categorical inputs with error handling
    gender_encoded = safe_transform(label_encoders['Gender'], input_data.gender)
    region_encoded = safe_transform(label_encoders['Region'], input_data.region)
    interest_encoded = safe_transform(label_encoders['Interest_Tags'], input_data.interest_tags)

    # Prepare feature vector for prediction
    user_features = [[
        input_data.age,
        float(gender_encoded),
        float(region_encoded),
        float(interest_encoded),
        input_data.avg_session_dur,
        input_data.ctr,
        input_data.pages_viewed
    ]]

    # Make prediction
    prediction = model.predict(user_features)
    subscription_status = label_encoders['Subscription_Status'].inverse_transform(prediction)

    return JSONResponse(content={"predicted_subscription_status": subscription_status[0]})
