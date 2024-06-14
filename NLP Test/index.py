import json
import os
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from langdetect import detect
import joblib

# OpenWeatherMap API
API_KEY = '0076d51a87207fc4278e35f6e201abcb'  # OpenWeatherMap API anahtarınızı buraya ekleyin

# Data seti yükle
with open('./dataset.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Verileri çekmek
texts = []
labels = []
for intent in data['intents']:
    for pattern in intent['patterns']['en']:  # İngilizce patternler kullanılıyor
        texts.append(pattern)
        labels.append(intent['tag'])
    for pattern in intent['patterns']['tr']:  # Türkçe patternler kullanılıyor
        texts.append(pattern)
        labels.append(intent['tag'])

def train_model():
    # Naive Bayes modelini oluştur
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(texts, labels)
    # Modeli kaydet
    joblib.dump(model, 'chatbot_model.pkl')
    print("Model trained and saved.")
    return model

def load_model():
    if os.path.exists('chatbot_model.pkl'):
        model = joblib.load('chatbot_model.pkl')
        print("Model loaded.")
    else:
        model = train_model()
    return model

def get_coordinates(city):
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if len(data) > 0:
            lat = data[0]['lat']
            lon = data[0]['lon']
            return lat, lon
    return None, None

def get_weather(city):
    lat, lon = get_coordinates(city)
    if lat is not None and lon is not None:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        if response.status_code == 200:
            weather_data = response.json()
            if 'weather' in weather_data and 'main' in weather_data:
                weather_desc = weather_data['weather'][0]['description']
                temperature = weather_data['main']['temp']
                return f"{city} şehrinde hava şu anda {weather_desc} ve sıcaklık {temperature}°C."
            else:
                return "Üzgünüm, hava durumu bilgilerini alamadım. Lütfen şehir adını kontrol edin."
        else:
            return "Üzgünüm, hava durumu bilgilerini alamadım. Lütfen şehir adını kontrol edin."
    else:
        return "Üzgünüm, belirtilen şehrin koordinatlarını bulamadım."

def get_response(user_input, model, context):
    # Kullanıcı girdisinin dilini tespit et
    language = detect(user_input)
    prediction = model.predict([user_input])[0]
    
    if context == "ask_location":
        return get_weather(user_input), ""
    
    for intent in data['intents']:
        if intent['tag'] == prediction:
            if language == 'tr':
                return intent['responses']['tr'][0], intent.get('context_set', "")
            else:
                return intent['responses']['en'][0], intent.get('context_set', "")
    return "I'm not sure how to help with that.", ""

def chat():
    # Modeli yükle
    model = load_model()
    context = ""

    print("Merhaba! Ben senin asistanınım. Her şeyi bana sorabilirsin. (Çıkış yapmak için 'quit')")
    while True:
        user_input = input("Sen: ")
        if user_input.lower() == 'quit':
            print("Bot: Bay bay!")
            break
        response, context = get_response(user_input, model, context)
        print(f"Bot: {response}")

def retrain_model():
    return train_model()

# Botu başlat
chat()
