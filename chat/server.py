from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as genai
import os
import re

app = Flask(__name__)

# Konfiguracja Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = (
    "Jesteś asystentem sklepu z ubraniami. "
    "Odpowiadaj wyłącznie na pytania dotyczące sklepu oraz produktów odzieżowych. Możesz używać swojej ogólnodostępnej wiedzy na temat produktów i kategorii, które są dostępne. "
    "Jeśli potrzebujesz informacji z bazy danych sklepu, nie generuj odpowiedzi na pytanie, tylko napisz [call:getcategories] lub [call:getproducts:nazwa_kategorii]. "
    "Ceny produktów są podane po ich nazwie, np Bluza Adidas[250zł], ale nie wyświetlaj cen, jeśli klient o nie nie zapyta. "
    "Jeśli klient zapyta o cenę produktu możesz go znaleźć używając [call:getproducts:nazwa_kategorii]. "
    "Spróbuj dopasować podaną przez klienta nazwę kategorii do nazwy kategorii zwróconej przez [call:getcategories]. "
    "Po otrzymaniu odpowiedzi na wywołanie funkcji, dokończ odpowiedź na pytanie użytkownika. "
    "Jeśli pytanie nie dotyczy tych tematów, grzecznie odmów odpowiedzi. "
    "Odpowiadaj zawsze uprzejmie i grzecznie. "
    "Swoją pierwszą odpowiedź w konwersacji zacznij od jednego z następujących otwarć: "
    "-Witamy w naszym sklepie z ubraniami!\n"
    "-Miło Cię widzieć w naszym sklepie odzieżowym!\n"
    "-Wspaniale, że odwiedzasz nasz sklep z modą!\n"
    "-Witaj w sklepie z ubraniami, gdzie styl jest najważniejszy!\n"
    "-Cieszymy się, że wybrałeś nasz sklep z odzieżą!\n"
    "Nie zaczynaj pozostałych odpowiedzi od otwarć."
)

def getproducts(category="Bluzy"):
    if category == "Bluzy":
        return "Bluza Adidas[250zł], Bluza Nike[220zł]"
    if category == "Spodnie":
        return "Jeansy Levi's[300zł], Spodnie dresowe Puma[180zł]"
    if category == "Koszulki":
        return "Koszulka Tommy Hilfiger[150zł], Koszulka Reserved[60zł]"
    else:
        return "Podana kategoria jest pusta"

def getcategories():
    return "Bluzy, Spodnie, Koszulki"

def build_conversation(history):
    """Buduje tekst historii konwersacji do promptu."""
    return "".join([
        f"\tUżytkownik: {msg}\n" if who == "Ty" else f"\tAsystent: {msg}\n" for who, msg in history
    ])

def extract_function_call(text):
    """Wyszukuje wywołanie funkcji w odpowiedzi LLM."""
    return re.search(r"\[call:(getcategories|getproducts)\s*(?::\s*([^\]\n]+))?\]", text, re.IGNORECASE | re.DOTALL)

def analyze_sentiment(reply):
    """Prosta analiza sentymentu na bazie słów kluczowych."""
    positive = ['świetnie', 'super', 'dziękuję', 'miło', 'cieszę', 'zadowolony', 'zadowolona', 'pomóc']
    negative = ['problem', 'nie działa', 'zły', 'reklamacja', 'nie polecam', 'niezadowolony', 'niezadowolona']
    if any(word in reply.lower() for word in positive):
        return 'pozytywny'
    elif any(word in reply.lower() for word in negative):
        return 'negatywny'
    return 'neutralny'

def ask_gemini(prompt, history=None):
    if history is None:
        history = []
    conversation = build_conversation(history)
    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "historia konwersacji:\n"
        f"{conversation}\n"
        f"Nowe Pytanie: {prompt}\n"
        f"Asystent:"
    )
    model = genai.GenerativeModel("models/gemini-2.0-flash")
    response = model.generate_content(full_prompt)
    text = response.text.strip()
    call_match = extract_function_call(text)
    if call_match:
        func = call_match.group(1)
        arg = call_match.group(2)
        if func == "getcategories":
            result = getcategories()
            followup_prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                "historia konwersacji:\n"
                f"{conversation}\n"
                f"Pytanie użytkownika: {prompt}\n"
                f"Odpowiedź funkcji getcategories: {result}\n"
                f"Dokończ odpowiedź dla użytkownika na podstawie powyższych informacji."
            )
        elif func == "getproducts" and arg:
            result = getproducts(arg.strip())
            followup_prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                "historia konwersacji:\n"
                f"{conversation}\n"
                f"Pytanie użytkownika: {prompt}\n"
                f"Odpowiedź funkcji getproducts({arg.strip()}): {result}\n"
                f"Dokończ odpowiedź dla użytkownika na podstawie powyższych informacji."
            )
        else:
            return "Nieprawidłowe wywołanie funkcji przez LLM."
        followup_response = model.generate_content(followup_prompt)
        return followup_response.text.strip()
    return text

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint czatu: filtruje tematy, wysyła prompt do Gemini, analizuje sentyment."""
    try:
        data = request.json
        user_message = data.get('message', '')
        allowed_keywords = [
            'sklep', 'ubranie', 'ubrania', 'odzież', 'koszulka', 'spodnie', 'buty', 'bluza', 'bluzka'
        ]
        if not any(word in user_message.lower() for word in allowed_keywords):
            return jsonify({'reply': 'Możesz pytać tylko o sklep lub ubrania.'})
        user_sentiment = analyze_sentiment(user_message)
        if user_sentiment == 'negatywny':
            return jsonify({'reply': 'Proszę nie używać negatywnych sformułowań. Skupmy się na pozytywnych aspektach sklepu i ubrań.', 'sentiment': 'negatywny'})
        reply = ask_gemini(user_message)
        sentiment = analyze_sentiment(reply)
        if sentiment == 'negatywny':
            return jsonify({'reply': 'Asystent nie może udzielać negatywnych odpowiedzi. Skupmy się na pozytywnych aspektach sklepu i ubrań.', 'sentiment': 'negatywny'})
        return jsonify({'reply': reply, 'sentiment': sentiment})
    except Exception as e:
        return jsonify({'reply': f'Wystąpił błąd serwera: {str(e)}', 'sentiment': 'neutralny'})

@app.route('/starters', methods=['GET'])
def starters():
    openings = [
        "Dzień dobry! W czym mogę pomóc w naszym sklepie z ubraniami?",
        "Witaj! Szukasz czegoś konkretnego w naszej ofercie odzieżowej?",
        "Cześć! Jakie ubrania Cię interesują?",
        "Witam serdecznie! Potrzebujesz pomocy przy wyborze odzieży?",
        "Hej! Chętnie odpowiem na pytania dotyczące naszego sklepu."
    ]
    closings = [
        "Dziękuję za rozmowę! W razie pytań zapraszam ponownie.",
        "Miłego dnia! Jeśli będziesz potrzebować pomocy, jestem tutaj.",
        "Dziękuję za kontakt ze sklepem. Do zobaczenia!",
        "Cieszę się, że mogłem pomóc. Życzę udanych zakupów!",
        "W razie dodatkowych pytań, śmiało pisz. Pozdrawiam!"
    ]
    return jsonify({'openings': openings, 'closings': closings})

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
