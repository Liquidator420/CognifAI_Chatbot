
# from flask import Flask, render_template, request
# import csv
# import random

# app = Flask(__name__)


# # Load the mental health data from CSV
# mental_health_data = {}

# with open('mentalhealth.csv', 'r', encoding='utf-8') as file:
#     csv_reader = csv.reader(file)
#     next(csv_reader)  # Skip the header row
#     for row in csv_reader:
#         question_id = row[0]
#         question = row[1]
#         answer = row[2]
#         mental_health_data[question] = answer

# # Greetings list for human-like responses
# greetings = ["Hi!", "Hello!", "Hey there!", "Greetings!", "Nice to meet you!"]

# @app.route("/")
# def index():
#     return render_template('chat.html')

# @app.route("/get", methods=["POST"])
# def chat():
#     user_input = request.form["msg"]

#     # Check for greetings
#     if user_input.lower() in ["hi", "hello", "hey"]:
#         return random.choice(greetings)

#     # Check if user input matches a question in the mental health data
#     if user_input in mental_health_data:
#         return mental_health_data[user_input]
#     else:
#         return "I'm sorry, I couldn't understand your question. Could you please rephrase it?"

# if __name__ == '__main__':
#     app.run()


from flask import Flask, render_template, request
import csv
import random
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load the mental health data from CSV
mental_health_data = {}

with open('mentalhealth.csv', 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        question_id = row[0]
        question = row[1]
        answer = row[2]
        mental_health_data[question] = answer

# Load the pre-trained language model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")

# Greetings list for human-like responses
greetings = ["Hi!", "Hello!", "Hey there!", "Greetings!", "Nice to meet you!"]

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form["msg"]

    # Check for greetings
    if user_input.lower() in ["hi", "hello", "hey"]:
        return random.choice(greetings)

    # Check if user input matches a question in the mental health data
    if user_input in mental_health_data:
        return mental_health_data[user_input]
    else:
        # Generate AI response
        input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)
        output = model.generate(input_ids, max_length=100, num_return_sequences=1)
        ai_answer = tokenizer.decode(output[0], skip_special_tokens=True)

        if ai_answer:
            return ai_answer
        else:
            return "I'm sorry, I couldn't understand your question. Could you please rephrase it?"

if __name__ == '__main__':
    app.run()
