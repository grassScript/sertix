import random
import time
import torch
import torch.nn as nn
import os
import discord
from discord.ext import commands
from discord import app_commands

class MarkovChain:
    def __init__(self):
        self.transitions = {}
        self.dataset_file = 'dataset_sertix.txt'
        self.load_dataset()  # Загружаем датасет при инициализации

    def load_dataset(self):
        """Загружает предложения из файла в Markov Chain."""
        if os.path.exists(self.dataset_file):
            with open(self.dataset_file, 'r', encoding='utf-8') as f:
                for line in f:
                    self.add_sentence(line.strip())

    def add_sentence(self, sentence):
        words = sentence.split()
        for i in range(len(words) - 1):
            if words[i] not in self.transitions:
                self.transitions[words[i]] = []
            self.transitions[words[i]].append(words[i + 1])
        if words[-1] not in self.transitions:
            self.transitions[words[-1]] = []

    def generate_sentence(self, start_word, length=8):
        if start_word not in self.transitions:
            return random.choice(list(self.transitions.keys()))
        
        result = [start_word]
        current_word = start_word
        for _ in range(length - 2):
            if self.transitions.get(current_word):
                current_word = random.choice(self.transitions[current_word])
                result.append(current_word)
            else:
                # Если нет переходов, выбираем случайное слово
                current_word = random.choice(list(self.transitions.keys()))
                result.append(current_word)
        
        return ' '.join(result)


class DiscordBot(nn.Module):
    def __init__(self):
        super(DiscordBot, self).__init__()
        self.markov_chain = MarkovChain()
        self.sentences = [
            "here your dataset"

        ]
        for sentence in self.sentences:
            self.markov_chain.add_sentence(sentence)

        # Инициализация переменной для хранения баллов
        self.scores = {}

        # Определение LSTM сети
        self.lstm = nn.LSTM(input_size=1, hidden_size=1024, num_layers=1, batch_first=True)
        self.fc = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

        # Добавление слоев, ускоряющих процесс отвечания
        self.dropout = nn.Dropout(p=0.4)
        self.relu = nn.ReLU()

        # Check if the model file exists before loading
        model_path = 'sertix.pth'
        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path))
        else:
            print(f"Warning: Model file '{model_path}' not found. Initializing with random weights.")

    def forward(self, input_sentence):
        words = input_sentence.split()
        start_word = random.choice(words) if words else random.choice(self.sentences)
        length = random.randint(3, 8)
        response = self.markov_chain.generate_sentence(start_word, length)
        
        # Добавление новых слов в Markov Chain
        self.markov_chain.add_sentence(input_sentence)
        
        return response if response else "Извините, я не могу сгенерировать ответ."

    def evaluate_response(self, response):
        response_tensor = torch.tensor([len(response.split())], dtype=torch.float32).unsqueeze(0).unsqueeze(2)
        h0 = torch.zeros(1, 1, 1024)
        c0 = torch.zeros(1, 1, 1024)
        out, _ = self.lstm(response_tensor, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.dropout(out)
        out = self.relu(out)
        score = self.sigmoid(out)
        return score.item()

    def update_score(self, username, feedback):
        if username not in self.scores:
            self.scores[username] = 0
        
        if feedback.lower() == "плохо":
            self.scores[username] -= 1
        elif feedback.lower() == "хорошо":
            self.scores[username] += 1

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='s', intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s)")
    except Exception as e:
        print(e)

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Добавление новых слов в Markov Chain
    bot_model.markov_chain.add_sentence(message.content)

    # Проверка на оценку от пользователя orange227
    if message.author.name == "orange227":
        if "плохо" in message.content.lower():
            bot_model.update_score(message.author.name, "плохо")
            await message.channel.send("я постараюсь быть лучше")
        elif "хорошо" in message.content.lower():
            bot_model.update_score(message.author.name, "хорошо")
            await message.channel.send("ок")

    await bot.process_commands(message)


@bot.tree.command(name="chat")
@app_commands.describe(query="Введите ваш запрос")
async def chat(interaction: discord.Interaction, query: str):
    await interaction.response.defer(thinking=True)
    
    best_response = None
    best_score = -1
    for _ in range(320):
        response = bot_model.forward(query)
        score = bot_model.evaluate_response(response)
        if score > best_score:
            best_score = score
            best_response = response
    
    if not best_response:
        best_response = "Извините, я не могу сгенерировать ответ."
    
    try:
        await interaction.followup.send(best_response)
    except discord.NotFound:
        print("Ошибка: взаимодействие уже истекло.")
    except Exception as e:
        print(f"Ошибка при отправке ответа: {e}")

bot_model = DiscordBot()
bot.run('your_bot_token')
