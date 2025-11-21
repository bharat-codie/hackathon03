"""
Simple Chatbot (simple_chatbot.py)

Features:
- Two modes: rule-based (simple canned responses) and retrieval-based (TF-IDF + cosine similarity)
- Can load a folder of text files as a knowledge base (KB) and answer by finding the most similar file
- Easy to run locally

Requirements:
- Python 3.8+
- Optional: scikit-learn (for TF-IDF and cosine similarity). If not installed, the script falls back to a basic substring matching retrieval.

Install (recommended):
    pip install scikit-learn

Usage:
    python simple_chatbot.py
    python simple_chatbot.py --api
    python simple_chatbot.py --test

Drop text files (.txt) into a folder named "kb" (same dir) to use retrieval mode.

"""

from pathlib import Path
import sys
import tempfile
import unittest

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def load_kb_texts(kb_folder="kb"):
    p = Path(kb_folder)
    texts = []
    names = []
    if not p.exists():
        return texts, names
    for f in sorted(p.glob("*.txt")):
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            try:
                text = f.read_text(encoding="latin-1")
            except Exception:
                text = ""
        texts.append(text)
        names.append(f.name)
    return texts, names


class RetrievalResponder:
    def __init__(self, kb_texts, kb_names=None):
        self.kb_texts = kb_texts
        self.kb_names = kb_names or [f"doc_{i}" for i in range(len(kb_texts))]
        if SKLEARN_AVAILABLE and kb_texts:
            try:
                self.vectorizer = TfidfVectorizer(stop_words='english')
                self.tfidf = self.vectorizer.fit_transform(kb_texts)
            except Exception:
                self.vectorizer = None
                self.tfidf = None
        else:
            self.vectorizer = None
            self.tfidf = None

    def answer(self, query, top_k=1):
        if not self.kb_texts:
            return None, 0.0
        if SKLEARN_AVAILABLE and getattr(self, 'tfidf', None) is not None:
            q_vec = self.vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self.tfidf).flatten()
            idx = int(sims.argmax())
            return self.kb_texts[idx], float(sims[idx])
        q_tokens = set(query.lower().split())
        best_idx = -1
        best_score = 0.0
        for i, t in enumerate(self.kb_texts):
            t_tokens = set(t.lower().split())
            if not t_tokens:
                continue
            score = len(q_tokens & t_tokens) / max(1, len(t_tokens))
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx == -1:
            return None, 0.0
        return self.kb_texts[best_idx], float(best_score)


RULES = {
    "hello": "Hello! How can I help you today?",
    "hi": "Hi there — what can I do for you?",
    "hey": "Hey! What would you like to talk about?",
    "good morning": "Good morning! How can I assist you today?",
    "good afternoon": "Good afternoon! Ready to learn something?",
    "good evening": "Good evening! How can I help you tonight?",
    "how are you": "I'm always running at full speed — ready to assist!",
    "what can you do": "I can answer simple questions from a knowledge base, or respond to basic greetings.",
    "who are you": "I'm Simple Chatbot — your assistant.",
    "what's your name": "I'm Simple Chatbot.",
    "name": "I'm Simple Chatbot.",
    "help": "You can ask me about coding, general knowledge, or your KB files.",
    "thank": "You're welcome! Glad I could help.",
    "thanks": "You're welcome! Glad I could help.",
    "bye": "Goodbye! Take care.",
    "goodbye": "Goodbye! See you later.",

    "python": "Python is a versatile, beginner-friendly programming language.",
    "python install": "Install Python from python.org or via your system's package manager.",
    "pip": "pip installs Python packages using 'pip install package-name'.",
    "python list": "A Python list is defined like: [1, 2, 3].",
    "python dict": "A dict stores key-value pairs, like {'a': 1}.",
    "python function": "Define functions with 'def name(args):'.",
    "python loop": "For-loops iterate items: for x in list: print(x).",
    "python class": "Classes define objects using 'class MyClass:'.",

    "c++": "C++ is a powerful, high-performance programming language.",
    "c++ pointer": "A pointer stores a memory address in C++.",
    "c++ compile": "Compile using: g++ file.cpp -o app.",

    "programming": "Programming is writing instructions for a computer.",
    "variable": "A variable stores data in a program.",
    "loop": "A loop repeats a block of code.",
    "bug": "A bug is an error in a program.",
    "debug": "Debugging means finding and fixing bugs.",

    "planet": "There are 8 planets in our solar system.",
    "earth": "Earth is the third planet from the Sun.",
    "sun": "The Sun is a star at the center of our solar system.",
    "moon": "The Moon is Earth's only natural satellite.",
    "universe": "The universe is everything — space, time, matter, and energy.",
    "galaxy": "We live in the Milky Way galaxy.",
    "star": "A star is a massive, luminous sphere of plasma.",
    "black hole": "A black hole is an object with gravity so strong that not even light escapes.",

    "india": "India is a country in South Asia known for its diversity.",
    "usa": "The USA is a country in North America.",
    "china": "China is the most populous country in the world.",
    "japan": "Japan is known for technology and tradition.",
    "capital of india": "The capital of India is New Delhi.",
    "capital of usa": "The capital of the USA is Washington, D.C.",
    "capital of japan": "The capital of Japan is Tokyo.",
    "capital of china": "The capital of China is Beijing.",

    "water": "Water is H2O — essential for life.",
    "oxygen": "Oxygen is the element we breathe, symbol O2.",
    "gold": "Gold is a precious metal with atomic number 79.",
    "gravity": "Gravity is the force that pulls objects together.",

    "time": "Time is measured in seconds, minutes, and hours.",
    "day": "A day has 24 hours.",
    "year": "A year has 365 days (366 in a leap year).",
    "leap year": "A leap year occurs every 4 years and has 366 days.",

    "weather": "Weather describes conditions like rain, heat, or wind.",
    "rain": "Rain is liquid water droplets that fall from clouds.",
    "cloud": "Clouds are made of tiny water droplets or ice crystals.",
    "wind": "Wind is moving air.",
    "temperature": "Temperature measures how hot or cold something is.",

    "food": "Food provides nutrients for energy and body functions.",
    "fruit": "Fruits are sweet, edible plant products like apples and bananas.",
    "vegetable": "Vegetables include carrots, spinach, and broccoli.",
    "protein": "Protein helps build muscles and repair tissues.",

    "internet": "The internet is a global network of interconnected computers.",
    "wifi": "WiFi allows wireless internet access.",
    "computer": "A computer processes information and performs tasks.",
    "cpu": "The CPU is the brain of the computer.",
    "gpu": "The GPU handles graphics and parallel processing.",
    "ram": "RAM is memory used by applications while running.",
    "ssd": "An SSD is a fast storage device.",

    "ai": "AI stands for Artificial Intelligence — machines that mimic human intelligence.",
    "machine learning": "Machine learning lets computers learn patterns from data.",
    "neural network": "A neural network is a model inspired by the human brain.",
    "robot": "A robot is a machine capable of carrying out complex tasks.",

    "health": "Health is the state of physical, mental, and social well-being.",
    "exercise": "Exercise improves physical fitness and health.",
    "sleep": "Adults need 7–9 hours of sleep per night.",
    "vitamin": "Vitamins are essential nutrients for the body.",

    "history": "History is the study of past events.",
    "science": "Science explains the natural world through observation and experiments.",
    "math": "Math is the study of numbers, shapes, and patterns.",
    "english": "English is one of the most widely spoken languages.",
    "geography": "Geography studies Earth's landscapes and environments.",
    "biology": "Biology is the study of life.",
    "chemistry": "Chemistry studies matter and its interactions.",
    "physics": "Physics studies motion, energy, and forces.",

    "car": "A car is a motor vehicle used for transportation.",
    "bike": "A bike is a two-wheeled vehicle.",
    "train": "Trains are fast transportation systems running on rails.",
    "airplane": "Airplanes fly using lift generated by wings.",

    "music": "Music is the art of arranging sound.",
    "song": "A song is a short musical composition.",
    "movie": "A movie is a story shown as moving images.",
    "game": "Games are structured forms of play.",

    "sports": "Sports are competitive physical activities like football and cricket.",
    "football": "Football is played with a round ball on a rectangular field.",
    "cricket": "Cricket is a bat-and-ball game popular in many countries.",
    "olympics": "The Olympics are international sporting events held every four years.",

    "animals": "Animals are living organisms that feed on organic matter.",
    "dog": "Dogs are domesticated mammals often kept as pets.",
    "cat": "Cats are small carnivorous mammals, popular as companions.",
    "lion": "Lions are large wild cats known as 'king of the jungle'.",
    "elephant": "Elephants are large mammals with trunks.",
    "whale": "Whales are large marine mammals.",

    "plants": "Plants produce oxygen and food via photosynthesis.",
    "tree": "Trees are tall plants with woody trunks.",
    "flower": "Flowers are the reproductive parts of many plants.",
    "ocean": "Oceans cover about 71% of Earth's surface.",
    "mountain": "Mountains are large landforms that rise prominently above surroundings.",

    "capital of france": "The capital of France is Paris.",
    "currency of india": "The currency of India is the Indian Rupee (INR).",
    "currency of usa": "The currency of the USA is the US Dollar (USD).",
    "population of india": "India's population is over 1.4 billion (approx.).",

    "trivia": "Fun fact: Honey never spoils under the right conditions.",
    "fun fact": "Octopuses have three hearts.",

    "kids": "Kids questions are welcome — ask me simple facts or fun facts!",
    "easy math": "2 + 2 = 4.",
    "counting": "Counting starts from 1, 2, 3, and so on.",

    "anime": "Anime is a style of animation originating from Japan.",
    "manga": "Manga are Japanese comics and graphic novels.",
    "naruto": "Naruto is a popular anime about ninjas and perseverance.",
    "one piece": "One Piece follows Monkey D. Luffy on his pirate adventures.",
    "attack on titan": "Attack on Titan is a dark fantasy anime featuring giant humanoid creatures.",
    "demon slayer": "Demon Slayer follows Tanjiro as he fights demons to save his sister.",

    "history timeline": "Timelines show events in chronological order.",
    "world war 1": "World War I occurred from 1914 to 1918.",
    "world war 2": "World War II occurred from 1939 to 1945.",

    "sports world cup": "The FIFA World Cup is held every four years and features national football teams.",
    "olympic sports": "Olympic sports include athletics, swimming, gymnastics, and more.",

    "animals facts": "Some animals migrate long distances, like whales and birds.",
    "plants facts": "Plants convert sunlight into energy through photosynthesis.",

    "exams": "Study regularly, make notes, and practice past papers to prepare for exams.",
    "common exam question": "Practice problem-solving, time management, and clear writing for exams.",

    "currency": "Currency is the system of money used in a country.",
    "population": "Population counts the number of people living in an area.",

    "geography facts": "Geography covers physical features, countries, and climates.",
    "science facts": "Science uses experiments to learn about the natural world.",

    "sports stars": "Famous sports stars include athletes like Lionel Messi and Serena Williams.",
    "cricket world cup": "The ICC Cricket World Cup is a major international tournament held periodically.",

    "animals habitat": "Animals live in habitats like forests, deserts, oceans, and grasslands.",
    "plant parts": "Plant parts include roots, stems, leaves, flowers, and seeds.",

    "oceans names": "Major oceans: Pacific, Atlantic, Indian, Southern, and Arctic.",
    "highest mountain": "Mount Everest is the highest mountain above sea level.",

    "history facts": "History includes events like revolutions, discoveries, and cultural shifts.",
    "math facts": "Prime numbers are greater than 1 and divisible only by 1 and themselves.",

    "space facts": "Space is vast and contains many galaxies, stars, and planets.",
    "technology facts": "Technology evolves rapidly and influences daily life.",
    "JoJo": "Jojo`s Bizzare adventure is an Japanese Anime which is really bizzare",
} 


def rule_based_response(user_text):
    s = user_text.lower()
    for k, v in RULES.items():
        if k in s:
            return str(v)
    return None


def _safe_input(prompt):
    try:
        return input(prompt)
    except (KeyboardInterrupt, EOFError, OSError):
        return None


def interactive_chat():
    if not sys.stdin or not sys.stdin.isatty():
        print("Interactive mode requires a TTY. Run with --api or --test in non-interactive environments.")
        return
    print("Simple Chatbot — type 'exit' or 'quit' to leave.")
    print("Modes:\n  1) rule — simple canned replies\n  2) retrieval — search a 'kb' folder for answers")
    print("If a 'kb' folder with .txt files exists, retrieval mode will be available.")

    kb_texts, kb_names = load_kb_texts("kb")
    responder = None
    if kb_texts:
        responder = RetrievalResponder(kb_texts, kb_names)
        print(f"Loaded {len(kb_texts)} documents from 'kb' for retrieval mode.")
    else:
        print("No KB files found — retrieval mode unavailable.")

    mode = None
    while mode not in {"rule", "retrieval", "auto"}:
        val = _safe_input("Choose mode (rule/retrieval/auto) [auto]: ")
        if val is None:
            print("No input available. Exiting interactive mode.")
            return
        mode = val.strip().lower() or "auto"
        if mode == "retrieval" and not responder:
            print("Retrieval mode not available because no KB files were found. Choose another mode.")
            mode = None

    print("Starting chat. Enter your message:")
    while True:
        val = _safe_input("You: ")
        if val is None:
            print("No input available. Exiting interactive mode.")
            return
        user = val.strip()
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            print("Bot: Goodbye!")
            break
        resp = rule_based_response(user)
        if resp and mode in {"rule", "auto"}:
            print("Bot:", resp)
            continue
        if responder and mode in {"retrieval", "auto"}:
            answer, score = responder.answer(user)
            if answer and score > 0.15:
                snippet = answer.strip().replace('\n', ' ')[:800]
                print(f"Bot (kb, score={score:.2f}): {snippet}")
                continue
        print("Bot: Sorry, I don't know the answer to that. Try rephrasing or add documents to the 'kb' folder.")


class SimpleChatbot:
    def __init__(self, kb_folder="kb"):
        texts, names = load_kb_texts(kb_folder)
        self.responder = RetrievalResponder(texts, names) if texts else None

    def get_response(self, user_text):
        r = rule_based_response(user_text)
        if r:
            return r
        if self.responder:
            ans, score = self.responder.answer(user_text)
            if ans and score > 0.15:
                return ans
        return "I'm not sure. Try adding more knowledge files to 'kb' or ask something else."


class SimpleChatbotTests(unittest.TestCase):
    def test_rule_based(self):
        self.assertEqual(rule_based_response("Hello there"), "Hello! How can I help you today?")
        self.assertEqual(rule_based_response("hi! what's up"), "Hi there — what can I do for you?")
        self.assertEqual(rule_based_response("thanks a lot"), "You're welcome! Glad I could help.")
        self.assertEqual(rule_based_response("what can you do?"), "I can answer simple questions from a knowledge base, or respond to basic greetings.")

    def test_retrieval_fallback(self):
        r = RetrievalResponder([])
        self.assertEqual(r.answer("anything"), (None, 0.0))

    def test_simplechatbot_with_kb(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "a.txt").write_text("This document explains Python and programming.")
            (p / "b.txt").write_text("This one is about gardening and plants.")
            bot = SimpleChatbot(kb_folder=td)
            self.assertIn("Hello", bot.get_response("hello"))
            resp = bot.get_response("Tell me about programming in python")
            self.assertTrue("Python" in resp or "programming" in resp)


if __name__ == "__main__":
    if "--api" in sys.argv:
        bot = SimpleChatbot()
        print(bot.get_response("What is your name?"))
        sys.exit(0)
    if "--test" in sys.argv:
        unittest.main(argv=[sys.argv[0]])
    else:
        interactive_chat()
