from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from collections import defaultdict, Counter
import streamlit as st
import re

# --------------------Dictionary --------------------
dictionary = {



    # -------------------------
    # Basic Negative Words
    # -------------------------
    "no": "نه",
    "not": "نہیں",
    "never": "هرگز",
    "none": "هیچ",
    "nothing": "هیچ چیز",
    "nowhere": "هیچ جا",
    "neither": "نه این نه آن",
    "nor": "نه",
    "nobody": "هیچ‌کس",
    "without": "بدون",
    "fail": "شکست خوردن",
    "failure": "شکست",
    "wrong": "اشتباه",
    "bad": "بد",
    "worse": "بدتر",
    "worst": "بدترین",
    "stop": "توقف",
    "impossible": "غیر ممکن",
    "lack": "کمبود",
    "lost": "گم شده",

    # -------------------------
    # Negative Verbs
    # -------------------------
    "can't": "نمی توانم",
    "cannot": "نمی توانم",
    "won't": "نخواهم",
    "don't": "نکن",
    "didn't": "نکرد",
    "isn't": "نیست",
    "aren't": "نیستند",
    "wasn't": "نبود",
    "weren't": "نبودند",
    "shouldn't": "نباید",
    "couldn't": "نمی‌توانست",
    "doesn't": "نمی‌کند",
    "haven't": "ندارم",
    "hasn't": "ندارد",
    "hadn't": "نداشت",
    "avoid": "اجتناب کردن",
    "reject": "رد کردن",

    # -------------------------
    # Negative Emotions
    # -------------------------
    "sad": "غمگین",
    "unhappy": "ناراحت",
    "angry": "عصبانی",
    "upset": "ناراحت",
    "hate": "نفرت",
    "fear": "ترس",
    "scared": "ترسیده",
    "afraid": "ترسیده",
    "stress": "استرس",
    "depressed": "افسرده",
    "anxious": "نگران",
    "pain": "درد",
    "hurt": "آزار دیده",

    # -------------------------
    # Negative Adjectives
    # -------------------------
    "poor": "ضعیف",
    "weak": "کمزور",
    "ugly": "زشت",
    "lazy": "تنبل",
    "rude": "بی‌ادب",
    "fake": "جعلی",
    "broken": "خراب",
    "dangerous": "خطرناک",
    "negative": "منفی",
    "hard": "سخت",
    "difficult": "دشوار",
    "hopeless": "بی‌امید",

    # -------------------------
    # Negative Phrases
    # -------------------------
    "not at all": "اصلاً",
    "not really": "نه واقعاً",
    "no way": "به هیچ وجه",
    "nothing special": "چیز خاصی نیست",
    "not good": "خوب نیست",
    "not working": "کار نمی‌کند",
    "out of order": "خراب است",
    "don't like": "دوست ندارم",
    "don't want": "نمی‌خواهم",
    "no choice": "هیچ انتخابی نیست",
  # Pronouns
    "i": "من", "you": "تو", "he": "او", "she": "او", "it": "آن",
    "we": "ما", "they": "آنها",
    "me": "من", "him": "او", "her": "او", "us": "ما", "them": "آنها",
    "my": "من", "your": "تو", "his": "او", "our": "ما", "their": "آنها","very":"خیلی",

    # Verbs
    "am": "هستم", "is": "است", "are": "هستند", 
    "was": "بود", "were": "بودند",
    "have": "داشتن", "has": "دارد", "had": "داشت",
    "do": "کردن", "does": "می‌کند", "did": "کرد",
    "be": "بودن",
    "say": "گفتن",
    "get": "گرفتن",
    "make": "ساختن",
    "go": "رفتن", "went": "رفتم", "goes": "می‌رود", "going": "می‌رود",
    "see": "دیدن", "saw": "دیدم", "seen": "دیده",
    "take": "گرفتن",
    "come": "آمدن", "came": "آمدم",
    "know": "دانستن", "knew": "دانستم",
    "think": "فکر کردن", "thought": "فکر کردم",
    "want": "خواستن", "wanted": "خواستم",
    "like": "دوست داشتن", "liked": "دوست داشتم",
    "love": "دوست داشتن", "loved": "دوست داشتم",
    "work": "کار کردن", "worked": "کار کردم",
    "play": "بازی کردن", "played": "بازی کردم",
    "read": "خواندن", "reading": "می‌خواند",
    "write": "نوشتن", "writing": "می‌نویسد",
    "speak": "صحبت کردن", "speaking": "صحبت می‌کند",
    "understand": "فهمیدن", "understanding": "می‌فهمد",
    "learn": "یاد گرفتن", "learning": "یاد می‌گیرد",
    "teach": "آموزش دادن", "teaching": "آموزش می‌دهد",
    "look": "نگاه کردن",
    "give": "دادن",
    "use": "استفاده کردن",
    "find": "پیدا کردن",
    "tell": "گفتن",
    "ask": "پرسیدن",
    "seem": "به نظر رسیدن",
    "feel": "احساس کردن",
    "try": "تلاش کردن",
    "leave": "ترک کردن",
    "call": "صدا زدن",
    "drink": "نوشیدن",
    "run": "دویدن",
    "live": "زندگی کردن",
    "move": "حرکت کردن",
    "study": "مطالعه کردن",
    "start": "شروع کردن",
    "stop": "متوقف کردن",
    # FIX: Added 'beat', 'kick', 'hate', 'feeling'
    "hate": "نفرت",
    "beat": "زدن",
    "kick": "لگد زدن",
    "feeling": "احساس",


    # Modal verbs
    "can": "می‌توانم", "could": "می‌توانستم",
    "will": "خواهم", "would": "می‌کردم",
    "shall": "باید", "should": "باید",
    "may": "ممکن است", "might": "ممکن بود",
    "must": "باید",

    # Nouns
    "time": "زمان", "person": "شخص", "people": "مردم",
    "man": "مرد", "woman": "زن",
    "child": "کودک", "children": "کودکان",
    "family": "خانواده",
    "friend": "دوست",
    "house": "خانه", "home": "خانه",
    "room": "اتاق", "door": "در", "window": "پنجره",
    "car": "ماشین", "bus": "اتوبوس", "train": "قطار", "bike": "دوچرخه",
    "water": "آب", "food": "غذا", "bread": "نان", "rice": "برنج", "fruit": "میوه",
    "book": "کتاب", "pen": "قلم", "paper": "کاغذ", "computer": "کامپیوتر",
    "phone": "تلفن", "tv": "تلویزیون", "music": "موسیقی",
    "school": "مدرسه", "teacher": "معلم", "student": "دانشجو",
    "work": "کار", "job": "شغل", "money": "پول", "price": "قیمت",
    "city": "شهر", "country": "کشور", "world": "جهان",
    "street": "خیابان",
    "day": "روز", "night": "شب", "week": "هفته", "month": "ماه", "year": "سال",
    "sun": "خورشید", "moon": "ماه", "star": "ستاره", "sky": "آسمان",
    "tree": "درخت", "flower": "گل", "animal": "حیوان",
    "dog": "سگ", "cat": "گربه", "bird": "پرنده", "fish": "ماهی",
    "name": "نام", "word": "کلمه", "number": "عدد",
    "problem": "مشکل", "fact": "واقعیت",
    "way": "راه", "thing": "چیز",
    "life": "زندگی",
    "hand": "دست", "eye": "چشم", "head": "سر",
    "place": "مکان", "point": "نقطه",
    "toy": "اسباب بازی", # FIX: Added 'toy'
    
    # Adjectives
    "good": "خوب", "bad": "بد",
    "big": "بزرگ", "small": "کوچک",
    "beautiful": "زیبا", "ugly": "زشت",
    "happy": "خوشحال", "sad": "غمگین",
    "angry": "عصبانی", "tired": "خسته", "hungry": "گرسنه", "thirsty": "تشنه",
    "hot": "داغ", "cold": "سرد", "warm": "گرم", "cool": "خنک",
    "new": "جدید", "old": "قدیمی",
    "young": "جوان", "fast": "سریع", "slow": "آهسته",
    "easy": "آسان", "difficult": "سخت",
    "important": "مهم",
    "first": "اولین", "last": "آخرین",
    "long": "بلند",
    "great": "عالی",
    "little": "کوچک",
    "own": "مال خود",
    "other": "دیگر",
    "high": "بالا",
    "different": "متفاوت",
    "public": "عمومی",
    "red": "قرمز", "blue": "آبی", "green": "سبز",
    "ready": "آماده",
    "exhausted": "خسته", # FIX: Added 'exhausted'
    "dumb": "احمق", # FIX: Added 'dumb'


    # Prepositions
    "in": "در", "on": "روی", "at": "در",
    "to": "به", "from": "از",
    "with": "با", "without": "بدون",
    "for": "برای", "about": "درباره",
    "by": "توسط", "of": "از",
    "up": "بالا", "down": "پایین",
    "out": "بیرون",
    "over": "روی",
    "under": "زیر",

    # Question words
    "what": "چه", "who": "چه کسی", "where": "کجا",
    "when": "کی", "why": "چرا",
    "how": "چطور", "which": "کدام",

    # Common phrases/Other words
    "hello": "سلام", "hi": "سلام",
    "goodbye": "خداحافظ", "bye": "خداحافظ",
    "please": "لطفا",
    "thank you": "متشکرم", "thanks": "تشکر",
    "sorry": "متاسفم",
    "excuse me": "ببخشید",
    "yes": "بله", "no": "نه",
    "ok": "باشه", "okay": "باشه",
    "this": "این", "that": "آن",
    "these": "اینها", "those": "آنها",
    "here": "اینجا", "there": "آنجا",
    "now": "اکنون", "then": "سپس",
    "today": "امروز", "tomorrow": "فردا", "yesterday": "دیروز",
    "later": "بعدا",
    "a": "یک", "an": "یک", "the": "حرف تعریف"
}

# -------------------- Part-of-Speech Map --------------------
POS_MAP = {
    
"i": "PRON", "you": "PRON", "he": "PRON", "she": "PRON", "it": "PRON",
    "we": "PRON", "they": "PRON", "me": "PRON", "him": "PRON", "her": "PRON",
    "us": "PRON", "them": "PRON", "my": "PRON", "your": "PRON", "his": "PRON",
    "our": "PRON", "their": "PRON",
    "am": "V_ACTION", "is": "V_ACTION", "are": "V_ACTION", "was": "V_ACTION", "were": "V_ACTION",
    "be": "V_ACTION", "say": "V_ACTION", "get": "V_ACTION", "make": "V_ACTION",
    "go": "V_ACTION", "see": "V_ACTION", "take": "V_ACTION", "come": "V_ACTION",
    "know": "V_ACTION", "think": "V_ACTION", "want": "V_ACTION", "love": "V_ACTION",
    "work": "V_ACTION", "read": "V_ACTION", "write": "V_ACTION", "run": "V_ACTION",
    "play": "V_ACTION", "feel": "V_ACTION", "give": "V_ACTION", "ask": "V_ACTION",
    "start": "V_ACTION", "stop": "V_ACTION", "live": "V_ACTION", "use": "V_ACTION",
    # FIX: Added 'hate', 'beat', 'kick', 'feeling' to V_ACTION
    "hate": "V_ACTION",
    "beat": "V_ACTION",
    "kick": "V_ACTION",
    "feeling": "V_ACTION",
}

def get_pos(word):
    return POS_MAP.get(word.lower(),'OTHER')



# -------------------- Training Phrases (Expanded and Improved) --------------------
training_phrases = [
    # Basic Positives/Negatives/Neutrals
    "how are you", "i am fine", "thank you", "good morning", "see you later",
    "i love food", "you are beautiful", "today is good day", "i am happy",
    "you are my friend", "i am tired", "are you hungry", "what is your name",
    "good night", "i am very sad", "i am very happy", "i read book",
    "i write with pen", "i drink water", "i run fast", "the car is big",
    "my house is small", "i am tired and hungry", "red car", "i want to go",
    "i have a car", "i live in city", "i feel good", "she is happy",
    "we live here", "they are ready",
    
    # Advanced Sentences (New Additions for better context)
    # Negatives
    "this is bad and disappointing", 
    "i am angry with the service",    
    "i hate everything about this",    
    "the food was not good",         
    "i don't like this product at all", 
    "i feel terrible right now",
    "that service was the worst i have ever seen", 
    "i am feeling so sad and angry", 
    "this looks so ugly",
    "i am totally exhausted and dumb", # Includes new harsh words
    "i want to beat and kick them", # Includes new harsh words
    
    # Positives
    "this is an awesome product",     
    "everything is absolutely great", 
    "i feel wonderful today",
    "i love spending time here",
    "you are the best friend ever",
    "i like playing with my new toy", # Includes new word 'toy'
    
    # Neutrals
    "i wish things were different but it is okay",   
    "i am not sure about this plan",  
    "the weather is very cold",       
    "the service was okay, nothing special", 
    "i need to buy a new car",
    "i am watching tv tonight",
]

# -------------------- AI-based Sentiment (NLP ML) --------------------
# Simple pseudo-labeled training for ML
positive_words = {"good", "happy", "love", "fine", "great", "awesome", "wonderful", "best"}
# FIX: Added 'ugly', 'worst', 'don't', 'dumb', 'exhausted' to capture harshness/negative state.
negative_words = {"sad", "angry", "bad", "hate", "tired", "hungry", "disappointing", "terrible", "ugly", "worst", "don't", "dumb", "exhausted"}

train_labels = []
for text in training_phrases:
    words = set(text.lower().split())
    pos_count = len(words & positive_words)
    neg_count = len(words & negative_words)
    if pos_count > neg_count:
        train_labels.append("Positive")
    elif neg_count > pos_count:
        train_labels.append("Negative")
    else:
        train_labels.append("Neutral")

# Train TF-IDF + Logistic Regression
vectorizer_sent = TfidfVectorizer()
X_train = vectorizer_sent.fit_transform(training_phrases)

# Logistic Regression with aggressive C (100.0) and class_weight='balanced' for better fitting sparse negative data.
ml_model_sent = LogisticRegression(max_iter=5000, class_weight='balanced', C=100.0)
ml_model_sent.fit(X_train, train_labels)

def ml_sentiment_predict(sentence):
    X_test = vectorizer_sent.transform([sentence])
    pred = ml_model_sent.predict(X_test)[0]
    proba = ml_model_sent.predict_proba(X_test).max()
    return f"{pred} ({int(proba*100)}% confidence)"


# -------------------- Translation & Reordering --------------------
def simple_translate_and_reorder(sentence, dictionary):
    words = re.findall(r"\b\w+\b", sentence.lower())
    original_words = re.findall(r"\b\w+\b", sentence)
    translated = [dictionary.get(w.lower(), w) for w in words]
    en_tags = [get_pos(w) for w in words]
    if en_tags and en_tags[0]=='PRON':
        action_verb_indices = [i for i,tag in enumerate(en_tags) if tag=='V_ACTION']
        if len(action_verb_indices)==1:
            verb_index = action_verb_indices[0]
            fa_before_verb = translated[:verb_index]
            fa_after_verb = translated[verb_index+1:]
            fa_verb = translated[verb_index]
            reordered_fa = fa_before_verb + fa_after_verb + [fa_verb]
            return " ".join(reordered_fa)
    return " ".join(translated)

# -------------------- AI-based Next Word Prediction (Bigram) --------------------
bigram_freq = defaultdict(Counter)
for phrase in training_phrases:
    words = phrase.lower().split()
    for i in range(len(words) - 1):
        bigram_freq[words[i]][words[i + 1]] += 1

def ml_next_word_predict(sentence, top_n=3):
    words = sentence.lower().split()
    if not words:
        return ""
    last_word = words[-1]
    next_words_counter = bigram_freq.get(last_word)
    if not next_words_counter:
        return ""
    # Return top N next words sorted by frequency
    most_common = next_words_counter.most_common(top_n)
    return ", ".join([word for word, _ in most_common])

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="English-Persian Translator", page_icon="🤖", layout="centered")
st.title(" 🗣️ English → Persian Translator")
st.markdown("---")

if "translation_history" not in st.session_state:
    st.session_state["translation_history"] = []

def clear_history():
    st.session_state["translation_history"] = []

col_buttons = st.columns([1, 4])
with col_buttons[0]:
    st.button("🗑️ Clear History", on_click=clear_history, use_container_width=True)

with st.container():
    if not st.session_state["translation_history"]:
        st.info("Start a conversation by typing a sentence below!")
    for item in st.session_state.translation_history[::-1]:
        with st.chat_message("user"):
            st.write(item['user_text'])
        with st.chat_message("translator", avatar="🤖"):
            st.success(item['translation'])

user_input = st.chat_input("Type your English sentence here...")

if user_input:
    persian_reordered = simple_translate_and_reorder(user_input, dictionary)
    sentiment_ml = ml_sentiment_predict(user_input)
    suggested_ml = ml_next_word_predict(user_input)

    result = {
        "user_text": user_input,
        "translation": persian_reordered,
        "sentiment": sentiment_ml,
        "suggestions": suggested_ml
    }
    st.session_state.translation_history.insert(0, result)
    st.rerun()

if st.session_state.translation_history:
    st.subheader("📊 Latest Translation Analysis")
    latest_result = st.session_state.translation_history[0]
    st.markdown(f"**Input analyzed:** *{latest_result['user_text']}*")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Persian Structure**")
        st.caption("SVO → SOV correction applied.")
    with col2:
        st.warning("🧠 **ML-based Sentiment**")
        st.metric(label="Mood", value=latest_result['sentiment'])
    if latest_result['suggestions']:
        st.markdown("---")
        st.code(f"💡 Next Word Prediction: {latest_result['suggestions']}", language='text')
