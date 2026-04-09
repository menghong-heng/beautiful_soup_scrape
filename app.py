import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and vectorizer
model = joblib.load("logistic_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# Label map
label_map = {
    0: "😞 Negative",
    1: "😐 Neutral",
    2: "😊 Positive"
}
label_names = [label_map[i] for i in sorted(label_map)]

# Extra commonly used sentiment phrases to improve model context
positive_words = [
    "love", "like", "amazing", "awesome", "great", "fantastic", "excellent", "perfect", "recommend",
    "best", "wonderful", "happy", "enjoyed", "delightful", "worth it", "top quality", "superb",
    "very pleased", "brilliant", "incredible", "outstanding", "top-notch", "impressive", "flawless"
]

neutral_words = [
    "okay", "fine", "average", "not bad", "mediocre", "decent", "so-so", "neutral", "standard",
    "plain", "basic", "normal", "expected", "nothing special", "fair", "alright", "acceptable"
]

negative_words = [
    "hate", "terrible", "awful", "bad", "worst", "disappointed", "poor", "buggy", "not worth it",
    "broken", "cheap", "glitchy", "unreliable", "doesn't work", "waste", "slow", "frustrating",
    "hard to use", "ugly", "annoying", "low quality", "regret", "horrible", "no support"
]

# Add extra common review-style context phrases
common_phrases = [
    "In my opinion,", "I recently tried this and", "To be honest,",
    "From my experience,", "Honestly,", "I feel like",
    "I thought it would be better,", "The product seemed",
    "What I noticed was", "Overall,", "After using it for a while,",
    "At first glance,", "Initially,", "It turned out that",
    "After testing it," , "Compared to others,", "I believe",
    "Here’s what I found", "I noticed that", "My impression is"
]

def enrich_input(text):
    if len(text.split()) < 5:
        context = " ".join(np.random.choice(common_phrases, 2, replace=False))
        return f"{context} {text}"
    return text

# UI Setup
st.set_page_config(page_title="Sentiment Checker", page_icon="💬")
st.title("💬 Modern Sentiment Predictor")
st.markdown("Enter your review below to get a clean and confident sentiment analysis result.")

# Input
user_input = st.text_area("📝 Enter your review here:", height=150)

if st.button("🔍 Predict Sentiment"):
    if user_input.strip():
        text_for_model = enrich_input(user_input)
        X_input = vectorizer.transform([text_for_model])
        prediction = model.predict(X_input)[0]
        probability = model.predict_proba(X_input)[0]
        confidence = np.max(probability) * 100

        sentiment = label_map.get(prediction, "❓ Unknown")

        # Display results
        st.markdown("## 🎯 Prediction Result")
        st.success(f"**Sentiment:** {sentiment}")
        st.info(f"**Confidence:** {confidence:.2f}%")

        # Chart
        st.markdown("### 📊 Probability Distribution")
        probs_df = pd.DataFrame({
            'Sentiment': label_names,
            'Probability': probability
        })
        st.bar_chart(probs_df.set_index("Sentiment"))

        # Raw values
        st.markdown("### 🔍 Raw Values")
        for i, prob in enumerate(probability):
            st.write(f"{label_map[i]}: {prob:.2%}")

        # Feedback
        st.markdown("---")
        st.markdown("### Was this prediction helpful?")
        col1, col2 = st.columns(2)
        with col1:
            st.button("👍 Yes")
        with col2:
            st.button("👎 No")

# Footer
st.markdown("---")
st.caption("© 2025 Sentiment Insights | Powered by your custom-trained model")