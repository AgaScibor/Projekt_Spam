import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st 
from text_processing import preprocessing_text
from predict import predict_spam

# wyniki wykorzystanego modelu SVM
svm_metrics_df = pd.DataFrame({
    "Accuracy": [0.9897],
    "Precision": [0.9766],
    "Recall": [0.9921],
    "F1-score": [0.9843]
}, index=["SVM"])

# modele i ich metryki
all_models_df = pd.DataFrame({
    "Accuracy": [0.9897, 0.9853, 0.9810],
    "Precision": [0.9766, 0.9892, 0.9917],
    "Recall": [0.9921, 0.9657, 0.9499],
    "F1-score": [0.9843, 0.9773, 0.9704]
}, index=["SVM", "Logistic Regression", "Naive Bayes"])

# macierz pomyłek dla modelu SVM
confusion_df = pd.DataFrame({
    "Przewidziany HAM": [772, 9],
    "Przewidziany SPAM": [3, 376]
}, index=["HAM", "SPAM"])


# Strona główna

def main_page():
    st.title("Spam Detection App")
    st.write(
        "Aplikacja służy do automatycznej klasyfikacji wiadomości tekstowych "
        "na **SPAM** lub **HAM** z wykorzystaniem algorytmów uczenia maszynowego."
    )

    with st.expander("Informacje o zbiorze danych"):
        st.markdown("""
        Zbiór danych zawiera wiadomości oznaczone jako **SPAM** lub **HAM**.

        **Struktura danych:**
        - `label` – typ wiadomości (SPAM / HAM)
        - `text` – treść wiadomości

        **Statystyki:**
        - Liczba próbek: 5796  
        - HAM: 3900 (~67%)  
        - SPAM: 1896 (~32%)
        """)

    with st.expander("Użyte techniki"):
        st.markdown("""
        - Normalizacja tekstu
        - Tokenizacja
        - Usuwanie stop words
        - Lematizacja (SpaCy)
        - Klasyfikacja: SVM, Logistic Regression, Naive Bayes
        """)


# Klasyfikator

def spam_classifier():
    st.title("Klasyfikator wiadomości")

    text = st.text_area(
        "Wpisz wiadomość do klasyfikacji:",
        value=st.session_state.get("loaded_text", ""),
        height=200
    )

    if st.button(" Klasyfikuj"):
        if text.strip() == "":
            st.warning("Wpisz wiadomość.")
            return

        processed_text = preprocessing_text(text)
        prediction, probability = predict_spam(processed_text)

        if prediction == "SPAM":
            st.error("Wiadomość została zaklasyfikowana jako **SPAM**")
            prob_spam = probability
        else:
            st.success("Wiadomość została zaklasyfikowana jako **HAM**")
            prob_spam = 1 - probability

        st.write(f"Prawdopodobieństwo: **{prob_spam:.2%}**")

        # wykres
        fig, ax = plt.subplots()
        values = [1 - prob_spam, prob_spam]
        ax.bar(["HAM", "SPAM"], values)
        ax.set_ylabel("Prawdopodobieństwo")
        ax.set_title("Prawdopodobieństwo klas")
        for i, v in enumerate(values):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center")
        st.pyplot(fig)


# Analiza modelu

def model_analysis():
    st.title("Analiza modelu")

    st.subheader("Metryki modelu SVM")
    st.dataframe(svm_metrics_df)

    st.subheader("Porównanie modeli")
    st.dataframe(all_models_df)

    st.subheader("Porównanie F1-score")
    fig, ax = plt.subplots()
    all_models_df["F1-score"].plot(kind="bar", ax=ax)
    ax.set_ylabel("F1-score")
    st.pyplot(fig)


# Przykłady demonstracyjne

def demonstration():
    st.title("Przykłady demonstracyjne")

    example_messages = {
        "SPAM": """*** Attention: US Citizens *** Earn extra money working at home in your spare time.

The government needs your help!

Become a mortgage refund tracer in a few easy steps.

Millions of dollars are left to be distrubuted each year and each state government
pays tracers to help track down the poeple who are to receive the money owed to them.
It is the law that they must disperse the funds.

We provide you with the tools and information necessary to make yourself thousands
each month working from the privacy of your own home.

Part time or full time - You spend as much or as little time as you want while helping
your neighbors and the US government!

Visit: http://www3.sympatico.ca/mark.daisy/mortgage/

Don't miss this awesome opportunity!
No risk to participate in this great program!!!

----------------------------------------------
Please note: This is a one time mailing from our marketing department.

126432211111
""",
        "HAM": """URL: http://boingboing.net/#85516563

Date: Not supplied



Interview with cartoonist Ted Rall, who has traveled to South Asia recently, 

and has tips on how to deal with bribe-hungry border guards and the like: 



    Rall: Now I realize that's just the way it is, and I know how to do it and 

    get away without paying a bribe, or paying something very modest. You have 

    to show them that you know the routine and that you know you don't have to 

    give them anything, and just have a low-key demeanor. But if you get angry, 

    that's just going to make things worse for you. You get out of your vehicle 

    and you walk up to them -- you don't try to avoid these guys; you don't try 

    to avoid their eyes -- you go up with a big smile, give them a big 

    handshake and sort of rub their shoulders and say, "Hey, great to see you. 

    You're my new best friend for the next five minutes." 



    GROTH: Basically act like a used car salesman. 



    RALL: It's exactly like that! You always carry cigarettes. You offer them a 

    cigarette, and you say, "Hey, what's goin' on? How's it goin'? Great. 

    Here's my documents. How's the road?" Just small talk, because these guys 

    are bored. They're in the middle of nowhere, and you're sometimes the only 

    vehicle they've seen for many hours. They're often very drunk, so you just 

    have to be cool.  



Link[1] Discuss[2]



[1] http://www.tcj.com/247/i_rall.html

[2] http://www.quicktopic.com/16/H/de7BfPnMekg




"""
    }

    example_choice = st.selectbox("Wybierz przykład:", list(example_messages.keys()))

    if st.button("Załaduj przykład"):
        st.write(f"Przykład wiadomości ({example_choice}):")
        st.write(example_messages[example_choice])


# Nawigacja

if "page" not in st.session_state:
    st.session_state["page"] = "Strona główna"

st.sidebar.title("Nawigacja")
selected_page = st.sidebar.radio(
    "Wybierz stronę:",
    ["Strona główna", "Klasyfikator", "Analiza modelu", "Przykłady demonstracyjne"]
)

st.session_state["page"] = selected_page

if selected_page == "Strona główna":
    main_page()
elif selected_page == "Klasyfikator":
    spam_classifier()
elif selected_page == "Analiza modelu":
    model_analysis()
elif selected_page == "Przykłady demonstracyjne":
    demonstration()
