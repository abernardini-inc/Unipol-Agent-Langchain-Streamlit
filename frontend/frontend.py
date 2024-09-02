import streamlit as st
import os
import requests

CHATBOT_URL = os.getenv(
    "CHATBOT_URL", "http://127.0.0.1:8000"
)

with st.sidebar:
    st.header("Info")
    st.markdown(
        """
        Chatbot creato dalla società Unipol per rispondere alle domande dei clienti riguardanti 
        i servizi messi a disposizione da Unipol. È in grado di sottoscrivere contratti, 
        fornire le informazioni personali di un utente e mostrare lo stato dei pagamenti a un cliente. 
        L'agente utilizza la generazione aumentata dal recupero (RAG) su dati sia strutturati 
        che non strutturati, generati sinteticamente.
        """
    )

    st.header("Esempio di Domande")
    questions = [
        "Quale è il mio numero di telefono collegato?",
        "Quali sono le fatture che devo ancora pagare?",
        "Voglio sottoscrivere un nuovo contratto, aiutami!",
        "Quali sono gli sconti autostradali offerti da Unipol?",
        "Come posso cambiare lingua dalla mia applicazione?",
        "Che cosa è lo sconto Moto?",
        "Quale è la email collegata al mio profilo?",
        "Sono il cliente con ID 1234 voglio sottoscrivere un contratto UNIPOLMOVE CITY?",
        "Come posso modificare i miei dati anagrafici oppure i miei dati di residenza?",
        "La sbarra di accesso impiega troppo tempo per aprirsi. Cosa posso fare?",
        "È possibile effettuare la sostituzione del vetro a domicilio?",
        "Cos’è e come funziona lo Sconto sul Sistema Libero RAV (RAV rimborso pedaggio)?",
        "Ho richiesto il Soccorso stradale, ma non ho ricevuto il servizio. Cosa posso fare?"
    ]
    for question in questions:
        st.markdown(f"- {question}")

    # Button to clear the cache and start a new chat
    if st.button("Nuova Chat"):
        endpoint = f"{CHATBOT_URL}/new_chat"
        response = requests.get(endpoint)

        if response.status_code == 200:
            st.session_state.clear()
            st.experimental_rerun()

# Main content
st.title("Unipol Chatbot")
st.info(
    """Chiedimi qualsiasi cosa riguardante i servizi offerti da Unipol! Inoltre, posso sottoscrivere contratti, 
mostrarti le fatture in sospeso e fornirti le informazioni a te collegate. Provami!"""
)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "output": "Ciao! Come posso esserti utile?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["output"])

if prompt := st.chat_input("Di cosa hai bisogno?"):
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({"role": "user", "output": prompt})

    query = {"query": prompt}

    with st.spinner("in cerca di risposte..."):
        endpoint = f"{CHATBOT_URL}/send_query"
        response = requests.post(endpoint, json=query)

        if response.status_code == 200:
            output_text = response.json()

        else:
            output_text = """Errore durante la richiesta, riprovare."""

    st.chat_message("assistant").markdown(output_text)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "output": output_text,
        }
    )
