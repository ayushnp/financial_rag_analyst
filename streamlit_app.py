import streamlit as st
import requests
import os

st.set_page_config(page_title="Financial RAG Analyst", page_icon="📈")

st.title("📈 Financial 10-K Analyst")
st.markdown("Query your financial documents using **Llama 3.3 on Groq**")

# --- Sidebar: Upload ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a 10-K PDF", type="pdf")
    
    if st.button("Process Document"):
        if uploaded_file:
            with st.spinner("Processing..."):
                # Save temp file
                temp_path = os.path.join("data", uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Call your FastAPI Ingest Endpoint
                response = requests.post(f"http://127.0.0.1:8000/ingest?file_path={temp_path}")
                if response.status_code == 200:
                    st.success("Document Ingested Successfully!")
                else:
                    st.error("Error during ingestion.")
        else:
            st.warning("Please upload a PDF first.")

# --- Main Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask about the 10-K (e.g., 'What are the legal risks?')"):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI Ask Endpoint
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/ask",
                    json={"question": prompt}
                )
                data = response.json()
                
                answer = data.get("answer", "No answer received.")
                sources = data.get("sources", [])
                
                # Display Answer
                st.markdown(answer)
                
                # Display Sources in an accordion
                if sources:
                    with st.expander("View Sources"):
                        for source in sources:
                            st.write(f"📄 {source}")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Failed to connect to Backend: {e}")