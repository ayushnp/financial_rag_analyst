import streamlit as st
import requests
import os

# Your Production Backend URL
BACKEND_URL = "https://financial-rag-analyst.onrender.com"

st.set_page_config(page_title="DeepAudit.ai", page_icon="📈")

st.title("📈 DeepAudit.ai")
st.markdown("Query your financial documents using **Llama 3.3 on Groq**")

# Ensure the 'data' directory exists for temporary file storage
if not os.path.exists("data"):
    os.makedirs("data")

# --- Sidebar: Upload ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a 10-K PDF", type="pdf")
    
    if st.button("Process Document"):
        if uploaded_file:
            with st.spinner("Uploading and Processing..."):
                # Prepare the file to be sent over the internet
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                
                try:
                    # Send the actual file data to Render
                    response = requests.post(f"{BACKEND_URL}/ingest", files=files)
                    
                    if response.status_code == 200:
                        st.success("Document Ingested Successfully!")
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")
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

    # Call LIVE Render Ask Endpoint
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={"question": prompt}
                )
                
                if response.status_code == 200:
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
                else:
                    st.error(f"Backend returned an error: {response.status_code}")
                
            except Exception as e:
                st.error(f"Failed to connect to Backend: {e}")
