# app.py

import streamlit as st
import sys
import io
import contextlib
import querydata

st.set_page_config(page_title="RAG QA System", layout="centered")

st.title("📚 RAG Question Answering System")

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        try:
            # Simulate CLI argument
            sys.argv = ["querydata.py", query]

            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                querydata.main()

            full_output = buffer.getvalue()

            # Extract only the line starting with "Response:"
            response_text = ""

            for line in full_output.splitlines():
                if line.startswith("Response:"):
                    response_text = line.replace("Response:", "").strip()
                    break

            if response_text:
                st.success(response_text)
            else:
                st.warning("No response found.")

        except Exception as e:
            st.error(f"Error: {e}")
