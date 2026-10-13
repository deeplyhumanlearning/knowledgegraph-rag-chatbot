from collections import Counter

import streamlit as st

from kgrag.guard import REFUSAL, check_input, check_output
from kgrag.rag import GraphRAG

st.set_page_config(page_title="TN Schemes Assistant", page_icon="🌾")
st.title("🌾 TN Schemes Assistant")
st.caption("Ask about Tamil Nadu government schemes for farmers and MSMEs.")


@st.cache_resource
def get_rag() -> GraphRAG:
    return GraphRAG()


rag = get_rag()

with st.sidebar:
    st.subheader("Knowledge graph")
    st.write(f"{rag.g.number_of_nodes()} nodes · {rag.g.number_of_edges()} edges")
    counts = Counter(d["type"] for _, d in rag.g.nodes(data=True))
    st.dataframe({"Type": list(counts), "Count": list(counts.values())}, hide_index=True)
    st.subheader("Try asking")
    st.markdown(
        "- Which schemes support groundnut farmers?\n"
        "- What subsidy is given for gypsum?\n"
        "- How many schemes provide incentives?\n"
        "- What is the subsidy on patent registration?"
    )
    if st.button("Clear chat"):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if question := st.chat_input("Ask about a scheme..."):
    st.chat_message("user").markdown(question)
    with st.chat_message("assistant"), st.spinner("Thinking..."):
        sources = []
        if not check_input(question):
            answer = REFUSAL
        else:
            result = rag.answer(question, st.session_state.messages)
            answer = result["answer"] if check_output(question, result["answer"]) else REFUSAL
            sources = result["sources"]
        st.markdown(answer)
        if sources and answer != REFUSAL:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"- [{s['name']}]({s['url']})")
    st.session_state.messages += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
