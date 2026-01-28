import streamlit as st
import cohere
import wikipedia
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.llms import Cohere as CohereLLM
from langchain.embeddings.base import Embeddings
from pydantic import BaseModel, Field
from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 🌐 Page Config
st.set_page_config(page_title="Unified Skill India + Institution Info Chatbot", page_icon="🇮🇳", layout="centered")
st.title("🇮🇳 Skill India & Institution Info Assistant")

# 🔐 Set your Cohere API key
COHERE_API_KEY = "CgTgthJwHTXT9BBHnqinTrIiut864WPGUzMNYEtO"
co = cohere.Client(COHERE_API_KEY)

# 🧠 LLM and Embeddings
class CustomCohereEmbeddings(Embeddings):
    def embed_documents(self, texts):
        response = co.embed(texts=texts, model="embed-english-v3.0", input_type="search_document")
        return response.embeddings
    def embed_query(self, text):
        return self.embed_documents([text])[0]

embedding = CustomCohereEmbeddings()
llm = CohereLLM(cohere_api_key=COHERE_API_KEY, model="command-r-plus")

# --------------------- Part 1: Skill India QA --------------------- #
st.subheader("🔍 Ask about Skill India")

file_path = "skill_india.txt"
try:
    loader = TextLoader(file_path)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embedding)
    retriever = vectorstore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.text_input("Ask about Skill India:")
    if user_input:
        response = qa.run(user_input)
        st.session_state.history.append((user_input, response))
        for q, a in st.session_state.history:
            st.write(f"**You:** {q}")
            st.write(f"**AI:** {a}")
except FileNotFoundError:
    st.error(f"File {file_path} not found. Please upload skill_india.txt.")

# --------------------- Part 2: Institution Info Extractor --------------------- #
st.subheader("🏛️ Institution Info from Wikipedia")

class InstitutionDetails(BaseModel):
    founder: str = Field(..., description="Founder of the Institution")
    founded: str = Field(..., description="Year when it was founded")
    branches: List[str] = Field(..., description="Current branches in the institution")
    employees: str = Field(..., description="Number of employees")
    summary: str = Field(..., description="Brief 4-line summary")

parser = PydanticOutputParser(pydantic_object=InstitutionDetails)

prompt = PromptTemplate(
    template="""
Use the following Wikipedia content to answer the questions.
Return all fields as plain strings. If listing branches, format them as a Python list of strings.

Wikipedia content:
{context}

{format_instructions}

Institution name: {institution}
""",
    input_variables=["context", "institution"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

institution_name = st.text_input("Enter Institution Name:")
if st.button("Fetch Institution Info") and institution_name:
    try:
        wiki_text = wikipedia.page(institution_name).content[:2000]  # Limit content
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run({"context": wiki_text, "institution": institution_name})
        data = parser.parse(result)

        st.success("Institution Info Extracted:")
        st.markdown(f"**Founder:** {data.founder}")
        st.markdown(f"**Founded:** {data.founded}")
        st.markdown(f"**Branches:** {', '.join(data.branches)}")
        st.markdown(f"**Employees:** {data.employees}")
        st.markdown(f"**Summary:** {data.summary}")

    except wikipedia.exceptions.PageError:
        st.error("Institution not found on Wikipedia.")
    except Exception as e:
        st.error(f"Error: {str(e)}") 