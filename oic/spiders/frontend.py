import streamlit as st
import pickle
import warnings
warnings.filterwarnings("ignore")
from langchain.llms import CTransformers
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Configure page
st.set_page_config(
    page_title="OIC Documentation Assistant",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_llm():
    """Load the Llama2 model"""
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q4_0.bin", 
        model_type="llama",
        config={
            'max_new_tokens': 600,
            'temperature': 0.01,
            'context_length': 5000
        }
    )
    return llm

@st.cache_resource
def load_embeddings():
    """Load the embedding model"""
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings

def load_vector_store(file_path):
    """Load the pretrained FAISS index"""
    try:
        with open(file_path, "rb") as f:
            vector_store = pickle.load(f)
        return vector_store
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

def initialize_chain(llm, vector_store):
    """Initialize the conversation chain"""
    template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
    Chat History:
    {chat_history}
    Follow up Input: {question}
    Standalone questions: """
    
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(template)
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        return_source_documents=True,
        verbose=True
    )
    return chain

def main():
    st.title("🔍 OIC Documentation Assistant")
    st.markdown("""
    Ask questions about Oracle Integration Cloud (OIC) and get answers based on the documentation.
    """)
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Load models and vector store
    with st.spinner("Loading models..."):
        llm = load_llm()
        vector_store = load_vector_store("oic_llama.pkl")  # Update path as needed
        
        if vector_store is None:
            st.error("Failed to load the vector store. Please check if the .pkl file exists.")
            return
        
        chain = initialize_chain(llm, vector_store)
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This assistant helps you find information about Oracle Integration Cloud (OIC).
        It uses:
        - Llama2 7B model
        - FAISS vector store
        - BGE embeddings
        """)
        
        # Add a clear chat history button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            st.chat_message("user").write(message.content)
        elif isinstance(message, AIMessage):
            st.chat_message("assistant").write(message.content)
    
    # Query input
    query = st.chat_input("Ask your question about OIC:")
    
    if query:
        st.chat_message("user").write(query)
        
        # Show spinner while processing
        with st.spinner("Searching for answer..."):
            try:
                # Get response from chain
                result = chain.invoke({
                    "question": query,
                    "chat_history": st.session_state.chat_history
                })
                
                # Display answer
                answer = result["answer"]
                st.chat_message("assistant").write(answer)

                #Display Sources
                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources:")
                    sources_list = sources.split("\n")  # Split the sources by newline
                    for source in sources_list:
                        st.write(source)
                
                # Update chat history
                st.session_state.chat_history.extend([
                    HumanMessage(content=query),
                    AIMessage(content=answer)
                ])
                
                # Display source documents if available
                if "source_documents" in result:
                    with st.expander("View Source Documents"):
                        for doc in result["source_documents"]:
                            st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                            st.markdown(f"**Page:** {doc.metadata.get('page', 'Unknown')}")
                            st.markdown("**Content:**")
                            st.markdown(doc.page_content)
                            st.divider()
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.markdown("""
                    Common issues:
                    - Check if the model files are in the correct location
                    - Verify that all required dependencies are installed
                    - Ensure the question is clear and well-formed
                    """)

if __name__ == "__main__":
    main()