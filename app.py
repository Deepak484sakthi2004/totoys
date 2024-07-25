from flask import Flask, request, jsonify, render_template
import os
import whisper
from pathlib import Path

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import warnings
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from audioToText import audio_to_transcript

from model import *

classFetch = "class8"



app = Flask(__name__)

# importin the embedding model from hugging face
model_name = "BAAI/bge-base-en"
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

# creating an object for embedding model
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},  # use cuda, if gpu is available
    encode_kwargs=encode_kwargs
)

def load_audio_model():
    model = whisper.load_model("base")
    return model


def prettify_text(text):
    prettified = text.replace('**', '\n').replace('*', '\n')
    return prettified  



# Define the function to get the conversational chain
def get_conversational_chain(expertLevel):
    prompt_template = f"""
        Chat History: {chat_history}
    system
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    Use three points minimum and keep the answer concise. <eot_id><start_header_id>user<end_head_id>
    Question: {user_question}
    Context: {context}
    Student_expertise_level : {expertLevel}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "user_question", "chat_history"])
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, google_api_key='AIzaSyClo-Pfhrww33nYHWXNW_UbjXa7pVggGtM')
    chain = LLMChain(
        llm=model,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )
    return chain

def generate_answer_llm(user_question,expertLevel):
    context = getContext(classFetch,user_question)
   # context = " ".join([doc.page_content for doc in docs])
    chain = get_conversational_chain(expertLevel)
    response = chain.predict(input=user_question, context=context)
    print(response)
    return response, context

# def get_conversational_chain(question, context, expertLevel):
#     print("expertLevel ",expertLevel)
#     prompt_template = """
#         system
#         You are an intelligent school teacher capable of answering students' questions based on relevant information retrieved from a database. Your responses should be clear, informative, and suitable for educational purposes.
#         Use the following pieces of retrieved context to answer the question to the students. Make sure the answer is self-explanatory and understandable for children to learn from you!
#         Use three points minimum and keep the answer concise.
#         Every student cannot understand your generic answer, so the student himself speficies his/her knowledge in this specific question. The knowledge level is from 1 to 10, where 1 implies no knowledge in the question/domain, so explain it accordingly from basics. 
#         Level 10 implies, the student is well expertised in the topic and you have to answer to the question at first however and ask him a challenging question related to the topic!.

#         <eot_id><start_header_id>user<end_head_id>
#         Question: {question}
#         Context: {context}
#         Student_expertise_level : {expertLevel}

#         Answer: 
#     """
    
#     prompt = PromptTemplate(template=prompt_template, input_variables=["question", "context", "expertLevel"])
#     # replace the gemini api 
#     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, google_api_key='AIzaSyClo-Pfhrww33nYHWXNW_UbjXa7pVggGtM')
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt, document_variable_name="context")
    
#     return chain
##=========================================================================================================================
# def generate_answer_llm(user_question,expertLevel):
#     context = getContext(classFetch,user_question)
#     out = ""
#     for doc in context:
#         out += doc.page_content +"\n======================\n"
#     chain = get_conversational_chain(user_question, out,expertLevel)
#     response = chain({"input_documents": context, "question": user_question,"expertLevel":expertLevel}, return_only_outputs=True)
#     return response["output_text"], out


# # @app.route('/')
# # def meeting():

# #     return render_template('recordAudio.html')


# # @app.route('/save_audio', methods=['POST'])
# # def save_audio():
# #     audio_data = request.files['audio']
# #     expertLevel = int(request.form['volumeLevel'])
# #     print("slider: ",expertLevel)
# #     # Here you can save the audio data to a file or a database
# #     # Example:
# #     audio_filename = 'recorded_audio.wav'
# #     audio_data.save(audio_filename)
# #     out = audio_to_transcript('recorded_audio.wav')
# #     print("\n\n\n----------------------------USER QUESTION !!----------------------------------\n")
# #     print(out)
    
# #     generated_text, relevant_docs = generate_answer_llm(out,expertLevel)
# #     print("\n\n----------------------------GENERATED ANSWER !!----------------------------------\n")
# #     print(generated_text)
# #     print("\n\n----------------------------RELEVANT CONTEXT !!----------------------------------\n")
# #     print(relevant_docs)
# #     # x = temp(generated_text) # remove 
# #     # print(x)
# #     # speak(x) 
# #     return jsonify({'generated_text': prettify_text(generated_text), 'relevant_docs': prettify_text(relevant_docs)})


# # @app.route('/QA', methods=['POST'])
# # def QA():
# #     question = request.form['question']
# #     expertLevel = int(request.form['volumeLevel'])
# #     print("slider: ",expertLevel)
# #     # Here you can save the audio data to a file or a database
# #     # Example:
# #     print("\n\n\n----------------------------USER QUESTION !!----------------------------------\n")
# #     print(out)
    
# #     generated_text, relevant_docs = generate_answer_llm(out,expertLevel)
# #     print("\n\n----------------------------GENERATED ANSWER !!----------------------------------\n")
# #     print(generated_text)
# #     print("\n\n----------------------------RELEVANT CONTEXT !!----------------------------------\n")
# #     print(relevant_docs)
# #     return jsonify({'generated_text': prettify_text(generated_text), 'relevant_docs': prettify_text(relevant_docs)})



@app.route('/')
def main_page():
    return render_template('index.html')

@app.route('/audio')
def audio_page():
    english = request.args.get('english')
    maths = request.args.get('maths')
    science = request.args.get('science')
    socialScience = request.args.get('socialScience')
    return render_template('recordAudio1.html', english=english, maths=maths, science=science, socialScience=socialScience)

@app.route('/qa')
def qa_page():
    english = request.args.get('english')
    maths = request.args.get('maths')
    science = request.args.get('science')
    socialScience = request.args.get('socialScience')
    return render_template('qa.html', english=english, maths=maths, science=science, socialScience=socialScience)

########################################################################

@app.route('/save_audio', methods=['POST'])
def save_audio():
    audio_data = request.files['audio']
    english = int(request.form['english'])
    maths = int(request.form['maths'])
    science = int(request.form['science'])
    socialScience = int(request.form['socialScience'])
    print("Levels - English: {}, Maths: {}, Science: {}, Social Science: {}".format(english, maths, science, socialScience))
    audio_filename = 'recorded_audio.wav'
    audio_data.save(audio_filename)
    audio_model = load_audio_model()
    out = audio_to_transcript(audio_filename,audio_model)
    print("\n\n\n----------------------------USER QUESTION !!----------------------------------\n")
    print(out)
    print("\n\n------------------------------FIND SUBJECT!-----------------------------------\n")
    subject = findSubject(question) # english, maths, science, socialScience
    print(subject)
    generated_text, relevant_docs = generate_answer_llm(out,subject)
    print("\n\n----------------------------GENERATED ANSWER !!----------------------------------\n")
    print(generated_text)
    print("\n\n----------------------------RELEVANT CONTEXT !!----------------------------------\n")
    print(relevant_docs)
    return jsonify({'generated_text': prettify_text(generated_text), 'relevant_docs': prettify_text(relevant_docs)})


@app.route('/qa', methods=['POST'])
def qa():
    data = request.get_json()
    question = data['question']
    english = int(data['english'])
    maths = int(data['maths'])
    science = int(data['science'])
    socialScience = int(data['socialScience'])

    print("\n\n\n----------------------------USER QUESTION !!----------------------------------\n")
    print(question)

    subject = findSubject(question) # english, maths, science, socialScience
    print(subject)

    generated_text, relevant_docs = generate_answer_llm(question,subject)
    print("\n\n----------------------------GENERATED ANSWER !!----------------------------------\n")
    print(generated_text)
    print("\n\n----------------------------RELEVANT CONTEXT !!----------------------------------\n")
    print(relevant_docs)
    return jsonify({'generated_text': prettify_text(generated_text), 'relevant_docs': prettify_text(relevant_docs)})



if __name__ == '__main__':
    app.run(debug=True)
