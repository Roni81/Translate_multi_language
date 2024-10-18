from dotenv import load_dotenv
load_dotenv()

import json
import streamlit as st
import numpy as np
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


np.float_ = np.float64

st.title("Multi LanguageTranslator")
language = st.sidebar.radio("원하는 나라의 언어를 고르시오",
                            ("영어", "독일어", "프랑스어", "중국어")
            )
            
            
# city = st.text_input()



# 대화기록 리스트 작성
if 'message' not in st.session_state:
    st.session_state['message'] = []


# 이전대화 메세지 출력 함수 생성
def print_messages():
    for chat_message in st.session_state['message']:
        st.chat_message(chat_message.role).write(chat_message.content)

# 이전대화 메세지 출력 함수 저장
def save_message(role, message):
    st.session_state['message'].append(ChatMessage(role=role,content=message))




st.chat_message('ai').write('안녕하세요 Multi Language Translator 입니다, 해석을 원하는 문장을 입력하세요')
print_messages()

user_input = st.chat_input("Enter your Message")

if language == '영어':
    if user_input:
        #말풍선 생성 코드
        st.chat_message('user').write(user_input)
        save_message('user',user_input)
        with open('./en.json', encoding='utf-8') as f:
            json_data_en = f.read()  # Read the file contents as a string

        # Parse the JSON string into a Python object
        parsed_json_data_en = json.loads(json_data_en)

        # Now you can use parsed_json_data_en as a Python object
        
        
        example_prompt = ChatPromptTemplate.from_messages([ChatMessage(role='user',content='{instruction}\n\n--사용자 입력--\n{user}'),
                                                          ChatMessage(role='assistant',content='{ai}')
                                                    
        ])

        fewshot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples = parsed_json_data_en)

        content1 = """
한국어로 입력된 문장을 영어로 출력해 주세요 
번역된 영어 문장에서 키워드 3개를 추출해서 각각의 한국어 뜻과 영어 예문을 작성해주세요
출력은 아래의 출력 포맷에 따라 출력해주세요.

--사용자 입력--
{user_input}

--출력 포맷--

{{영어 번역}}

1. {{영어단어}}
- {{한국어 의미}}
- 영어 예문 : {{영어 예문}}{{한국어 번역}}
"""
        
        # final_prompt = ChatPromptTemplate.from_messages([
        #     fewshot_prompt,
        #     ChatMessage(role='user',content=content1)
        # ])

        final_prompt = ChatPromptTemplate.from_messages([
            fewshot_prompt,
            ('user',content1)
        ])
        model = ChatOpenAI(model_name = 'gpt-4o-mini',temperature=0.7)
        chain = final_prompt | model | StrOutputParser()

        answer = chain.invoke({"user_input":user_input}) 

        st.chat_message('ai').write(answer)
        save_message('ai', answer)




elif language == '독일어':
    if user_input:
        #말풍선 생성 코드
        st.chat_message('user').write(user_input)
        save_message('user',user_input)
        with open('./de.json', encoding='utf-8') as f:
            json_data_de = f.read()  # Read the file contents as a string

        # Parse the JSON string into a Python object
        parsed_json_data_de = json.loads(json_data_de)

        # Now you can use parsed_json_data_en as a Python object
        
        
        example_prompt = ChatPromptTemplate.from_messages([ChatMessage(role='user',content='{instruction}\n\n--사용자 입력--\n{user}'),
                                                          ChatMessage(role='assistant',content='{ai}')
                                                    
        ])

        fewshot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples = parsed_json_data_de)

        content1 = """
한국어로 입력된 문장을 독일어로 출력해 주세요 
번역된 독일어 문장에서 키워드 3개를 추출해서 각각의 한국어 뜻과 독일어 예문을 작성해주세요
출력은 아래의 출력 포맷에 따라 출력해주세요.

--사용자 입력--
{user_input}

--출력 포맷--

{{독일어 번역}}

1. {{독일어 단어}}
- {{한국어 의미}}
- 독일어 예문 : {{독일어 예문}}{{한국어 번역}}
"""
        
        # final_prompt = ChatPromptTemplate.from_messages([
        #     fewshot_prompt,
        #     ChatMessage(role='user',content=content1)
        # ])

        final_prompt = ChatPromptTemplate.from_messages([
            fewshot_prompt,
            ('user',content1)
        ])
        model = ChatOpenAI(model_name = 'gpt-4o-mini',temperature=0.7)
        chain = final_prompt | model | StrOutputParser()

        answer = chain.invoke({"user_input":user_input}) 

        st.chat_message('ai').write(answer)
        save_message('ai', answer)


elif language == '프랑스어':
    if user_input:
        #말풍선 생성 코드
        st.chat_message('user').write(user_input)
        save_message('user',user_input)
        with open('./fr.json', encoding='utf-8') as f:
            json_data_fr = f.read()  # Read the file contents as a string

        # Parse the JSON string into a Python object
        parsed_json_data_fr = json.loads(json_data_fr)

        # Now you can use parsed_json_data_en as a Python object
        
        
        example_prompt = ChatPromptTemplate.from_messages([ChatMessage(role='user',content='{instruction}\n\n--사용자 입력--\n{user}'),
                                                          ChatMessage(role='assistant',content='{ai}')
                                                    
        ])

        fewshot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples = parsed_json_data_fr)

        content1 = """
한국어로 입력된 문장을 프랑스어로 출력해 주세요 
번역된 프랑스어 문장에서 키워드 3개를 추출해서 각각의 한국어 뜻과 프랑스어 예문을 작성해주세요
출력은 아래의 출력 포맷에 따라 출력해주세요.

--사용자 입력--
{user_input}

--출력 포맷--

{{프랑스어 번역}}

1. {{프랑스어 단어}}
- {{한국어 의미}}
- 프랑스어 예문 : {{프랑스어 예문}}{{한국어 번역}}
"""
        
        # final_prompt = ChatPromptTemplate.from_messages([
        #     fewshot_prompt,
        #     ChatMessage(role='user',content=content1)
        # ])

        final_prompt = ChatPromptTemplate.from_messages([
            fewshot_prompt,
            ('user',content1)
        ])
        model = ChatOpenAI(model_name = 'gpt-4o-mini',temperature=0.7)
        chain = final_prompt | model | StrOutputParser()

        answer = chain.invoke({"user_input":user_input}) 

        st.chat_message('ai').write(answer)
        save_message('ai', answer)




elif language == '중국어':
    if user_input:
        #말풍선 생성 코드
        st.chat_message('user').write(user_input)
        save_message('user',user_input)
        with open('./fr.json', encoding='utf-8') as f:
            json_data_cn = f.read()  # Read the file contents as a string

        # Parse the JSON string into a Python object
        parsed_json_data_cn = json.loads(json_data_cn)

        # Now you can use parsed_json_data_en as a Python object
        
        
        example_prompt = ChatPromptTemplate.from_messages([ChatMessage(role='user',content='{instruction}\n\n--사용자 입력--\n{user}'),
                                                          ChatMessage(role='assistant',content='{ai}')
                                                    
        ])

        fewshot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples = parsed_json_data_cn)

        content1 = """
한국어로 입력된 문장을 중국어로 출력해 주세요 
번역된 중국어 문장에서 키워드 3개를 추출해서 각각의 한국어 뜻과 중국어 예문을 작성해주세요
출력은 아래의 출력 포맷에 따라 출력해주세요.

--사용자 입력--
{user_input}

--출력 포맷--

{{중국어 번역}}

1. {{중국어 단어}}
- {{한국어 의미}}
- 중국어 예문 : {{중국어 예문}}{{한국어 번역}}
"""
        
        # final_prompt = ChatPromptTemplate.from_messages([
        #     fewshot_prompt,
        #     ChatMessage(role='user',content=content1)
        # ])

        final_prompt = ChatPromptTemplate.from_messages([
            fewshot_prompt,
            ('user',content1)
        ])
        model = ChatOpenAI(model_name = 'gpt-4o-mini',temperature=0.7)
        chain = final_prompt | model | StrOutputParser()

        answer = chain.invoke({"user_input":user_input}) 

        st.chat_message('ai').write(answer)
        save_message('ai', answer)


