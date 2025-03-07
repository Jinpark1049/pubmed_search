import streamlit as st
import pandas as pd
from typing import Dict, Tuple, Optional
from pubmed_parser import *
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis  
import os     
import re

nltk.download('stopwords')
nltk.download('wordnet')


stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def lda_vis(results, n_components=5):
    abstracts = [data[2] for data in results.values() if data[2]]  # 초록이 None이 아닌 경우만 추출
    if not abstracts:
        st.error("초록이 포함된 논문이 없습니다. LDA 시각화를 실행할 수 없습니다.")
        return
    
    processed_abstracts = [preprocess_text(abstract) for abstract in abstracts]
    
    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(processed_abstracts)

    # LDA 모델 학습
    lda_model = LatentDirichletAllocation(n_components=n_components, random_state=42)
    lda_model.fit(X)
    
    # pyLDAvis.prepare에 필요한 데이터 준비
    topic_term_dists = lda_model.components_ / lda_model.components_.sum(axis=1)[:, None]  # 토픽-단어 분포 정규화
    doc_topic_dists = lda_model.transform(X)  # 문서-토픽 분포
    doc_lengths = X.sum(axis=1).A1  # 각 문서의 단어 수 (행렬을 1D 배열로 변환)
    vocab = vectorizer.get_feature_names_out()  # 단어 사전
    term_frequency = X.sum(axis=0).A1  # 전체 말뭉치에서 단어 빈도 (행렬을 1D 배열로 변환)

    # LDA 시각화 준비
    lda_vis_data = pyLDAvis.prepare(
        topic_term_dists=topic_term_dists,
        doc_topic_dists=doc_topic_dists,
        doc_lengths=doc_lengths,
        vocab=vocab,
        term_frequency=term_frequency,
        mds='tsne'
    )
    html_string = pyLDAvis.prepared_data_to_html(lda_vis_data)
    st.components.v1.html(html_string, height=800)

    # 주요 단어 추출
    n_words = 10
    words = vectorizer.get_feature_names_out()
    for idx, topic in enumerate(lda_model.components_):
        st.write(f"Topic {idx + 1}:")
        st.write([words[i] for i in topic.argsort()[-n_words:]])
        
    # 주제 예측
    topic_predictions = lda_model.transform(X)
    for i, topic_probabilities in enumerate(topic_predictions):
        dominant_topic = topic_probabilities.argmax()
        st.write(f"문서 {i+1}의 주요 주제: {dominant_topic}, 확률: {topic_probabilities[dominant_topic]}")
        
def preprocess_text(text):
    text = text.lower()  # 소문자화
    tokens = re.findall(r'\b\w+\b', text)  # 단어 경계로 알파벳/숫자 토큰 추출
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

def display_results(results: Dict[str, Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]]):
    """결과를 DataFrame으로 변환하여 Streamlit에서 표시."""
    df = pd.DataFrame(
        [
            {   "DOI": doi if doi else "Not Available",
                "PMID": pmid,
                "PMCID": pmcid if pmcid else "Not Available",
                "Title": title,
                "Authors": authors,
                "Year": pub_year,
                "Journal Title": journal_title if journal_title else "Not Available",  # 저널명 추가
                "MeSH Keywords": mesh_keywords if mesh_keywords else "Not Available", # MeSH 키워드 추가
                "Abstract": abstract,

            }
            for pmid, (pmcid, doi, abstract, title, authors, pub_year, journal_title, mesh_keywords) in results.items()
        ]
    )
    st.dataframe(df)
    
def main():
    st.title("NCBI 논문 검색 서비스")
    st.write("NCBI PubMed, PMC에서 논문을 검색하고 결과를 확인하세요.")
    
    # 세션 상태 초기화
    if 'results' not in st.session_state:
        st.session_state.results = None

    use_llm = st.checkbox("LLM 사용하여 검색", value=True)

    if use_llm:
        user_input = st.text_area("검색 조건 입력", "제목에 cancer 있고 초록에 gene therapy가 포함된 2024년 오픈 액세스 논문")
        st.warning("LLM을 사용하여 자동 키워드를 생성합니다.")
        search_term = "AI-generated keyword"  # 실제로는 LLM 호출하여 키워드 생성
        search_year = "all"
        abstract_keyword = ""
    else:
        search_term = st.text_input("Enter keyword:", value="cancer")
        search_year = st.text_input("Enter year (or 'all' for all years):", value="all")
        abstract_keyword = st.text_input("Filter abstracts containing (or 'none' for all):", value="none")

    max_results = st.number_input("검색할 논문 개수", min_value=1, max_value=10000, value=10, step=1)

    if st.button("논문 검색"):
        with st.spinner("논문을 검색 중입니다..."):
            term = text_generator(
                user_input=user_input if use_llm else None,
                keyword=search_term,
                abstract_keyword=abstract_keyword,
                year=search_year,
                free_full_text=True,
                use_llm=use_llm
            )
            results = fetch_pubmed(term, max_results=max_results)
            st.session_state.results = results  # 검색 결과를 세션 상태에 저장
            num_results = len(results)
        
            if num_results > 0:
                st.success(f"총 {num_results}개의 논문을 찾았습니다!")
                display_results(results)
            else:
                st.warning("검색 결과가 없습니다.")

    # LDA 시각화 버튼
    if st.session_state.results and st.button("LDA 시각화"):
        with st.spinner("LDA 시각화를 생성 중입니다..."):
            n_components = st.slider("토픽 수 설정", min_value=2, max_value=10, value=5)
            lda_vis(st.session_state.results, n_components=n_components)

if __name__ == "__main__":
    main()
