import streamlit as st
import pandas as pd
from typing import Dict, Tuple, Optional
from pubmed_parser import *

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
    st.write("NCBI PubMed, PMC 에서 논문을 검색하고 결과를 확인하세요.")
    
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

    # 🔥 사용자가 검색할 논문 개수를 조정할 수 있도록 설정
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
            num_results = len(results)  # Dictionary의 키 개수 = 논문 개수
        
            if num_results > 0:
                st.success(f"총 {num_results}개의 논문을 찾았습니다!")
                display_results(results)
            else:
                st.warning("검색 결과가 없습니다.")



if __name__ == "__main__":
    main()
