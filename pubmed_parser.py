import re
import requests, time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
import ollama

def text_generator(
    user_input: Optional[str] = None,
    keyword: str = 'cancer',
    abstract_keyword: str = 'none',
    year: str = '2024',
    free_full_text: bool = True,
    use_llm: bool = False
    ) -> str:
    """
    NCBI PubMed 검색에 적합한 'term' 쿼리를 생성하는 함수.
    - use_llm=True: LLM(Ollama)을 사용해 자연어 입력을 처리.
    - use_llm=False: 규칙 기반으로 keyword, year, open_access를 조합.
    
    Args:
        user_input (str, optional): LLM 사용 시 자연어 입력 (예: "제목에 cancer 있고...").
        keyword (str): 기본 키워드 (기본값: 'cancer').
        year (str): 출판 연도 (기본값: '2024').
        free full text (bool): Open Access 필터 추가 여부 (기본값: True).
        use_llm (bool): LLM 사용 여부 (기본값: False).
    
    Returns:
        str: NCBI PubMed 검색에 사용할 'term' 문자열.
    """
    if use_llm:
        # LLM 사용 시 keyword와 year를 프롬프트에 포함
        if not user_input:
            user_input = f"{keyword} 관련 논문 중 {year}년에 출판된 것"
        query_prompt = f"""
        사용자가 다음과 같은 자연어로 검색 요청을 했습니다: '{user_input}'  
        이를 NCBI PubMed 검색에 적합한 'term' 쿼리로 변환해주세요.  
        - 검색 필드(예: [Title], [Abstract], [Author])와 논리 연산자(AND, OR, NOT)를 적절히 사용하세요.  
        - 구문 검색이 필요하면 쌍따옴표("...")를 사용하세요.  
        - free full text or open acess 논문만 찾으려면 'free full text[filter]'를 추가하세요.  
        - 년도를 지정하려면 [pdat]을 추가하세요. 예시) 2023년의 경우 2023[pdat]
        - 결과는 'term' 값으로 바로 사용할 수 있는 문자열 (String) 로 반환하세요. 설명하지 말고 결과만 반환하세요.
        """
        response = ollama.chat(
            model="gemma2:27b",
            messages=[{"role": "user", "content": query_prompt}]
        )
        generated_term = response["message"]["content"].strip()
        print(f"LLM 생성 term: {generated_term}")
        return generated_term
    else:
        # 규칙 기반 term 생성
        term = keyword.lower()
        if abstract_keyword:
            if not abstract_keyword == "none":
                term += f" AND {abstract_keyword}[Abstract]"
        if year:
            if not year == "all":
                term += f" AND {year}[pdat]"
        if free_full_text:
            term += " AND free full text[filter]"
        print(f"규칙 기반 term: {term}")
        return term

def fetch_pubmed(term: str = 'cancer', max_results=20, api_key=None) -> Dict[str, Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]]:
    
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    elink_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    base_params = {"api_key": api_key} if api_key else {}
    
    esearch_params = base_params | {
        "db": "pubmed",
        "term": term,
        "retmax": max_results,
        "retmode": "json"
    }
        
    try:
        esearch_response = requests.get(esearch_url, params=esearch_params)
        esearch_response.raise_for_status()
        pmids = esearch_response.json()["esearchresult"]["idlist"]
        print(pmids)
        if not pmids:
            return {}
    except (requests.RequestException, KeyError) as e:
        print(f"ESearch 실패: {e}")
        return {}
    
    time.sleep(0.1)    
    
    elink_params = base_params | {
        "dbfrom": "pubmed",
        "db": "pmc",
        "id": pmids,
        "retmode": "xml"
    }
    try:
        elink_response = requests.get(elink_url, params=elink_params)
        elink_response.raise_for_status()
        elink_root = ET.fromstring(elink_response.text)
        
    except (requests.RequestException, ET.ParseError) as e:
        print(f"ELink 실패: {e}")
        return {pmid: (None, None, None, None, None, None) for pmid in pmids}
        
    pmid_pmcid_map = {pmid: None for pmid in pmids}
    
    for linkset in elink_root.findall(".//LinkSet"):
        pmid = linkset.find(".//IdList/Id").text
        pmcid_elem = linkset.find(".//Link/Id")
        if pmcid_elem is not None:
            pmid_pmcid_map[pmid] = pmcid_elem.text

    efetch_params = base_params | {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml"
    }
    
    time.sleep(0.1)
    
    try:
        efetch_response = requests.get(efetch_url, params=efetch_params)
        efetch_response.raise_for_status()
        efetch_root = ET.fromstring(efetch_response.text)
    except (requests.RequestException, ET.ParseError) as e:
        print(f"EFetch 실패: {e}")
        return {pmid: (pmid_pmcid_map[pmid], None, None, None, None, None) for pmid in pmids}
    
    # 논문 데이터 저장
    pmid_data_map = {}
    for article in efetch_root.findall(".//PubmedArticle"):
        pmid = article.find(".//PMID").text
        doi_elem = article.find(".//ArticleId[@IdType='doi']")
        abstract_elem = article.find(".//Abstract/AbstractText")
        title_elem = article.find(".//ArticleTitle")
        author_list_elem = article.findall(".//AuthorList/Author")
        pub_year_elem = article.find(".//PubDate/Year")
        medline_date_elem = article.find(".//PubDate/MedlineDate")
        journal_title_elem = article.find(".//Journal/Title")  # 저널명
        mesh_heading_elem = article.findall(".//MeshHeadingList/MeshHeading")  # MeSH 용어 추가

        # DOI, 초록, 제목 추출
        doi = doi_elem.text if doi_elem is not None else None
        abstract = abstract_elem.text if abstract_elem is not None else "Not Available"
        title = title_elem.text if title_elem is not None else "Not Available"

        # 저자 목록 추출 (이름 성 형식으로 변환)
        authors = []
        for author in author_list_elem:
            last_name = author.find("LastName")
            fore_name = author.find("ForeName")
            if last_name is not None and fore_name is not None:
                authors.append(f"{fore_name.text} {last_name.text}")
        authors_str = ", ".join(authors) if authors else "Not Available"

        # 발행 연도 추출
        if pub_year_elem is not None:
            pub_year = pub_year_elem.text
        elif medline_date_elem is not None:
            pub_year = medline_date_elem.text.split()[0]  # "2023 Jul-Aug" 같은 형식이면 첫 번째 값(연도)만 가져오기
        else:
            pub_year = "Not Available"

        # 저널명 추출
        journal_title = journal_title_elem.text if journal_title_elem is not None else "Not Available"

        # MeSH 키워드 추출
        mesh_keywords = []
        for mesh_heading in mesh_heading_elem:
            mesh_term = mesh_heading.find("DescriptorName")
            if mesh_term is not None:
                mesh_keywords.append(mesh_term.text)
        mesh_keywords_str = ", ".join(mesh_keywords) if mesh_keywords else "Not Available"

        # 데이터 저장 (MeSH 키워드 추가)
        pmid_data_map[pmid] = (pmid_pmcid_map.get(pmid), doi, abstract, title, authors_str, pub_year, journal_title, mesh_keywords_str)


    return pmid_data_map


if __name__ == "__main__":
    user_input = "제목에 cancer 있고 초록에 gene therapy가 포함된 오픈 액세스 논문" 
    user_input = str(input("찾고자 하는 논문을 작성해주세요: 예) 제목에 cancer 있고 초록에 gene therapy가 포함된 오픈 액세스 논문: "))
    term_llm = text_generator(
            user_input,
            keyword="cancer",
            abstract_keyword="none",
            year="2024",
            free_full_text=True,
            use_llm=True
        )
    # 2. 규칙 기반 사용
    term = text_generator(
            keyword="cancer",
            year="2024",
            abstract_keyword="none",
            free_full_text=True,
            use_llm=False
        )
    
    result = fetch_pubmed(term, max_results=10)
    # 결과 출력
    print(f"Total number of papers found: {len(result)}")
    for pmid, (pmcid, doi, abstract, title, authors, pub_year) in result.items():
        print(f"PMID: {pmid}, Title: {title}, Authors: {authors}, Year: {pub_year}, PMCID: {pmcid if pmcid else 'Not Available'}, DOI: {doi if doi else 'Not Available'}")
