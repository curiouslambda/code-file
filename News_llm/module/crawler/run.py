import requests
from bs4 import BeautifulSoup
from collections import defaultdict

def fetch_webpage(url):
    """주어진 URL의 웹페이지 HTML을 가져옵니다."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Failed to fetch the webpage. Status code: {response.status_code}")

def extract_headings_and_body(html):
    """HTML에서 <h2> 태그의 제목과 해당하는 <p> 태그들의 본문을 추출합니다."""
    soup = BeautifulSoup(html, 'html.parser')
    extracted_data = defaultdict(list)  # 제목을 키로 하고 본문을 리스트로 저장
    
    paragraphs = soup.find_all('p')  # 본문 내용
    for paragraph in paragraphs:
        heading = paragraph.find_previous('h2')  # 현재 <p> 태그의 가장 가까운 이전 <h2> 태그 찾기
        title = heading.get_text(strip=True) if heading else "No Title"
        text = paragraph.get_text(strip=True)
        if text:  # 본문 내용이 비어있지 않은 경우만 추가
            extracted_data[title].append(text)
    
    return {title: "\n".join(body) if body else "No Body" for title, body in extracted_data.items()}  # 본문이 없으면 "No Body" 반환

def scrape_webpage_with_headings(url):
    """웹페이지에서 제목(<h2>)과 본문(<p>)을 추출합니다."""
    html = fetch_webpage(url)
    extracted_texts = extract_headings_and_body(html)
    return extracted_texts

if __name__ == "__main__":
    test_url = "https://futuresnow.gitbook.io/newstoday/news/today/bloomberg"
    extracted_data = scrape_webpage_with_headings(test_url)
    
    print("Extracted Titles and Body Text:")
    for title, body in extracted_data.items():
        print(f"Title: {title}\nBody:\n{body}\n")
