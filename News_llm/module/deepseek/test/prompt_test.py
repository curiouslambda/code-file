from deep_seek_client import DeepSeekClient

context = r"""아마존이 4분기 실적을 발표했으나 1분기 전망이 시장 기대치를 하회하면서 시간외 거래에서 주가가 하락함.
인공지능(AI) 인프라 확충에 적극 투자 중으로, 4분기 자본지출이 전년 146억 대비 두 배 가까이 오른 278억 달러를 기록함.
2025년에는 자본지출을 1,000억 달러까지 늘릴 계획이며, 이는 2024년의 약 830억 달러보다 크게 증가한 수치임.
아마존은 단기적으로 달러 강세에 따른 매출 타격을 경고하며, 1분기 매출 성장률이 5~9%로 둔화될 수 있다고 전함;
이는 1997년 상장 이후 최저 성장률이 될 수 있음.
앤디 재시 CEO는 이번 AI 투자 확대가 '일생에 한 번 있는 사업 기회'라며, 중장기적 수익성과 시장 주도권 확보에 대한 자신감을 보임.

실적발표:
- EPS: $1.86 (예상: $1.50)
- 매출: $1877.9억 (예상: $1873.2억)
- AWS 매출: $287.9억 (예상: $288.2억)
- AWS 매출 (환율 제외): +19% (예상: 19%)
- 1분기 매출 전망: $1510~$1555억 (예상: $1586.4억)
"""

prompt_template = """
1. Extract relevant tags (industries, companies, market keywords, etc.) from the given context in the form of a list.
2. If it is not possible to determine the tags with certainty, return the message: "Error: Unable to determine relevant tags."
3. Always return the answer strictly as a JSON-formatted list.
4. Output only the result list without any explanations.


context: {context}

answer:
"""

# 프롬프트를 context와 함께 포맷팅
prompt = prompt_template.format(context=context)

# DeepSeekClient 인스턴스 생성
client = DeepSeekClient()

# 모델 실행
answer = client.ask(prompt)

client.save_result_csv(csv_path='prompt_test.csv', prompt_text=prompt, response=answer)

# 결과 출력
print(answer)
