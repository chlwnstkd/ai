from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
import re
okt=Okt()

# .제거
token=re.sub("(\.)","","힌극폴리텍대학 서울 강서캠퍼스 데이터 분석과 이협건 교수는"
                       "한국폴리텍대학에서 데이터분석 과목과 인공지능 과목을 교육하는 교수이다.")

# 형태소 단위로 나누기
token=okt.morphs(token)

# 형태소 단위로 분리도니 벡터 데이터 구조를 문장 형태로 변환
# 문장 형태는 벡터마다 띄어쓰기 추가함
corpus = [" ".join(token)]

# CountVectorizer는 BOW 알고리즘의 단어의 단위를 띄어쓰기 한칸으로 구분함
vector = CountVectorizer()

print(vector.fit_transform(corpus).toarray()) # 코퍼스로부터 각 단어의 빈도 수를 기록한다
print(vector.vocabulary_) # 각 단어의 인덱스가 어떻게 부여되었는지를 보여준다

word2index={} # 인덱스 저장
bow=[] # BoW 저장

for word in token:

    # word2index에 값이 존재하지 않는 경우
    if word not in word2index.keys():

        # 값 넣기
        word2index[word]=len(word2index)

        # BoW 전체에 전부 기본값 1을 넣어줍니다. 단어의 개수는 최소 1개 이상이기 때문이다
        bow.insert(len(word2index)-1,1)

    else:
        # 재등장하는 단어의 인덱스 가져오기
        index=word2index.get(word)

        # 재등장한 단어는 해당하는 인덱스의 위치에 1을 더해줍니다. (단어의 개수를 세는 것입니다.)
        bow[index]=bow[index]+1

# 생성된 인덱스 출력
print(word2index)

# 단어별 출현 빈도가 연산된 결과 출력
print(bow)