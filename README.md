# 📊 프로젝트 데이터 준비 가이드

이 프로젝트는 **DACON**에서 제공하는 데이터셋을 사용합니다. 코드를 실행하기 전, 아래 단계에 따라 데이터 환경을 먼저 구축해 주세요.

## 1. 데이터 다운로드
먼저 아래 링크에 접속하여 대회 공식 데이터셋(`open.zip`)을 다운로드합니다.
* **데이터 다운로드 주소:** [DACON 대회 페이지 바로가기](https://dacon.io/competitions/official/236716/leaderboard?tab=submit)

## 2. 데이터 압축 해제 및 경로 설정
다운로드한 파일의 압축을 해제한 후, 기본 폴더 이름인 `open`을 `data`로 변경해야 스크립트가 정상적으로 작동합니다.

## 3. 최종 Directory 
.
├── .gitignore           # data/ 폴더 제외 설정 포함
├── data/                # old open folder
│   ├── train.csv
│   ├── test.csv
│   └── ...
├── result/
│   ├── ...              # submission.csv
├── main.py              # 실행 스크립트
└── README.md            
