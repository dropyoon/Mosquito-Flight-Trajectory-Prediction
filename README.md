# 📊 프로젝트 가이드

이 프로젝트는 **DACON** 모기 비행 경로 예측 대회 데이터셋을 사용합니다. 코드를 실행하기 전, 아래 단계에 따라 환경 구축 및 데이터 준비를 진행해 주세요.

## 1. 환경 설정 (Environment Setup)
이 프로젝트는 Conda 환경을 사용합니다. 제공된 `environment.yml` 파일을 사용하여 필요한 패키지를 한 번에 설치할 수 있습니다.

```bash
# 1. 가상환경 생성
conda env create -f environment.yml

# 2. 가상환경 활성화
conda activate DACON_2605
```

## 2. 데이터 다운로드
먼저 아래 링크에 접속하여 대회 공식 데이터셋(`open.zip`)을 다운로드합니다.
* **데이터 다운로드 주소:** [DACON 대회 페이지 바로가기](https://dacon.io/competitions/official/236716/data)

## 3. 데이터 압축 해제 및 경로 설정
다운로드한 파일의 압축을 해제한 후, 기본 폴더 이름인 `open`을 `data`로 변경해야 스크립트가 정상적으로 작동합니다.

## 4. 최종 Directory 구조
```
.  
├── .gitignore          # data/ 폴더 제외 설정 포함   
├── environment.yml     # 환경 설정 파일 (Conda)
├── data/               # 데이터 폴더 (기존 open 폴더)
│   ├── train.csv  
│   ├── test.csv  
│   └── ...  
├── result/             # 결과 저장 폴더  
│   └── submission.csv  # 최종 제출 파일  
├── main.py             # 실행 스크립트  
└── README.md
```
