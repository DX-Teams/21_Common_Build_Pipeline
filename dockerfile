# 기반이 되는 이미지 레이어입니다.
# <이미지 이름>:<태그> 형식으로 작성
FROM python:3.9.1-slim

# fashion(작업) 디렉토리 생성
RUN mkdir -p /fashion

# Docker 이미지 내부에서 RUN, CMD, ENTRYPOINT의 명령이 실행될 디렉터리 설정
WORKDIR /fashion

# 현재 디렉터리에 있는 파일들을 이미지 내부 /fashion 디렉터리에 추가함(ADD or Copy)
ADD . /fashion

# Docker 이미지 생성 전, 수행 쉘 명령어
RUN pip3 install -r requirements.txt

# 호스트와 연결할 포트 번호
EXPOSE 80

# 컨테이너 시작 시, 실행할 실행 파일 or 쉘 스크립트
CMD python Fashion.py