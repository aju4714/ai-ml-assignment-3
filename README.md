# Assignment 3 · 점핑잭 3D 모션 데이터 후처리

수업의 BlazePose 추출 설정과 y축 부호 규칙을 적용해 공개 점핑잭 영상의 0.5초 이상, 3.0초 미만 구간을 분석했다. 교수님 자료의 `create_height_corrected_target`으로 낮은 쪽 발끝을 프레임별 높이 0으로 맞췄다. 수업 스크립트에서 선택 사항인 Viser 실행 부분을 제외했으며 보정, 입력 형식, Unity CSV 출력 로직은 그대로 사용했다.

## 재실행

Python 3.11 환경에서 다음을 실행한다. 최초 실행에는 영상과 BlazePose 모델을 내려받을 인터넷 연결이 필요하다.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python analyze_motion.py
```

기존 영상이 있으면 `python analyze_motion.py --video /path/to/video.webm`으로 실행한다. 결과 NPZ에는 원본 프레임 번호, 시간, 원본 및 보정 좌표(단위 m), 가시도가 들어 있다. Unity CSV의 프레임 번호는 수업 코드의 규칙에 따라 0부터 다시 시작한다. 그래프의 프레임 시간은 원본 영상 기준이다.

## 영상 출처

Taco Fleur, [Jumping jacks and burpees](https://commons.wikimedia.org/wiki/File:Jumping_jacks_and_burpees.webm), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). 원본 영상의 일부 구간으로 키포인트를 추출했다.
