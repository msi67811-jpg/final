# AR 평가용 공개 데이터 (research/data/ar_eval_public)

> 연구·팀 내부 평가 전용. 이 폴더는 `.gitignore`의 `research/data/`에 포함되어 커밋되지 않는다.
> 얼굴 이미지를 채팅·드라이브·PR·발표 자료에 올리지 않는다 (AR_SPEC 1-4, 1-13 원칙과 동일).

## 목적
직접 촬영 전, 공개 데이터로 AR 평가 절차(위치·경계, 질감, 자연스러움)를 먼저 돌려 본다.
**색상 일치 점수는 실제 제품 hex가 있는 우리 촬영 사진으로만 낸다.** 공개 데이터에는 제품명·호수가 없어서,
REAL 입술 색을 재서 AR에 넣으면 같은 색을 맞추는 순환 계산이 된다 (파운데이션 비교 때와 같은 문제).

## 폴더
```
ar_eval_public/
  README.md            이 문서
  sources.csv          후보 데이터셋 링크·라이선스·수량·쌍 여부·접근 상태
  select_samples.py    20개 선별 (정면·입 다묾·입술 크기·피부 밝기 분포·입술 색 계열)
  raw/
    ffhq_makeup/<id>/bare.jpg, makeup_01..05.jpg   ← 허깅페이스에서 받은 일부만
    commons/ (선택)                                 ← 라이선스 확인한 사진만, attribution 기록
  selected_pairs/      e01_bare.jpg, e01_real.jpg ... + manifest.csv
  selected_bare/       LaPa 맨얼굴 e01_bare.jpg ... + manifest.csv
  hw01_raw/            render_ar_offline.py 이름 규칙으로 복사한 것 + meta.csv
  ar/                  오프라인 렌더 결과 (앱 셰이더, 폰 카메라 경로 아님)
```

## 순서
1. **맨얼굴 20장 (LaPa, 이미 보유)**
   `python research/data/ar_eval_public/select_samples.py --source flat-bare --root research/data/reference/lapa/LaPa/test/images --out research/data/ar_eval_public/selected_bare --n 20 --dataset LaPa --license "비상업 연구용"`
   → manifest의 `reviewed` 열에 사람이 화장 여부·미성년자 여부·가림을 확인하고 `ok`/`drop` 기록.
   (맨입술도 채도가 높아 색만으로 '화장 안 함'을 자동 판정할 수 없다.)
2. **BEFORE/AFTER 20쌍 (FFHQ-Makeup)**
   브라우저로 https://huggingface.co/datasets/cyberagent/FFHQ-Makeup 에서 일부 폴더만 받아 `raw/ffhq_makeup/`에 둔다
   (전체 3.8GB 불필요. 80~100명 분량이면 20쌍 선별에 충분).
   `python research/data/ar_eval_public/select_samples.py --source ffhq-makeup --root research/data/ar_eval_public/raw/ffhq_makeup --out research/data/ar_eval_public/selected_pairs --n 20`
   AFTER는 확산 모델 합성이므로 **'실제 화장(REAL)'로 부르지 않고 'AFTER(합성)'**으로 기록한다.
3. **hw01 이름으로 복사 → 오프라인 렌더**
   `p01_day_00_<색계열>_0.jpg`(맨얼굴), `p01_day_00_<색계열>_3.jpg`(AFTER) + `meta.csv`의 `hex`는
   **앱 카탈로그 셰이드 hex**를 넣는다 (AFTER 입술 색을 재서 넣지 않는다).
   `python research/lip_reference/render_ar_offline.py --raw research/data/ar_eval_public/hw01_raw --out research/data/ar_eval_public/ar`
4. **평가**: 위치·경계(LaPa 마스크), 질감(반짝임 종류 통과/실패), 자연스러움(진짜 화장 사진과 섞은 블라인드 테스트).

## 라이선스 지킬 것
- FFHQ / FFHQ-Makeup: CC BY-NC-SA 4.0. 비상업 연구만, 출처 표시, 변경 사실 표시(AR 렌더 = 변경), 2차 결과물 공유 시 동일 조건.
  개별 FFHQ 사진의 작가·라이선스는 FFHQ 메타데이터(`ffhq-dataset-v1.json`)로 manifest의 `attribution`에 적는다.
- MT, LADN, YMU, VMU, MIW: 데이터 라이선스가 명시되지 않았거나 요청제 → 저자 확인 전에는 쓰지 않는다.
- YouTube 영상은 내려받거나 캡처하지 않는다.
