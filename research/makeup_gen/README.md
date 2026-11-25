# 메이크업 생성 모델 탐색 (`research/makeup_gen`)

11월 21일 멘토 점검용. AR 런타임에 넣지 않는다. 생성형 모델은 `docs/AR_SPEC.md` 10장대로 앱·숙제 02 표에 넣지 않는다.

| 파일 | 내용 |
|---|---|
| `SURVEY.md` | PDF ① 원문 |
| `VISHUE_2026-11-20_makeup-gen-survey.pdf` | PDF ① |
| `COMPARE_PROTOCOL.md` | PDF ② 프로토콜 (측정 없음) |
| `VISHUE_2026-11-20_candidate-compare-protocol.pdf` | PDF ② |
| `env_check.md` | GPU·가중치·전후 사진 |
| `build_pdf.py` | 한글 Malgun Gothic으로 PDF 재생성 |
| `IO_STRUCTURE.md` | (11-23) 모델별 입력·출력, 고정 프롬프트, 부위별 비교 가능 여부 |
| `COMPARE_LOCAL.md` | (11-23) AR·MagicMakeup·Stable-Makeup·FPMT·후처리 전체 비교표와 해석 |
| `eval_makeup.py`, `eval_sweep.py` | 입술 색·영역·보존 지표(11-20). 11-23부터 함수 import 가능(`main()`) |
| `region_composite.py` | (11-23) 생성 결과의 입술만 맨얼굴에 합성: 잘라 붙이기 / 색 차이만 |
| `texture_metric.py` | (11-23) 입술 주름 대역 상관·세기(광택 점 제외) |
| `face_parse.py`, `run_fpmt.py` | (11-23) 19클래스 얼굴 분할, FPMT CPU 실행(저장소 무수정) |
| `build_compare_local.py` | (11-23) 저장된 결과로 `COMPARE_LOCAL.md` 표와 비교 그림 생성 |
| `MULTI_SAMPLE.md` | (11-23) 공개 10쌍 + 사용자 1쌍 샘플별 지표·평균, 후처리 전·후, 모델별 가속화 결과 |
| `build_multi_figs.py` | (11-23) `multi_metrics.json`으로 `MULTI_SAMPLE.md` 그림 2장 생성 |
| `prep_samples.py`, `eval_multi.py` | (11-23) 여러 샘플 세트 준비(`samples.json`, 크롭, 분할), 샘플별 평가와 평균표 |
| `fpmt_accel.py` | (11-23) FPMT CPU 가속 실험(스레드, 입력 256, ONNX) |
| `runpod/run_sm_multi.py`, `runpod/run_mm_multi.py` | (11-23) RunPod GPU에서 Stable-Makeup·MagicMakeup 기준·가속 설정 실행 |

11-23 실행 환경: `research/data/makeup_gen/.venv`(uv, Python 3.12, mediapipe 1.1.1, scikit-image, torch 2.15 CPU, git 제외).
FPMT·얼굴 분할 저장소와 가중치는 `research/data/fpmt/`(git 제외). 사용자 얼굴 사진은 사용자 승인 하에 PC 안에서만 쓴다.

```bash
python research/makeup_gen/build_pdf.py
```

추론·가중치 다운로드는 이 폴더에서 하지 않는다. CUDA GPU가 생기면 공개 샘플로 Stable-Makeup부터, 사용자 얼굴은 쓰지 않는다.
