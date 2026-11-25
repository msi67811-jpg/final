# VISHUE AR 개발 발전 현황 — 근거 정리

이 문서는 `VISHUE_AR_개발발전현황_기술보고서.pdf`(11쪽)에 쓴 주장마다 근거 자료를 정리한 것이다.

- **작성일:** 2026-11-24
- **소스 기준:** 브랜치 `feat/ar-lighting-lip-finish`, 커밋 `4c4262e7`(2026-11-23 12:22). 앱 코드는 작업 트리 변경 없음.
- **작업 범위:** 보고서를 만들면서 기존 코드·실험·원본 파일은 수정하지 않았다. 수치는 모두 기존 자료에 적힌 값이며, 이번에 새로 측정하지 않았다.

> 얼굴 사진이 들어 있다. PDF와 `figures/`는 커밋하지 않는다(`reports/ar_progress/`는 `.gitignore` 대상이 아님).

## 1. 코드에서 확인한 사실 (3·6·7쪽)

| 주장 | 근거 |
|---|---|
| AR 페이지는 HTML 문자열 한 파일이다(7,197줄) | `components/ar/faceTrackerHtml.ts` (`FACE_TRACKER_HTML`) |
| MediaPipe Tasks Vision 1.1.1, Face Landmarker float16, GPU 생성이 실패하면 CPU로 다시 만든다 | 같은 파일 `MEDIAPIPE_VERSION`, `MODEL_URL`, `start()`, `createLandmarker` |
| Image Segmenter selfie_multiclass_256을 쓴다 | `SEG_MODEL_URL`, `createSegmenter`, `runSegmenter` |
| 프레임마다 업로드 → 검출 → 필터 → 메시 → 렌더 순으로 처리한다 | `processFrame` (6856행~): `renderer.upload()` → `detectForVideo` → `filterLandmarks` → `updateMesh` → `renderer.render` |
| 루프는 `requestVideoFrameCallback`을 쓰고, 없으면 rAF를 쓴다 | `startLoop`, `onVideoFrame` |
| 조명은 0.5초마다, 얼굴이 멈춰 있으면 1.5초마다 계산한다. 분할은 2초(자동 1초) 간격이다 | `TUNING.lightEveryMs 500`, `lightStillMs 1500`, `segmentEveryMs 2000`, `segmentEveryMsAuto 1000` |
| 립 제형 재질표 값 | `MATERIALS` (662행~): matte·velvet·cream·satin·sheer·glossy·shimmer·oil·balm |
| 섀도 마감은 6종이다 | `SHADOW_FINISHES` |
| AR 패널 부위는 8개이고, 마스카라는 룩 전용이다 | `store/makeupStore.ts` `AR_PAINT_ZONES`(주석 "Mascara is look/video only") |
| UI 제형 칩은 5개다(벨벳→매트, 시어→글로시) | `lib/lipCare.ts` `LIP_TEXTURE_CHIPS`, `lipTextureChip` |
| 앱과 페이지는 메시지로 통신한다 | `components/ar/ArFaceView.tsx` `injectJavaScript(window.__kb.receive)`, 페이지 `post({type:...})`: status·env·light_hint·skin_tone·pose_hint·face_fit·crash |
| 카탈로그 립은 617개다(glossy 283 · velvet 103 · satin 101 · cream 41 · sheer 40 · matte 39 · shimmer 10) | `lib/devCatalog/oyCatalog.json` 집계 |
| 조명 안내 코드는 커밋 `38b1d52d`(11-23)에 들어갔다 | `lib/arEnv.ts`, `hooks/useArEnvMessage.ts`, `docs/AR_ENV_THRESHOLDS.md` |

## 2. Git 이력

`faceTrackerHtml.ts`를 바꾼 커밋은 30개다(10-25 ~ 11-23). 그중 주요 커밋:

- 10-25: 얼굴 메시 렌더링, 립·눈·얼굴 부위 메이크업(첫 AR)
- 10-29: 조명 변화 안내, 아이라인 6종
- 11-02: 입술 광택·경계 개선
- 11-04: 조명 추정 2~5단계, 후보 조명 맞추기, 입술 질감·치아 가드
- 11-06: 쉐딩 오류 수정, 턱선·콧대 자동 쉐딩
- 11-15: 립 경계 안정화, 매트 반사, 제형 cavity
- 11-19: 영상 룩 조명·톤
- 11-23: 촬영 환경 안내, 립 제형(매트·새틴·글로시)

## 3. 문서 자료 (`G:\내 드라이브\ppt\`)

| 자료 | 보고서에서 쓴 곳 |
|---|---|
| `VISHUE_AR_변천사_11-05갱신.pdf` (12쪽) | 2쪽 타임라인, 4쪽 경계 표, 7쪽 FPS·설문·F1 |
| `VISHUE_AR_조명대응_구조_변경_근거.pdf` (15쪽, 11-20) | 3쪽 조명 단계, 7쪽 조명 그래프·판정 수치, 8~10쪽에 원본 7·8·10쪽을 그대로 넣고 실사화 관점 설명 추가 |
| `VISHUE_AR_촬영환경_통제_보정_안내.pdf` (7쪽, 11-19) | 조명 단계 설명 보조(수치는 아래 4장의 충돌 참고) |
| `AR_얼굴추적_기술비교_정리.pdf`, `AR_SDK_성능비교_정리.pdf` (11-01) | 3쪽 선택 근거, 7쪽 감지 시간 |
| `lip_ABC/`, `lip_edge_matte/` (11-13) | 4~6쪽 같은 조건 전후 비교 |
| `발전도/` 폴더(사진 160장 중 얼굴 검출 126장) | 1·2·4·5·7쪽 실기기 사진 |
| `Claude outputs/VISHUE_립제형_고도화전후_비교.pdf` | 6쪽 그래프(11-12 vs 11-16 제품색 ΔE) |
| `docs/AR_PROGRESS.md` | 날짜별 변경·측정값 전반 |
| `research/makeup_gen/COMPARE_LOCAL.md`, `SURVEY.md` | 8쪽 생성 모델 비교 |

## 4. 서로 충돌하거나 다른 자료

이 보고서가 어떤 값을 택했는지 함께 적는다.

1. **저조도 오탐 수치:** `촬영환경_안내.pdf`(11-19)의 "1.5/0/0%"는 `조명대응_근거.pdf`(11-20)에서 표본 내 평가라는 이유로 철회됐다. 보고서는 재실험 값(평가 분할 0/0/0%, LaPa 2.5%)을 썼다.
2. **조건 수 "13가지" vs "8개":** 13가지는 위치·거리·움직임을 포함한 수이고, 8개는 조명 조건만 센 수다. 같은 내용을 범위만 다르게 센 것이다.
3. **진행표 마일스톤 표:** `docs/AR_PROGRESS.md` 상단 표는 M2~M21이 모두 "대기"로 남아 실제 구현과 맞지 않는다(표 갱신 누락). 보고서는 코드와 일자별 기록을 따랐다.
4. **AR에서 뺀 부위:** 변천사(10-27)는 "마스카라·하이라이터·쉐딩을 AR에서 제외"라고 적었다. 하이라이터·쉐딩은 11-03에 다시 켜졌고, 지금 룩 전용은 마스카라뿐이다.
5. **점 개수 478 vs 468:** Face Landmarker가 주는 점은 478개(홍채 10개 포함)이고, 렌더 메시는 468점 캐노니컬 UV를 쓴다. 둘 다 맞다.
6. **조명 작업의 커밋 여부:** 조명대응 PDF(11-20)는 "11-19 이후 작업은 아직 커밋 안 함"이라고 적었다. 그 뒤 11-23 커밋 `38b1d52d`에 들어갔다.

## 5. 추정한 것

- **11-23 매트·새틴·글로시 폰 캡처:** D안 적용 전 화면으로 추정했다. 근거는 AR_PROGRESS 11-23 (25)에 이 캡처가 D안 계획의 근거로 적혀 있다는 점이다. 저장 시각(10:07)과 커밋 시각(12:22)의 순서도 맞는다.
- **11-20 조명 캡처의 장소:** 파일명(창가·실내·스탠드·불 끈 방)으로 정했다. 진행표에도 "장소 대응은 추정"이라고 적혀 있다.
- **발전도 사진의 날짜:** 카카오톡 파일명의 날짜를 썼다. 파일명에 날짜가 없으면(입술1~7 등) 파일 저장 시각과 변천사 PDF의 날짜를 썼다.

## 6. 측정 자료가 없는 것

- 모델 로딩 시간, 메모리 사용량(iOS WebView는 JS 힙 값을 주지 않음)
- 11-23 D안 이후 실기기 화면, 아이폰 실기기 WB 잠금
- 눈·볼·쉐딩 부위의 정답 데이터 기반 정확도
- 11-02 이후 렌더로 한 자연스러움 재설문

## 7. 11쪽 B AR default 교체 (11-24)

- **바꾼 것:** 방법별 결과 그림의 B AR default 입술을 카탈로그 #9E6364(glossy 0.65) 렌더로 바꿨다. 표와 그래프의 값도 맞췄다.
- **측정:** 기존 `research/makeup_gen/eval_makeup.py`의 `compare`와 `texture_metric.py`의 `measure`를 그대로 불러 `ar_out_catalog/ar_lip_0~2.png`(3프레임)에 돌렸다. 기존 B와 같은 기준(AR off 프레임, 랜드마크 볼록 껍질 얼굴 영역, 착용 사진 E의 입술 중앙값)이다.
- **결과(3프레임 평균):** 입술 ΔE00 vs 착용 사진 **2.65**(기존 B #C77276은 8.01), 입술 밖 ΔE00 0.00, 입술 밖 SSIM 1.000, 입술 모양 IoU 0.946, 주름 상관 0.870, 주름 세기 1.163, 광택 점 7%. 렌더된 입술 중앙값 색 #A85E5B.
- **주의:** #9E6364는 착용 사진 입술색 #A26263과 가장 가까운 카탈로그 glossy 셰이드를 골라 쓴 값이다(`ar_render_catalog.mjs` 주석). 착용 사진의 제품(롬앤 03 베어 그레이프)과 같은 제품이 아니므로 2.65는 "입력 색이 가까우면 AR 오차가 줄어든다"는 뜻이다. 제품 정확도가 아니다.
- **그대로 둔 것:** 7쪽 남은 문제 목록의 "공식몰 hex 색 오차(ΔE00 8.01)"는 #C77276에 대한 측정값이라 그대로 두었다. `COMPARE_LOCAL.md`와 `eval/metrics.json`도 수정하지 않았다.

## 8. 아이섀도 제형별 펄감 (7쪽, 11-24 새로 렌더)

- **렌더:** 기존 `research/makeup_gen/ar_zone_render.mjs`를 수정 없이 한 번 실행했다. 현재 `faceTrackerHtml.ts`를 빌드해 같은 정지 프레임에 아이섀도 하나만 켜고, `geometry.shadowFinish`만 matte·satin·shimmer·glitter·glossy로 바꿨다. 메탈릭은 glitter에 `particle.metal: true`를 주어 앱과 같은 경로(`base = SHADOW_FINISHES.metallic`)로 그렸다. 색 #A6706D(멘토링 자료와 같은 카탈로그 섀도), 진하기 0.65. 오류 0.
- **측정:** 6제형 중 하나라도 맨얼굴과 ΔE00 > 2가 된 화소를 변화 영역(20,209화소)으로 잡았다.
  - 반짝 알갱이 = 주변(가우시안 σ4)보다 L*가 10 넘게 밝은 화소 덩어리 수
  - 측정값: 매트 0 · 새틴 0 · 쉬머 1.3 · 글리터 8.5 · 메탈릭 8.3 · 글로시 0.5개/1,000화소
  - 얼굴 1장·색 1개·정지 프레임 1회라 경향 참고용이다.
- **확인 필요:** 참고 이미지는 글리터가 메탈릭보다 강하지만, 이 렌더에서는 알갱이 수가 비슷하고 메탈릭이 더 밝았다. 11-19 v8·v9는 폰에서 아직 확인하지 않았다(AR_PROGRESS).

## 9. 원본 페이지를 그대로 넣은 쪽

- **8회차 멘토링 대응 PDF 7·8쪽 → 9·10쪽:** 표의 값은 원본 8쪽 그대로다.
- **조명대응 근거 PDF 2·4·5·7·8·10쪽 → 11~16쪽:** 원본 4쪽의 "11-19 이후 작업 미커밋"은 이후 11-23 커밋 `38b1d52d`에 들어갔다고 오른쪽 설명에 적었다.
