# One-off builder for the opacity-layer comparison PDF. Not part of the app.
import os
from PIL import Image, ImageDraw, ImageFont

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle,
    PageBreak, KeepTogether, ListFlowable, ListItem,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "AR_OPACITY_LAYER_COMPARE.pdf")
TMP = os.path.join(ROOT, "_opacity_pdf_tmp")
os.makedirs(TMP, exist_ok=True)

PHOTOS = [
    (r"C:\Users\playdata2\.cursor\projects\c-Users-playdata2-SKN34-Final-Project\assets\______-ce849e7d-2edd-472f-a9cd-fd2d9eabcbfb.jpg", "좌측 (9:34)"),
    (r"C:\Users\playdata2\.cursor\projects\c-Users-playdata2-SKN34-Final-Project\assets\______-9af47ad3-2805-4a6c-8e68-61dde80b1dae.jpg", "정면 (9:34)"),
    (r"C:\Users\playdata2\.cursor\projects\c-Users-playdata2-SKN34-Final-Project\assets\______-37d99d75-6079-4a32-86f7-bc626e046116.jpg", "우측 (9:36)"),
]

pdfmetrics.registerFont(TTFont("Malgun", r"C:\Windows\Fonts\malgun.ttf"))
pdfmetrics.registerFont(TTFont("MalgunBd", r"C:\Windows\Fonts\malgunbd.ttf"))

SKIN = (214, 176, 156)
PIGMENT = (176, 58, 72)  # a coral used only as a swatch, not a product hex change


def src_over(base, color, a):
    return tuple(int(round(a * c + (1 - a) * b)) for b, c in zip(base, color))


def coats(base, color, alphas):
    px = base
    for a in alphas:
        px = src_over(px, color, a)
    return px


def patch(w, h, rgb):
    im = Image.new("RGB", (w, h), rgb)
    return im


def draw_swatches():
    """Uniform 60% vs 3x20% vs 3 coats that sum to the same 60% vs spatially varied coats."""
    W, H = 220, 280
    skin_tex = Image.new("RGB", (W, H), SKIN)
    dr = ImageDraw.Draw(skin_tex)
    # Fake lip lines so a flat coat vs a varied coat can be seen.
    for y in range(40, H - 30, 7):
        shade = 18 if (y // 7) % 2 == 0 else -10
        col = tuple(max(0, min(255, c + shade)) for c in SKIN)
        dr.line([(18, y), (W - 18, y + 2)], fill=col, width=2)
    dr.ellipse([70, 90, 150, 150], fill=tuple(max(0, c - 28) for c in SKIN))

    def apply_uniform(alpha):
        out = skin_tex.copy()
        pix = out.load()
        for y in range(24, H - 24):
            for x in range(16, W - 16):
                pix[x, y] = src_over(pix[x, y], PIGMENT, alpha)
        return out

    def apply_passes(alphas, vary=False):
        out = skin_tex.copy()
        pix = out.load()
        cx, cy = W / 2, H / 2
        for a in alphas:
            for y in range(24, H - 24):
                for x in range(16, W - 16):
                    aa = a
                    if vary:
                        # Center holds more pigment; the rim stays sheer. Lines stay.
                        nx = (x - cx) / (W * 0.42)
                        ny = (y - cy) / (H * 0.42)
                        fall = max(0.15, 1 - (nx * nx + ny * ny))
                        aa = a * fall
                    pix[x, y] = src_over(pix[x, y], PIGMENT, aa)
        return out

    # a such that 1-(1-a)^3 = 0.60
    a_match = 1 - (0.4 ** (1 / 3))
    imgs = [
        ("1장 60%", apply_uniform(0.60)),
        ("3장 x 20%", apply_passes([0.20, 0.20, 0.20])),
        ("3장, 합이 60%", apply_passes([a_match, a_match, a_match])),
        ("3장, 가운데만 진함", apply_passes([0.34, 0.34, 0.34], vary=True)),
    ]

    def mean_diff(a, b):
        pa, pb = a.load(), b.load()
        s = 0
        n = 0
        for y in range(24, H - 24):
            for x in range(16, W - 16):
                s += sum(abs(pa[x, y][i] - pb[x, y][i]) for i in range(3)) / 3
                n += 1
        return s / n

    d20 = mean_diff(imgs[0][1], imgs[1][1])
    dmatch = mean_diff(imgs[0][1], imgs[2][1])
    dvary = mean_diff(imgs[0][1], imgs[3][1])

    font = ImageFont.truetype(r"C:\Windows\Fonts\malgun.ttf", 18)
    font_s = ImageFont.truetype(r"C:\Windows\Fonts\malgun.ttf", 14)
    gap = 16
    canvas = Image.new("RGB", (4 * W + 5 * gap, H + 78), (255, 255, 255))
    cd = ImageDraw.Draw(canvas)
    notes = [
        "한 번에 60%",
        "덮임 48.8%",
        "픽셀 거의 동일",
        "가장자리만 옅음",
    ]
    for i, ((title, im), note) in enumerate(zip(imgs, notes)):
        x = gap + i * (W + gap)
        canvas.paste(im, (x, 8))
        cd.text((x, H + 16), title, fill=(20, 20, 20), font=font)
        cd.text((x, H + 42), note, fill=(80, 80, 80), font=font_s)
    path = os.path.join(TMP, "swatches.png")
    canvas.save(path, quality=92)
    return path, d20, dmatch, dvary, a_match


def thumb(src, label):
    im = Image.open(src).convert("RGB")
    im.thumbnail((420, 740))
    # Caption bar
    bar = Image.new("RGB", (im.width, im.height + 36), (255, 255, 255))
    bar.paste(im, (0, 0))
    d = ImageDraw.Draw(bar)
    font = ImageFont.truetype(r"C:\Windows\Fonts\malgun.ttf", 18)
    d.text((8, im.height + 8), label, fill=(20, 20, 20), font=font)
    path = os.path.join(TMP, "ph_" + str(abs(hash(src))) + ".jpg")
    bar.save(path, quality=85)
    return path


SWATCH, D20, DMATCH, DVARY, A_MATCH = draw_swatches()
THUMBS = [thumb(p, lab) for p, lab in PHOTOS]

styles = getSampleStyleSheet()
S = ParagraphStyle("K", fontName="Malgun", fontSize=10, leading=15, textColor=black)
H1 = ParagraphStyle("KH1", fontName="MalgunBd", fontSize=16, leading=22, spaceAfter=8)
H2 = ParagraphStyle("KH2", fontName="MalgunBd", fontSize=12, leading=16, spaceBefore=10, spaceAfter=4)
SM = ParagraphStyle("KSm", fontName="Malgun", fontSize=8.5, leading=12, textColor=HexColor("#333333"))
TH = ParagraphStyle("KTh", fontName="MalgunBd", fontSize=8, leading=11, textColor=white)
TD = ParagraphStyle("KTd", fontName="Malgun", fontSize=8, leading=11)

def P(text, style=S):
    return Paragraph(text, style)

doc = SimpleDocTemplate(
    OUT, pagesize=A4,
    leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
    title="AR 부위별 투명도와 층 분할 비교",
    author="VISHUE",
)

story = []
story.append(P("AR 부위별 투명도 — 한 장 vs 여러 층", H1))
story.append(P("2026-11-11. 기준 코드: components/ar/faceTrackerHtml.ts 의 TUNING·MATERIALS. 폰 캡처 3장은 같은 날 9:34–9:36 전면 카메라(메이크업이 화면에 보이지 않음).", SM))
story.append(Spacer(1, 6))
story.append(P("결론", H2))
story.append(P(
    "60%를 20%짜리 3장으로 나누면 더 자연스러워지지 않습니다. 같은 색을 같은 자리에 세 번 얹으면 "
    "보이는 덮임은 60%가 아니라 약 48.8%가 되고, 합이 60%가 되도록 장마다 약 26%로 맞추면 "
    "한 장 60%와 픽셀이 같습니다. 그래서 렌더러에는 층을 나누는 코드를 넣지 않았습니다. "
    "자연스러움은 층 개수가 아니라, 이미 있는 가장자리 페이드·주름·가닥 끝 감쇠에서 나옵니다."
))

story.append(P("왜 20% × 3 이 60%가 아닌가", H2))
story.append(P(
    "반투명은 더하기가 아닙니다. 한 장을 얹고 남은 맨살 위에 다음 장이 올라갑니다. "
    "덮임 = 1 − (1 − 한 장 투명도)<super>장수</super>. "
    "20%를 세 번이면 1 − 0.8<super>3</super> = 0.488, 즉 48.8%입니다. "
    "입술은 이미 이 식으로 바르는 횟수를 한 번의 셰이더에서 계산합니다 "
    "(layersPerIntensity = 3, 진하기 100%이면 3회). 화면을 세 번 그리는 것과 결과가 같습니다."
))
story.append(P(
    f"합을 60%로 맞추는 한 장 투명도는 1 − 0.4<super>1/3</super> = {A_MATCH*100:.1f}%입니다. "
    f"아래 견본에서 1장 60%와 이 3장의 평균 채널 차이는 {DMATCH:.2f} (0에 가까움). "
    f"20%×3과의 차이는 {D20:.1f}. 가운데만 진한 3장과의 차이는 {DVARY:.1f}인데, "
    "이것은 투명도를 나눈 효과가 아니라 위치를 다르게 칠한 효과입니다. 블러셔 타원, 립 가장자리, "
    "속눈썹 끝 감쇠가 이미 그 역할을 합니다."
))

story.append(Spacer(1, 4))
story.append(RLImage(SWATCH, width=178 * mm, height=178 * mm * (280 + 78) / (4 * 220 + 5 * 16)))
story.append(P("견본은 제품 색이 아닙니다. 산호색을 피부색 줄무늬 위에 올려, 층 나눔이 결을 살리지는 못한다는 것만 보여 줍니다. 세 번째 칸은 첫 칸과 눈으로도 구분되지 않습니다.", SM))

story.append(P("부위별 현재 덮임 (진하기 막대 100%일 때)", H2))
story.append(P("슬라이더를 낮추면 표의 값에 진하기를 곱하거나, 입술·파운데이션은 아래 식으로 줄어듭니다. 캡처 3장에는 이 색이 켜져 있지 않습니다.", SM))

rows_raw = [
    ["부위", "한 장의 기준", "진하기 100%일 때 덮임", "이미 층으로 계산되나"],
    ["립 매트", "한 번 바름 a1 72%", "1−(1−0.72)^3 = 97.8%", "예. 횟수 3"],
    ["립 벨벳", "a1 66%", "96.1%", "예"],
    ["립 크림", "a1 58%", "92.6%", "예"],
    ["립 새틴", "a1 50%", "87.5%", "예"],
    ["립 글로시", "a1 46%", "84.3%", "예. 윤기는 별도"],
    ["립 쉬머", "a1 55%", "90.9%", "예"],
    ["립 시어", "a1 38%", "76.2% + 물들이기 0.8", "예"],
    ["립 오일", "색 덮임 10~20%×0.6", "진하기 100%·틴트 1이면 12%", "색과 윤기(60~100%)를 분리"],
    ["립 밤", "오일와 같은 색 덮임", "최대 12%", "윤기 거칠기만 다름"],
    ["블러셔", "blushCoverage 0.60", "60% × 진하기", "아니오. 가루 물들이기 40%는 섞임"],
    ["아이섀도 매트", "opacity 0.70", "70% × 진하기", "아니오"],
    ["아이섀도 쉬머·글리터", "opacity 0.30", "30% × 진하기", "입자 레이어는 따로"],
    ["아이섀도 글로시", "opacity 0.25", "25% × 진하기", "아니오"],
    ["아이섀도 보조색", "shadow2Coverage 0.80", "80% × 진하기", "아니오"],
    ["아이라인", "코어 opacity 1.0", "100% (진하기와 무관)", "가장자리 반 픽셀만 연함"],
    ["마스카라", "0.55+0.35×진하기", "90%. 가닥 끝은 60% 감쇠", "가닥이 이미 뿌리→끝 층"],
    ["속눈썹 라인", "lashLineAlpha 0.40", "40%", "아니오"],
    ["아래 속눈썹", "lowerLashAlpha 0.22", "22%", "아니오"],
    ["아이브로우 파우더", "powder 0.45", "앞 40%·산 85%·꼬리 90% 밀도", "2층: 파우더 + 브로우 마스카라"],
    ["파운데이션", "foundCoverage 0.30→0.85", "시어 30% ~ 풀 85%", "그중 불투명 몫 15%→90%"],
    ["하이라이터", "hlCoverage 0.50", "50% × 진하기 × 영역 세기", "아니오"],
    ["쉐딩", "턱 경로 0.85 (그 외 0.80)", "85% × 진하기", "가루 물들이기 32%"],
]

header = [P(c, TH) for c in rows_raw[0]]
data = [header]
for r in rows_raw[1:]:
    data.append([P(c, TD) for c in r])

table = Table(data, colWidths=[32 * mm, 42 * mm, 58 * mm, 46 * mm], repeatRows=1)
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#222222")),
    ("TEXTCOLOR", (0, 0), (-1, 0), white),
    ("BACKGROUND", (0, 1), (-1, 1), HexColor("#f4f4f4")),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("GRID", (0, 0), (-1, -1), 0.3, HexColor("#cccccc")),
    ("LEFTPADDING", (0, 0), (-1, -1), 3),
    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(Spacer(1, 4))
story.append(table)

story.append(P("진하기를 낮출 때", H2))
story.append(P(
    "립: 횟수 = max(0.3, 진하기 × 3). 진하기 50%·새틴이면 횟수 1.5, 덮임 1−0.5<super>1.5</super> = 64.6%. "
    "블러셔: 60% × 진하기 (50%이면 30%). "
    "파운데이션: 덮임 0.30+0.55×진하기, 그중 불투명 몫 0.15+0.75×진하기. 진하기 50%이면 덮임 57.5%, 불투명 몫 52.5%. "
    "마스카라: 0.55+0.35×진하기. 진하기 50%이면 72.5%, 가닥 끝은 그 값에서 다시 60%를 뺌."
))

story.append(P("적용하지 않은 이유", H2))
story.append(P(
    "1. 합을 같게 나눈 3장은 1장과 픽셀 차이가 없습니다. 폰으로 다시 찍어도 유의미한 전후가 나오지 않습니다."
))
story.append(P(
    "2. 20%×3으로 나누면 사용자가 고른 진하기가 60%에서 48.8%로 바뀝니다. 더 연해질 뿐 결이 살아나지 않습니다."
))
story.append(P(
    "3. 셰이더를 세 번 돌리면 프레임당 일이 늘어 지금 23fps 근처인 루프가 더 느려집니다. 화면은 같습니다."
))
story.append(P(
    "4. 스펙(AR_SPEC 6-5-3)은 입술 색 값을 바꾸지 말고 명암·결·반사로 자연스럽게 하라고 합니다. "
    "층 개수를 늘리는 것은 그 결을 만들지 못합니다. 립은 이미 카메라 주름 위에 횟수 식을 올리고, "
    "블러셔·쉐딩은 가장자리가 페이드되며, 마스카라는 뿌리에서 끝으로 옅어집니다."
))

story.append(PageBreak())
story.append(P("1번. 지금 화면 (제품이 보이지 않는 캡처)", H1))
story.append(P(
    "아래 세 장은 이번 요청에 첨부된 전면 카메라입니다. 좌·정면·우 모두 맨얼굴이고, "
    "립·블러셔·섀도·라인·파운데이션 색이 올라간 상태가 아닙니다. "
    "그래서 '현재 구현'의 색 사진으로는 쓸 수 없고, 비교의 기준 얼굴로만 둡니다. "
    "2번(층을 나눈 구현)은 화면이 같으므로 다시 찍지 않았습니다."
))
story.append(Spacer(1, 6))

ims = []
for path in THUMBS:
    ims.append(RLImage(path, width=55 * mm, height=55 * mm * 1.85))
photo_table = Table([ims], colWidths=[58 * mm, 58 * mm, 58 * mm])
photo_table.setStyle(TableStyle([
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("LEFTPADDING", (0, 0), (-1, -1), 2),
    ("RIGHTPADDING", (0, 0), (-1, -1), 2),
]))
story.append(photo_table)
story.append(Spacer(1, 8))
story.append(P("전후 비교에서 확인된 차이", H2))
story.append(P(
    f"균일한 층을 나눈 경우, 덮임만 같게 맞추면 평균 채널 차이 {DMATCH:.2f}/255 입니다. "
    "사람이 두 사진을 나란히 봐도 구분되지 않는 크기입니다. "
    f"20% 세 장으로 나누면 평균 {D20:.1f}/255 만큼 더 옅어지지만, 모양·주름·가장자리는 그대로입니다. "
    "유의미한 차이라고 볼 항목(입술 결, 가닥, 볼 가장자리)은 이 실험에서 바뀌지 않았습니다."
))
story.append(P(
    "다음에 실기기에서 부위별 현재 색을 남기려면, 부위마다 진하기 100%로 켠 뒤 "
    "정면 한 장씩 HUD(fps, 빌드 문자열)가 보이게 찍어 두면 이 표의 퍼센트와 맞춰 볼 수 있습니다. "
    "그 사진이 있어야 층 나눔이 아닌 다른 수정(가장자리, 결)의 전후를 비교할 수 있습니다."
))

doc.build(story)
print(OUT)
print("diff20", round(D20, 2), "diffmatch", round(DMATCH, 3), "diffvary", round(DVARY, 2))
