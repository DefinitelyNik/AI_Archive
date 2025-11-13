from flask import Flask, render_template, request, redirect, url_for
import easyocr
from navec import Navec
from slovnet import NER
from PIL import Image
import os
import re

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ocr_reader = easyocr.Reader(['ru'])

navec_path = 'navec_news_v1_1B_250K_300d_100q.tar'
navec = Navec.load(navec_path)

ner_model = NER.load('slovnet_ner_news_v1.tar')
ner_model.navec(navec)

def perform_ocr(image_path):
    result = ocr_reader.readtext(image_path)
    text = " ".join([item[1] for item in result])
    return text

def perform_ner(text):
    markup = ner_model(text)
    return markup

def find_dates(text):
    date_pattern = r'\b(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})\b'
    iso_pattern = r'\b(\d{4}-\d{2}-\d{2})\b'
    year_pattern = r'\b(\d{4})\s*(?:г\.?|год|года)\b'
    year_decade_pattern = r'\b(?:в\s+)?(\d{3}0)-[хxs]\s*(?:годах|годов|году|год|гг\.?)?\b'

    dates = []
    for match in re.finditer(date_pattern, text):
        dates.append((match.start(), match.end(), 'date'))
    for match in re.finditer(iso_pattern, text):
        dates.append((match.start(), match.end(), 'date'))
    for match in re.finditer(year_pattern, text):
        dates.append((match.start(), match.end(), 'date'))
    for match in re.finditer(year_decade_pattern, text):
        dates.append((match.start(), match.end(), 'date'))

    dates.sort(key=lambda x: x[0])
    return dates

def annotate_text(markup):
    ner_spans = [(span.start, span.stop, span.type.lower()) for span in markup.spans]

    dates = find_dates(markup.text)

    all_spans = ner_spans + dates
    all_spans.sort(key=lambda x: x[0])

    tokens = []
    last = 0
    for start, stop, label in all_spans:
        if last < start:
            tokens.append(markup.text[last:start])
        entity_text = markup.text[start:stop]
        tokens.append(f'<mark class="ner-{label}">{entity_text}</mark>')
        last = stop
    if last < len(markup.text):
        tokens.append(markup.text[last:])
    return ''.join(tokens)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']

        if file.filename == '':
            return redirect(request.url)

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            text = perform_ocr(filepath)

            # text_placeholder = "Объединивший СССР и страны народной демократии Совет экономической взаимопомощи стал одним из крупнейших в мире проектов международной экономической интеграции. Сама задача развития кооперации между государствами Восточной Европы представлялась очень сложной. Изначально они были слабо связаны между собой в экономическом плане: в 1938 г. для Болгарии, Венгрии, Польши, Румынии и Чехословакии торговля друг с другом составляла всего 11,5% от их общего товарооборота, в то время как доля одной только Германии – 23,3%. Однако уже в 1959 г. торговля внутри СЭВ составляла для каждой из стран в среднем 75% от внешнего товарооборота. Во многом благодаря интеграции были достигнуты серьезные успехи в области промышленного развития стран народной демократии. По оценкам западных аналитиков, к концу 1957 г. Москва предоставила другим странам СЭВ кредитов на сумму 28 млрд руб. (примерно 7 млрд долл.). Одна только Болгария получила кредитов на 8 млрд руб. (2 млрд долл.), т. е. больше, чем Голландия (1 млрд долл.), Италия (1,3 млрд долл.) или ФРГ (1,3 млрд долл.) в рамках реализации плана Маршалла в 1948–1951 гг"
            # markup = perform_ner(text_placeholder)

            markup = perform_ner(text)
            annotated_text_html = annotate_text(markup)

            image_url = url_for('static', filename=f'uploads/{file.filename}')

            return redirect(url_for('results', image_path=image_url, extracted_text=annotated_text_html))

    return render_template('index.html')

@app.route('/results')
def results():
    image_path = request.args.get('image_path')
    extracted_text = request.args.get('extracted_text')

    if not image_path or not extracted_text:
        return redirect(url_for('index'))

    return render_template('results.html', image_path=image_path, extracted_text=extracted_text)

if __name__ == '__main__':
    app.run(debug=True)