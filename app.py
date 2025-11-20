from flask import Flask, render_template, request, redirect, url_for
from ocr import perform_ocr
from ner import perform_ner
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
            # annotated_text = perform_ner(text_placeholder)

            annotated_text = perform_ner(text)

            image_url = url_for('static', filename=f'uploads/{file.filename}')

            return redirect(url_for('results', image_path=image_url, extracted_text=annotated_text))

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