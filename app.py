import os
import sqlite3
import json
from flask import Flask, render_template, request

from db import init_db
from text_processor import TextProcessor

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"


class App:
    def __init__(self):
        self.text_processor = TextProcessor()

    def run(self):
        """Запуск Flask-приложения."""

        @app.route("/", methods=["GET", "POST"])
        def index():
            if request.method == "POST":
                # Получаем список загруженных файлов
                files = request.files.getlist("files")

                # Проверяем, что хотя бы один файл был загружен
                if not files or all(file.filename == "" for file in files):
                    return "Ошибка: Необходимо загрузить хотя бы один файл.", 400

                texts = []
                for file in files:
                    if file and file.filename.endswith(".txt"):
                        # Сохраняем файл
                        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                        file.save(filepath)

                        # Читаем и обрабатываем текст
                        with open(filepath, "r", encoding="utf-8") as f:
                            text = f.read()
                            processed_text = self.text_processor.preprocess_text(text)
                            texts.append(processed_text)

                # Проверяем, что хотя бы один текст был успешно обработан
                if not texts:
                    return "Ошибка: Загружены некорректные файлы.", 400

                # Вычисляем TF-IDF для всех текстов
                results = self.text_processor.calculate_tf_idf(texts)[:50]

                # Отображаем результаты
                return render_template("results.html", results=results)

            return render_template("index.html")

        # Создаем папку для загрузок
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        app.run(debug=True)


@app.route("/save", methods=["POST"])
def save_results():
    try:
        # Получаем данные в формате JSON
        data = request.get_json()
        if not data:
            return "Ошибка: Данные не найдены.", 400

        # Сохраняем данные в базу данных
        conn = sqlite3.connect("tfidf.db")
        cursor = conn.cursor()

        for result in data:
            cursor.execute(
                """
                INSERT INTO results (word, tf, idf, tf_idf)
                VALUES (?, ?, ?, ?)
            """,
                (result["word"], result["tf"], result["idf"], result["tf_idf"]),
            )

        conn.commit()
        conn.close()

        return "Данные успешно сохранены!"

    except json.JSONDecodeError as e:
        return f"Ошибка при декодировании JSON: {str(e)}", 400


if __name__ == "__main__":
    app_instance = App()
    app_instance.run()
    init_db()
