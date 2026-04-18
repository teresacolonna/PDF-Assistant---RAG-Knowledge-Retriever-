import fitz
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

texts = {
    "report_2020_2022.pdf": (
        "Nel periodo 2020-2022 le emissioni di CO2 sono diminuite del 5%. "
        "Il PIL verde ha mostrato un trend crescente." 
        "Consumo energetico più alto nel 2020 rispetto al 2021."
    ),
    "report_2021_2023.pdf": (
        "Nel periodo 2021-2023 si osserva un calo continuo delle emissioni "
        "di gas climalteranti. Il PIL verde è aumentato di anno in anno."
    ),
    "report_2022_2024.pdf": (
        "La spesa per la protezione dell'ambiente è cresciuta fino al 2024. "
        "Il consumo energetico raggiunge il picco nel 2023." 
    ),
}

for name, content in texts.items():
    path = os.path.join(OUTPUT_DIR, name)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), content)
    doc.save(path)
    doc.close()
    print(f"Created {path}")
