def clean_text(text: str) -> str:
    return " ".join(str(text).split())

def generate_symptom_summary(text: str) -> str:
    text = text.lower()
    found = []

    if "baş" in text and "dön" in text:
        found.append("baş dönmesi")
    if "bulant" in text:
        found.append("mide bulantısı")
    if "denge" in text:
        found.append("denge kaybı")
    if "nefes" in text:
        found.append("nefes darlığı")
    if "göğ" in text:
        found.append("göğüs ağrısı veya baskı")
    if "çarpınt" in text or "kalb" in text:
        found.append("çarpıntı")
    if "yanma" in text and "idrar" in text:
        found.append("idrarda yanma")
    if "sık" in text and "tuvalet" in text:
        found.append("sık idrara çıkma")
    if "karın" in text or "mide" in text:
        found.append("karın/mide rahatsızlığı")
    if "baş ağr" in text:
        found.append("baş ağrısı")
    if "yorgun" in text or "halsiz" in text:
        found.append("halsizlik/yorgunluk")

    if not found:
        return "Girdide belirgin semptom ifadeleri sınırlı."

    summary = ", ".join(found)
    return f"Girdide öne çıkan semptomlar: {summary}."
