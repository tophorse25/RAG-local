from pypdf import PdfReader

PDF_PATH = r"C:\Users\12572\RAG-local\samplenvidia.pdf"
needle = "For the quarterly period ended"

r = PdfReader(PDF_PATH)
print("Total pages:", len(r.pages))

hits = []
for i in range(min(5, len(r.pages))):
    text = (r.pages[i].extract_text() or "")
    if needle.lower() in text.lower():
        hits.append(i)

print("Hit pages:", hits)

print("\n--- PAGE 0 PREVIEW ---\n")
print((r.pages[0].extract_text() or "")[:2000])