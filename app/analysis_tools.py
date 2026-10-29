import re

def compare_papers(papers):
    patterns = {
        "objective / problem": r"(?is)(?:objective|aim|purpose|we propose|this paper presents|this study investigates)[\s:,-]{0,20}(.{0,420})",
        "methods": r"(?is)(?:methodology|methods?|approach|proposed method|we use|we used)[\s:,-]{0,20}(.{0,420})",
        "limitations": r"(?is)(?:limitations?|future work|future research)[\s:,-]{0,20}(.{0,420})"
    }
    out = []
    for p in papers:
        full = "\n".join(c["text"] for c in p["chunks"])
        fields = {}
        for label, pattern in patterns.items():
            m = re.search(pattern, full)
            fields[label] = re.sub(r"\s+", " ", m.group(1)).strip() if m else "Not clearly identified automatically; review source."
        out.append({"title": p["title"], **fields})
    return out

