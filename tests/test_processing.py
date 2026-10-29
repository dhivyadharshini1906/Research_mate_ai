from app.processing import chunk_pages, clean_text, guess_title
def test_clean_text():
    assert clean_text("hello   world\n\n\nagain") == "hello world\n\nagain"
def test_chunk_pages_preserves_page():
    pages=[{"page_number":4,"text":" ".join(["word"]*250)}]
    chunks=chunk_pages(pages,chunk_words=100,overlap_words=20)
    assert len(chunks)==3
    assert all(c["page_number"]==4 for c in chunks)
def test_chunk_validation():
    try: chunk_pages([{"page_number":1,"text":"hello"}],chunk_words=20)
    except ValueError: pass
    else: raise AssertionError("Expected ValueError")
def test_guess_title():
    assert guess_title([{"page_number":1,"text":"A useful research paper title\nMore text"}],"paper.pdf")=="A useful research paper title"

