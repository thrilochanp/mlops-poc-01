from clinical_mlops.data_pipeline.normalize import clean_text, extract_fields, normalize_document, redact_pii


def test_extract_fields_finds_mrn_and_dob():
    text = "Patient: Jane Doe   MRN: MRN-123456   DOB: 01/02/1970\nDate of Service: 05/01/2026\nDr. Smith"
    fields = extract_fields(text)
    assert fields["mrn"] == "MRN-123456"
    assert fields["dob"] == "01/02/1970"
    assert fields["service_date"] == "05/01/2026"
    assert fields["provider"] == "Smith"


def test_redact_pii_masks_identifiers():
    text = "Patient: Jane Doe   MRN: MRN-123456   DOB: 01/02/1970\nSSN 123-45-6789"
    redacted = redact_pii(text)
    assert "Jane Doe" not in redacted
    assert "MRN-123456" not in redacted
    assert "01/02/1970" not in redacted
    assert "123-45-6789" not in redacted
    assert "[REDACTED]" in redacted


def test_clean_text_collapses_whitespace():
    assert clean_text("a   b\n\n\n\nc") == "a b\n\nc"


def test_normalize_document_redacts_and_extracts():
    raw = "Patient: John Q Public   MRN: MRN-999999   DOB: 03/03/1980\nDr. Lee\nFINDINGS: stable"
    record = normalize_document("doc_0001", "radiology_report", raw)
    assert record["doc_id"] == "doc_0001"
    assert record["doc_type"] == "radiology_report"
    assert record["mrn"] == "MRN-999999"
    assert "John Q Public" not in record["text"]
    assert "MRN-999999" not in record["text"]
