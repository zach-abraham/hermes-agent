from tools.file_tools import WRITE_FILE_SCHEMA


def test_write_file_schema_guides_large_generated_content():
    content_description = WRITE_FILE_SCHEMA["parameters"]["properties"]["content"]["description"]
    combined = f"{WRITE_FILE_SCHEMA['description']} {content_description}"

    assert "200 lines" in combined
    assert "8 KB" in combined
    assert "execute_code" in combined
    assert "hermes_tools.write_file" in combined
