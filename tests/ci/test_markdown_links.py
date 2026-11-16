"""Local-link checks cover files, anchors and code-example boundaries."""

from scripts.ci.check_markdown_links import anchors, check_file


def test_links_and_reference_destinations_report_missing_files_and_anchors(tmp_path):
    page = tmp_path / "page.md"
    (tmp_path / "other page.md").write_text(
        "# Target\n## Repeat\n## Repeat\n<a id='explicit'></a>\n"
    )
    page.write_text(
        "[ok](<other page.md#target>)\n"
        "[ok](other%20page.md#repeat-1)\n"
        "[ok](other%20page.md#explicit)\n"
        "[bad](other%20page.md#absent)\n"
        "[ref]: missing.md\n"
        "[external](https://example.org/missing)\n"
        "```markdown\n[example](not-real.md)\n```\n"
    )
    errors = check_file(page)
    assert len(errors) == 2
    assert "missing anchor" in errors[0]
    assert "missing destination" in errors[1]


def test_anchors_support_inline_markup_unicode_duplicates_and_fences():
    assert anchors("# A `value` & [link](other.md)\n# Café\n# Café\n~~~\n# Hidden\n~~~\n") == {
        "a-value--link",
        "café",
        "café-1",
    }


def test_multiline_labels_and_html_assets_are_checked(tmp_path):
    page = tmp_path / "page.md"
    (tmp_path / "figure.png").write_bytes(b"image")
    page.write_text(
        "[A label spanning\ntwo lines](missing.md)\n"
        '<img src="figure.png" alt="Available figure">\n'
        "<video src='missing.mp4'></video>\n"
        '<a href="https://example.org">External</a>\n'
        '```html\n<img src="example-only.png">\n```\n'
    )
    errors = check_file(page)
    assert len(errors) == 2
    assert any("missing.md" in error for error in errors)
    assert any("missing.mp4" in error for error in errors)
