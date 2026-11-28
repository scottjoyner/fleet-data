"""Checks that documented and embedded references resolve inside this checkout."""

from contracts import BIN, DOCS, ROOT, markdown_links, require


def test_python_harnesses_compile_without_executing_them():
    scripts = sorted(BIN.glob("*.py"))
    require(len(scripts) >= 2, f"expected both Python harnesses, found {len(scripts)}")
    for path in scripts:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_docs_relative_links_resolve():
    doc = DOCS / "DATA_CONTRACTS.md"
    links = [link for link in markdown_links(doc.read_text(encoding="utf-8")) if not link.startswith(("http://", "https://", "mailto:", "#"))]
    require(bool(links), "DATA_CONTRACTS.md has no local links")
    for link in links:
        target = (doc.parent / link.split("#", 1)[0]).resolve()
        require(target.exists(), f"broken documentation link: {link}")


def test_readme_links_point_at_this_repository():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    urls = [link for link in markdown_links(text) if link.startswith("https://")]
    require(bool(urls), "README.md has no repository links")
    for url in urls:
        require(url.startswith("https://github.com/scottjoyner/fleet-data/"), f"unexpected README link: {url}")
