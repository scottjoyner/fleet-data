"""Hermetic checks for optional, gitignored TSV tables and model archives.

These artifacts are not tracked in this checkout; synthetic fixtures exercise
the validators, and the real-file tests skip cleanly when nothing is present.
"""

import io
import tarfile
import unittest

from contracts import ROOT, expect_invalid, require, validate_node_table, validate_tar_headers


def _real_files(suffix):
    return [path for path in ROOT.rglob(f"*{suffix}") if ".git" not in path.relative_to(ROOT).parts]


def test_table_shape_and_unique_node_ids(tmp_path):
    table = tmp_path / "fleet-matrix.tsv"
    table.write_text("node_id\thostname\trole\nnode-1\thost-a\tgpu\nnode-2\thost-b\tcpu\n", encoding="utf-8")
    require(validate_node_table(table) == 2, "two data rows expected")

    bad = tmp_path / "bad.tsv"
    bad.write_text("node_id\thostname\nnode-1\thost-a\textra\n", encoding="utf-8")
    expect_invalid(validate_node_table, bad)

    bad.write_text("node_id\thostname\nnode-1\thost-a\nnode-1\thost-b\n", encoding="utf-8")
    expect_invalid(validate_node_table, bad)

    bad.write_text("node_id\thostname\nnode-1\t\n", encoding="utf-8")
    expect_invalid(validate_node_table, bad)


def test_tar_headers_reject_unsafe_members(tmp_path):
    good = tmp_path / "good.tar.gz"
    with tarfile.open(good, "w:gz") as archive:
        for name, payload in (("bundle/", None), ("bundle/manifest.txt", b"a=b\n")):
            member = tarfile.TarInfo(name)
            if payload is None:
                member.type = tarfile.DIRTYPE
                archive.addfile(member)
            else:
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
    require(validate_tar_headers(good) == 2, "two members expected")

    traversal = tmp_path / "traversal.tar.gz"
    with tarfile.open(traversal, "w:gz") as archive:
        member = tarfile.TarInfo("../outside.txt")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    expect_invalid(validate_tar_headers, traversal)

    link = tmp_path / "link.tar.gz"
    with tarfile.open(link, "w:gz") as archive:
        member = tarfile.TarInfo("link")
        member.type = tarfile.SYMTYPE
        member.linkname = "/etc/passwd"
        archive.addfile(member)
    expect_invalid(validate_tar_headers, link)

    oversized = tmp_path / "oversized.tar.gz"
    with tarfile.open(oversized, "w:gz") as archive:
        member = tarfile.TarInfo("big.bin")
        member.size = 2048
        archive.addfile(member, io.BytesIO(b"x" * 2048))
    expect_invalid(validate_tar_headers, oversized, max_member_bytes=1024)


def test_committed_optional_artifacts_if_any():
    tables = _real_files(".tsv")
    archives = _real_files(".tar.gz")
    if not tables and not archives:
        raise unittest.SkipTest("no gitignored *.tsv or *.tar.gz present in this checkout")
    for table in tables:
        validate_node_table(table)
    for archive in archives:
        validate_tar_headers(archive)
