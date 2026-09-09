"""Structural regressions for the embedded, dependency-free admin surface.

Interactive scenarios live in admin_ui_browser.mjs and use an isolated fixture API.
"""
from html.parser import HTMLParser

from src.admin import _ADMIN_HTML


class UIParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


def elements():
    parser = UIParser()
    parser.feed(_ADMIN_HTML)
    return parser.elements


def test_ui_has_four_primary_destinations_and_secondary_settings():
    tabs = [attrs for _, attrs in elements() if "data-tab" in attrs]
    assert [attrs["data-tab"] for attrs in tabs] == [
        "overview", "models", "connections", "activity", "settings",
    ]
    assert tabs[0]["aria-selected"] == "true"
    assert "settings-nav" in tabs[-1]["class"]
    indexed = {attrs["id"]: attrs for _, attrs in elements() if "id" in attrs}
    for tab in tabs:
        panel = indexed[tab["aria-controls"]]
        assert panel["role"] == "tabpanel"
        assert panel["aria-labelledby"] == tab["id"]


def test_ui_ids_are_unique_and_errors_are_announced():
    ids = [attrs["id"] for _, attrs in elements() if "id" in attrs]
    assert len(ids) == len(set(ids))
    indexed = {attrs["id"]: attrs for _, attrs in elements() if "id" in attrs}
    for id_ in ("dashErr", "lockedErr", "usageErr"):
        assert indexed[id_]["role"] == "alert"
    assert _ADMIN_HTML.index('id="dashErr"') < _ADMIN_HTML.index('id="panel-overview"')


def test_ui_labels_all_named_fields():
    nodes = elements()
    labels = {attrs["for"] for tag, attrs in nodes if tag == "label" and "for" in attrs}
    for tag, attrs in nodes:
        if tag in {"input", "select", "textarea"} and attrs.get("type") != "checkbox":
            assert attrs["id"] in labels or attrs.get("aria-label"), attrs["id"]


def test_ui_has_honest_routing_and_bounded_activity_copy():
    assert "Upstream availability not verified" in _ADMIN_HTML
    assert "Latest 50 across all time" in _ADMIN_HTML
    assert "Failover / attempt chain" in _ADMIN_HTML
    assert "does not record the complete attempt chain" in _ADMIN_HTML
    assert "read-only" in _ADMIN_HTML.lower()
    assert 'id="modelFormset"' in _ADMIN_HTML
    assert 'id="providerFormset"' in _ADMIN_HTML



def test_admin_model_status_exposes_scoped_image_policy_without_probes(monkeypatch):
    from src import admin
    rows = [
        {"name": "local", "locality": "local", "vision": False},
        {"name": "cloud", "locality": "cloud", "vision": False},
        {"name": "native", "locality": "cloud", "vision": True},
        {"name": "mixed", "locality": "mixed", "vision": False},
        {"name": "composite", "locality": None, "composite": {"text_model": "local", "vision_model": "native"}},
    ]
    monkeypatch.setattr(admin, "model_status", lambda: rows)
    monkeypatch.setenv("GATEWAY_VISION_FALLBACK_LOCAL", "local-helper")
    monkeypatch.setenv("GATEWAY_VISION_FALLBACK_CLOUD", "cloud-helper")
    monkeypatch.delenv("GATEWAY_VISION_FALLBACK_MODE", raising=False)
    result = admin._admin_model_status()
    assert result[0]["vision_route"] == {"model": "local-helper", "mode": "extract_then_answer"}
    assert result[1]["vision_route"] == {"model": "cloud-helper", "mode": "extract_then_answer"}
    assert all(row["vision_route"] is None for row in result[2:])
