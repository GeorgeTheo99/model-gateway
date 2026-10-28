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


def test_ui_has_five_primary_destinations_and_secondary_settings():
    tabs = [attrs for _, attrs in elements() if "data-tab" in attrs]
    assert [attrs["data-tab"] for attrs in tabs] == [
        "overview", "models", "connections", "activity", "consumers", "settings",
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



def test_model_drawer_describes_image_notes_for_text_only_models():
    assert "Text only: each image is replaced with a note saying it was omitted." in _ADMIN_HTML
    assert "vision_route" not in _ADMIN_HTML


def test_ui_consumer_changes_are_gated_confirmed_and_show_keys_once():
    indexed = {attrs["id"]: attrs for _, attrs in elements() if "id" in attrs}
    assert "hidden" in indexed["newKeyPanel"]
    assert "readonly" in indexed["newKeyValue"]
    assert indexed["consumerFormset"]["class"] == "formset"
    assert indexed["consumerConfirm"]["class"] == "formset"
    assert "This key is not shown again" in _ADMIN_HTML
    assert "{ confirm: p.id }" in _ADMIN_HTML
    # The key never reaches persistent browser storage; lock clears it.
    lock_body = _ADMIN_HTML.split("function lock(")[1].split("async function unlock(")[0]
    assert "clearNewKey();" in lock_body
    assert _ADMIN_HTML.count("sessionStorage.setItem(") == 1
    assert 'sessionStorage.setItem(STORAGE_KEY, $("adminKey").value.trim())' in _ADMIN_HTML
    assert "localStorage" not in _ADMIN_HTML
