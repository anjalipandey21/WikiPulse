"""Conservative deterministic filtering for non-content Wikipedia pages."""

from collections.abc import Iterable
import re

from ..models import Article


MAIN_PAGE_REASON = "main_page"
UNKNOWN_TITLE_REASON = "unknown_title"
DISAMBIGUATION_REASON = "disambiguation"

ADMINISTRATIVE_NAMESPACES = frozenset(
    {
        "special",
        "media",
        "talk",
        "user",
        "user talk",
        "wikipedia",
        "wikipedia talk",
        "file",
        "file talk",
        "mediawiki",
        "mediawiki talk",
        "template",
        "template talk",
        "help",
        "help talk",
        "category",
        "category talk",
        "portal",
        "portal talk",
        "draft",
        "draft talk",
        "mos",
        "mos talk",
        "timedtext",
        "timedtext talk",
        "module",
        "module talk",
        "event",
        "event talk",
        # English Wikipedia aliases for administrative namespaces.
        "image",
        "image talk",
        "project",
        "project talk",
        "tm",
        "wp",
        "wt",
    }
)
DISAMBIGUATION_SUFFIX = re.compile(r"\(disambiguation\)$", re.IGNORECASE)


def get_noise_reason(normalized_title: str) -> str | None:
    """Return a stable reason when a normalized title is obvious noise."""
    casefolded_title = normalized_title.casefold()
    if casefolded_title == "main page":
        return MAIN_PAGE_REASON
    if normalized_title == "-":
        return UNKNOWN_TITLE_REASON

    namespace, separator, _ = normalized_title.partition(":")
    casefolded_namespace = namespace.casefold()
    if separator and casefolded_namespace in ADMINISTRATIVE_NAMESPACES:
        return f"administrative_namespace:{casefolded_namespace}"

    if DISAMBIGUATION_SUFFIX.search(normalized_title):
        return DISAMBIGUATION_REASON

    return None


def filter_noise_articles(articles: Iterable[Article]) -> list[Article]:
    """Remove obvious noise without mutating or reordering retained articles."""
    return [
        article
        for article in articles
        if get_noise_reason(article.normalized_title) is None
    ]
