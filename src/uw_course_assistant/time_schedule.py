"""Live UW schedules with bounded retries and explicit unavailable results."""

import asyncio
import json
import logging
import re
from urllib.parse import urlsplit
from uuid import uuid4

from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from playwright.async_api import TimeoutError as PlaywrightTimeoutError, async_playwright


logger = logging.getLogger(__name__)
MAX_ATTEMPTS = 3
NAVIGATION_TIMEOUT_MS = 30_000
LOGIN_TIMEOUT_MS = 300_000
TABLE_TIMEOUT_MS = 10_000
SCHEDULE_BASE_URL = "https://www.washington.edu/students/timeschd/"


class BrowserPool:
    """Keep Chromium alive between calls; each attempt owns its context."""

    _playwright = None
    _browser = None

    @classmethod
    async def get_browser(cls):
        if cls._playwright is None:
            cls._playwright = await async_playwright().start()
        if cls._browser is None or not cls._browser.is_connected():
            cls._browser = await cls._playwright.chromium.launch(headless=False)
        return cls._browser

    @classmethod
    async def close(cls):
        try:
            if cls._browser:
                await cls._browser.close()
        finally:
            cls._browser = None
            try:
                if cls._playwright:
                    await cls._playwright.stop()
            finally:
                cls._playwright = None


class _ScheduleFailure(Exception):
    def __init__(self, reason: str, *, retryable: bool = True):
        super().__init__(reason)
        self.reason = reason
        self.retryable = retryable


async def _fetch_schedule_html(url: str, quarter_year: str) -> str:
    context = None
    expected = urlsplit(url)

    def is_requested_page(candidate):
        actual = urlsplit(str(candidate))
        return actual.hostname == expected.hostname and actual.path == expected.path

    try:
        browser = await BrowserPool.get_browser()
        context = await browser.new_context()
        page = await context.new_page()
        response = await page.goto(
            url, timeout=NAVIGATION_TIMEOUT_MS, wait_until="domcontentloaded"
        )
        if response is None:
            raise _ScheduleFailure("empty_response")
        if response.status >= 400:
            raise _ScheduleFailure(
                f"http_{response.status}",
                retryable=response.status in (408, 429) or response.status >= 500,
            )

        if any(marker in page.url.lower()
               for marker in ("login", "signin", "identity", "shibboleth")):
            print("\n⚠️ Login required. Complete UW NetID login and 2FA in the browser.")
            print("⏳ Waiting up to 5 minutes for the requested schedule page...")
            try:
                await page.wait_for_url(
                    is_requested_page,
                    timeout=LOGIN_TIMEOUT_MS,
                )
            except PlaywrightTimeoutError as exc:
                # A retry would open another context and repeat the full login wait.
                raise _ScheduleFailure("login_timeout", retryable=False) from exc

        if not is_requested_page(page.url):
            raise _ScheduleFailure("unexpected_redirect", retryable=False)

        # Missing tables may be an error page or changed markup, never proof that
        # the requested quarter has no courses or seats.
        await page.wait_for_selector("table[bgcolor='#ccffcc']", timeout=TABLE_TIMEOUT_MS)
        return await page.content()
    finally:
        if context is not None:
            try:
                # Closing the context also closes any pages, including partially
                # initialized pages. Keep the shared browser available to other calls.
                await context.close()
            except Exception as exc:
                logger.warning("Schedule context cleanup failed (%s)", type(exc).__name__)


def _parse_schedule_documents(html: str) -> list[Document]:
    soup = BeautifulSoup(html, "html.parser")
    documents = []
    for course_table in soup.find_all("table", bgcolor="#ccffcc"):
        course_header = course_table.get_text(separator=" ", strip=True)
        current = course_table.find_next_sibling("table")
        sections = []
        while current and current.get("bgcolor") != "#ccffcc":
            pre = current.find("pre")
            if pre:
                text = pre.get_text(separator="  ", strip=True)
                if text:
                    sections.append(text)
            current = current.find_next_sibling("table")
        if course_header and sections:
            documents.append(Document(page_content=(
                f"Course Info: {course_header}\nSections:\n" + "\n---\n".join(sections)
            )))
    if not documents:
        raise _ScheduleFailure("no_structured_data")
    return documents


def _retrieve_schedule(documents: list[Document], query: str) -> str:
    # The tool controls retries. Disable SDK retries so the bound cannot multiply.
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", max_retries=0, request_timeout=30
    )
    vectorstore = None
    try:
        # Default Chroma collection names are shared across instances. Isolate
        # requests so a failed/new scrape cannot retrieve a previous quarter.
        vectorstore = Chroma(
            collection_name=f"uw_schedule_{uuid4().hex}", embedding_function=embeddings
        )
        vectorstore.add_documents(documents)
        docs = vectorstore.as_retriever(search_kwargs={"k": 10}).invoke(query)
        content = "\n\n".join(doc.page_content for doc in docs).strip()
        if not content:
            raise _ScheduleFailure("no_relevant_sections")
        return content
    finally:
        if vectorstore is not None:
            try:
                vectorstore.delete_collection()
            except Exception as exc:
                logger.warning("Schedule index cleanup failed (%s)", type(exc).__name__)


def _unavailable(url: str, attempts: int, reason: str) -> tuple[str, dict]:
    artifact = {
        "status": "unavailable", "source_url": url,
        "attempts": attempts, "reason": reason,
    }
    content = {
        **artifact,
        "message": "Live schedule data is currently unavailable; the requested schedule could not be verified.",
        "instruction": (
            "Tell the user that real-time schedule data is unavailable and provide the official source URL. "
            "Do not invent or infer class times, locations, instructors, or seat availability from memory, "
            "previous turns, or the static catalog. Do not treat this failure as evidence of no courses "
            "or no open seats. Do not call this tool again for the same request in this turn."
        ),
    }
    return json.dumps(content), artifact


@tool(response_format="content_and_artifact")
async def get_time_schedule(quarter_year: str, department: str, query: str) -> tuple[str, dict]:
    """Query live UW class times, locations, instructors, and seat availability.

    quarter_year is e.g. AUT2026, WIN2027, SPR2027, or SUM2027; department is
    the UW department URL code, e.g. cse or math; query is the user's question.
    Retries transient failures at most twice. An unavailable result means the
    real-time data could not be verified, NOT that there are no courses/seats.
    On failure, explicitly tell the user live data is unavailable, provide the
    source URL, and do not invent schedule details or repeat this call this turn.
    """
    url = SCHEDULE_BASE_URL
    try:
        quarter_year = quarter_year.strip().upper()
        department = department.strip().lower()
        if (not re.fullmatch(r"(?:WIN|SPR|SUM|AUT)\d{4}", quarter_year)
                or not re.fullmatch(r"[a-z][a-z0-9_-]*", department)
                or not query.strip()):
            return _unavailable(url, 0, "invalid_arguments")
        url = f"{SCHEDULE_BASE_URL}{quarter_year}/{department}.html"
    except Exception:
        return _unavailable(url, 0, "invalid_arguments")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        stage = "fetch"
        try:
            html = await _fetch_schedule_html(url, quarter_year)
            stage = "parse"
            documents = _parse_schedule_documents(html)
            stage = "retrieval"
            # Embedding and Chroma APIs are synchronous; keep the event loop free
            # while other tools or schedule requests are running.
            result = await asyncio.to_thread(_retrieve_schedule, documents, query)
            return result, {"status": "ok", "source_url": url, "attempts": attempt}
        except Exception as exc:
            # Catch all ordinary failures across browser, parser, embeddings and
            # retrieval. Cancellation/KeyboardInterrupt deliberately still propagate.
            reason = exc.reason if isinstance(exc, _ScheduleFailure) else f"{stage}_error"
            retryable = not isinstance(exc, _ScheduleFailure) or exc.retryable
            logger.warning(
                "Schedule attempt %d/%d failed: %s (%s)",
                attempt, MAX_ATTEMPTS, reason, type(exc).__name__,
            )
            if not retryable or attempt == MAX_ATTEMPTS:
                return _unavailable(url, attempt, reason)
            await asyncio.sleep(2 ** (attempt - 1))
