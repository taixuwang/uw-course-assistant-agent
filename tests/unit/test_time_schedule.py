"""Offline schedule failure regressions; never launch a browser or call an API."""

import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, call, patch

from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from uw_course_assistant import time_schedule as schedule


URL = "https://www.washington.edu/students/timeschd/AUT2026/cse.html"
HTML = """
<table bgcolor="#ccffcc"><tr><td>CSE 121</td></tr></table>
<table><tr><td><pre>12345 A 5 MWF 10:30-11:20 OPEN</pre></td></tr></table>
"""
DOCUMENTS = [Document(page_content="CSE 121: MWF 10:30-11:20 OPEN")]


async def invoke_schedule(**overrides):
    args = {"quarter_year": "AUT2026", "department": "cse", "query": "CSE 121 times"}
    args.update(overrides)
    return await schedule.get_time_schedule.ainvoke({
        "name": "get_time_schedule", "type": "tool_call", "id": "schedule-call", "args": args,
    })


def browser_doubles(status=200, url=URL):
    page = SimpleNamespace(
        url=url,
        goto=AsyncMock(return_value=SimpleNamespace(status=status)),
        wait_for_url=AsyncMock(),
        wait_for_selector=AsyncMock(),
        content=AsyncMock(return_value=HTML),
    )
    context = SimpleNamespace(new_page=AsyncMock(return_value=page), close=AsyncMock())
    browser = SimpleNamespace(new_context=AsyncMock(return_value=context), close=AsyncMock())
    return browser, context, page


class ScheduleToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_transient_failure_recovers_and_returns_tool_artifact(self):
        with patch.object(schedule, "_fetch_schedule_html", new_callable=AsyncMock) as fetch, \
             patch.object(schedule, "_parse_schedule_documents", return_value=DOCUMENTS), \
             patch.object(schedule, "_retrieve_schedule", return_value="verified schedule") as retrieve, \
             patch.object(schedule.asyncio, "sleep", new_callable=AsyncMock) as sleep:
            fetch.side_effect = [RuntimeError("temporary connection failure"), HTML]
            result = await invoke_schedule()

        self.assertIsInstance(result, ToolMessage)
        self.assertEqual(result.content, "verified schedule")
        self.assertEqual(result.artifact["status"], "ok")
        self.assertEqual(result.artifact["attempts"], 2)
        self.assertEqual(result.artifact["source_url"], URL)
        self.assertEqual(fetch.await_count, 2)
        sleep.assert_awaited_once_with(1)
        retrieve.assert_called_once_with(DOCUMENTS, "CSE 121 times")

    async def test_all_pipeline_exceptions_exhaust_exactly_three_attempts(self):
        for failing_stage in ("fetch", "parse", "retrieve"):
            with self.subTest(stage=failing_stage), \
                 patch.object(schedule, "_fetch_schedule_html", new_callable=AsyncMock) as fetch, \
                 patch.object(schedule, "_parse_schedule_documents", return_value=DOCUMENTS) as parse, \
                 patch.object(schedule, "_retrieve_schedule", return_value="verified schedule") as retrieve, \
                 patch.object(schedule.asyncio, "sleep", new_callable=AsyncMock) as sleep:
                fetch.return_value = HTML
                stages = {"fetch": fetch, "parse": parse, "retrieve": retrieve}
                stages[failing_stage].side_effect = RuntimeError("secret credential must not be echoed")
                result = await invoke_schedule()

                self.assertEqual(result.artifact["status"], "unavailable")
                self.assertEqual(result.artifact["attempts"], 3)
                self.assertEqual(json.loads(result.content)["status"], "unavailable")
                self.assertNotIn("secret credential", result.content)
                self.assertEqual(fetch.await_count, 3)
                self.assertEqual(stages[failing_stage].call_count, 3)
                self.assertEqual(sleep.await_args_list, [call(1), call(2)])

    async def test_invalid_arguments_do_not_start_browser(self):
        for overrides in ({"quarter_year": "FALL2026"}, {"department": "../cse"}, {"query": "  "}):
            with self.subTest(overrides=overrides), \
                 patch.object(schedule.BrowserPool, "get_browser", new_callable=AsyncMock) as browser:
                result = await invoke_schedule(**overrides)
                self.assertEqual(result.artifact["status"], "unavailable")
                self.assertEqual(result.artifact["attempts"], 0)
                browser.assert_not_awaited()

    async def test_permanent_http_errors_stop_without_retry(self):
        for status in (400, 401, 403, 404):
            browser, context, page = browser_doubles(status=status)
            with self.subTest(status=status), \
                 patch.object(schedule.BrowserPool, "get_browser", new_callable=AsyncMock, return_value=browser), \
                 patch.object(schedule.asyncio, "sleep", new_callable=AsyncMock) as sleep:
                result = await invoke_schedule()
                self.assertEqual(result.artifact["status"], "unavailable")
                self.assertEqual(result.artifact["attempts"], 1)
                page.goto.assert_awaited_once()
                context.close.assert_awaited_once()
                browser.close.assert_not_awaited()
                sleep.assert_not_awaited()

    async def test_retryable_http_errors_are_bounded(self):
        for status in (408, 429, 500, 503):
            browser, context, page = browser_doubles(status=status)
            with self.subTest(status=status), \
                 patch.object(schedule.BrowserPool, "get_browser", new_callable=AsyncMock, return_value=browser), \
                 patch.object(schedule.asyncio, "sleep", new_callable=AsyncMock) as sleep:
                result = await invoke_schedule()
                self.assertEqual(result.artifact["status"], "unavailable")
                self.assertEqual(result.artifact["attempts"], 3)
                self.assertEqual(page.goto.await_count, 3)
                self.assertEqual(context.close.await_count, 3)
                self.assertEqual(sleep.await_args_list, [call(1), call(2)])

    async def test_login_timeout_is_not_repeated(self):
        browser, context, page = browser_doubles(url="https://identity.uw.edu/login")
        page.wait_for_url.side_effect = PlaywrightTimeoutError("login timeout")
        with patch.object(schedule.BrowserPool, "get_browser", new_callable=AsyncMock, return_value=browser), \
             patch.object(schedule.asyncio, "sleep", new_callable=AsyncMock) as sleep:
            result = await invoke_schedule()
        self.assertEqual(result.artifact["status"], "unavailable")
        self.assertEqual(result.artifact["attempts"], 1)
        page.wait_for_url.assert_awaited_once()
        self.assertEqual(page.wait_for_url.await_args.kwargs["timeout"], 300000)
        context.close.assert_awaited_once()
        sleep.assert_not_awaited()

    async def test_redirect_to_another_quarter_cannot_return_wrong_schedule(self):
        browser, context, page = browser_doubles(url=URL.replace("AUT2026", "SPR2026"))
        with patch.object(schedule.BrowserPool, "get_browser", new_callable=AsyncMock, return_value=browser), \
             patch.object(schedule, "_retrieve_schedule") as retrieve, \
             patch.object(schedule.asyncio, "sleep", new_callable=AsyncMock) as sleep:
            result = await invoke_schedule()
        self.assertEqual(result.artifact["status"], "unavailable")
        self.assertEqual(result.artifact["attempts"], 1)
        self.assertEqual(result.artifact["reason"], "unexpected_redirect")
        retrieve.assert_not_called()
        page.content.assert_not_awaited()
        context.close.assert_awaited_once()
        sleep.assert_not_awaited()

    async def test_successful_login_continues_with_requested_schedule(self):
        browser, context, page = browser_doubles(url="https://identity.uw.edu/login")

        async def complete_login(predicate, **kwargs):
            self.assertFalse(predicate(URL.replace("AUT2026", "SPR2026")))
            self.assertTrue(predicate(URL))
            page.url = URL

        page.wait_for_url.side_effect = complete_login
        with patch.object(schedule.BrowserPool, "get_browser", new_callable=AsyncMock, return_value=browser), \
             patch.object(schedule, "_retrieve_schedule", return_value="verified schedule"):
            result = await invoke_schedule()
        self.assertEqual(result.artifact["status"], "ok")
        self.assertEqual(result.content, "verified schedule")
        page.wait_for_url.assert_awaited_once()
        context.close.assert_awaited_once()

    async def test_missing_table_is_unavailable_and_never_reaches_retrieval(self):
        browser, context, page = browser_doubles()
        page.wait_for_selector.side_effect = PlaywrightTimeoutError("no course table")
        with patch.object(schedule.BrowserPool, "get_browser", new_callable=AsyncMock, return_value=browser), \
             patch.object(schedule, "_retrieve_schedule") as retrieve, \
             patch.object(schedule.asyncio, "sleep", new_callable=AsyncMock):
            result = await invoke_schedule()
        self.assertEqual(result.artifact["status"], "unavailable")
        self.assertEqual(result.artifact["attempts"], 3)
        self.assertEqual(context.close.await_count, 3)
        retrieve.assert_not_called()
        page.content.assert_not_awaited()

    async def test_empty_http_response_is_unavailable(self):
        browser, context, page = browser_doubles()
        page.goto.return_value = None
        with patch.object(schedule.BrowserPool, "get_browser", new_callable=AsyncMock, return_value=browser), \
             patch.object(schedule.asyncio, "sleep", new_callable=AsyncMock):
            result = await invoke_schedule()
        self.assertEqual(result.artifact["status"], "unavailable")
        self.assertEqual(result.artifact["attempts"], 3)
        self.assertEqual(result.artifact["reason"], "empty_response")
        self.assertEqual(context.close.await_count, 3)

    async def test_cancellation_propagates_without_retry(self):
        with patch.object(schedule, "_fetch_schedule_html", new_callable=AsyncMock) as fetch, \
             patch.object(schedule.asyncio, "sleep", new_callable=AsyncMock) as sleep:
            fetch.side_effect = asyncio.CancelledError()
            with self.assertRaises(asyncio.CancelledError):
                await invoke_schedule()
        fetch.assert_awaited_once()
        sleep.assert_not_awaited()


class BrowserCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_page_failure_still_closes_context(self):
        browser, context, _ = browser_doubles()
        context.new_page.side_effect = RuntimeError("cannot create page")
        with patch.object(schedule.BrowserPool, "get_browser", new_callable=AsyncMock, return_value=browser):
            with self.assertRaisesRegex(RuntimeError, "cannot create page"):
                await schedule._fetch_schedule_html(URL, "AUT2026")
        context.close.assert_awaited_once()
        browser.close.assert_not_awaited()

    async def test_cleanup_failure_does_not_replace_navigation_failure(self):
        browser, context, page = browser_doubles()
        page.goto.side_effect = RuntimeError("original navigation failure")
        context.close.side_effect = RuntimeError("cleanup failure")
        with patch.object(schedule.BrowserPool, "get_browser", new_callable=AsyncMock, return_value=browser):
            with self.assertRaisesRegex(RuntimeError, "original navigation failure"):
                await schedule._fetch_schedule_html(URL, "AUT2026")
        context.close.assert_awaited_once()

    async def test_cleanup_failure_does_not_discard_valid_html(self):
        browser, context, _ = browser_doubles()
        context.close.side_effect = RuntimeError("cleanup failure")
        with patch.object(schedule.BrowserPool, "get_browser", new_callable=AsyncMock, return_value=browser):
            result = await schedule._fetch_schedule_html(URL, "AUT2026")
        self.assertEqual(result, HTML)
        context.close.assert_awaited_once()

    async def test_cancellation_still_closes_context(self):
        browser, context, page = browser_doubles()
        page.goto.side_effect = asyncio.CancelledError()
        with patch.object(schedule.BrowserPool, "get_browser", new_callable=AsyncMock, return_value=browser):
            with self.assertRaises(asyncio.CancelledError):
                await schedule._fetch_schedule_html(URL, "AUT2026")
        context.close.assert_awaited_once()


class ScheduleParsingAndRetrievalTests(unittest.TestCase):
    def test_parser_rejects_error_pages_and_empty_sections(self):
        for html in ("<h1>Service unavailable</h1>",
                     '<table bgcolor="#ccffcc"><tr><td>CSE 121</td></tr></table>',
                     '<table bgcolor="#ccffcc"></table><table><pre>  </pre></table>'):
            with self.subTest(html=html), self.assertRaises(schedule._ScheduleFailure):
                schedule._parse_schedule_documents(html)

    def test_parser_keeps_sections_with_their_course(self):
        html = HTML + HTML.replace("CSE 121", "CSE 122").replace("12345", "54321")
        docs = schedule._parse_schedule_documents(html)
        self.assertEqual(len(docs), 2)
        self.assertIn("CSE 121", docs[0].page_content)
        self.assertIn("12345", docs[0].page_content)
        self.assertNotIn("54321", docs[0].page_content)
        self.assertIn("CSE 122", docs[1].page_content)

    def test_retrieval_uses_separate_collections_and_cleans_up(self):
        stores = [Mock(), Mock()]
        for store in stores:
            store.as_retriever.return_value.invoke.return_value = DOCUMENTS
        with patch.object(schedule, "OpenAIEmbeddings") as embeddings, \
             patch.object(schedule, "Chroma", side_effect=stores) as chroma:
            first = schedule._retrieve_schedule(DOCUMENTS, "first query")
            second = schedule._retrieve_schedule(DOCUMENTS, "second query")
        self.assertEqual(first, DOCUMENTS[0].page_content)
        self.assertEqual(second, first)
        names = [item.kwargs["collection_name"] for item in chroma.call_args_list]
        self.assertNotEqual(names[0], names[1])
        for embedding_call in embeddings.call_args_list:
            self.assertEqual(embedding_call.kwargs["max_retries"], 0)
            timeout = embedding_call.kwargs.get("request_timeout", embedding_call.kwargs.get("timeout"))
            self.assertEqual(timeout, 30)
        for store in stores:
            store.add_documents.assert_called_once_with(DOCUMENTS)
            store.delete_collection.assert_called_once_with()

    def test_embedding_and_query_errors_still_delete_collection(self):
        for stage in ("embedding", "query"):
            store = Mock()
            target = store.add_documents if stage == "embedding" else store.as_retriever.return_value.invoke
            target.side_effect = RuntimeError(f"{stage} failed")
            with self.subTest(stage=stage), \
                 patch.object(schedule, "OpenAIEmbeddings"), \
                 patch.object(schedule, "Chroma", return_value=store):
                with self.assertRaisesRegex(RuntimeError, f"{stage} failed"):
                    schedule._retrieve_schedule(DOCUMENTS, "question")
                store.delete_collection.assert_called_once_with()

    def test_collection_cleanup_failure_preserves_result(self):
        store = Mock()
        store.as_retriever.return_value.invoke.return_value = DOCUMENTS
        store.delete_collection.side_effect = RuntimeError("cleanup failed")
        with patch.object(schedule, "OpenAIEmbeddings"), \
             patch.object(schedule, "Chroma", return_value=store):
            result = schedule._retrieve_schedule(DOCUMENTS, "question")
        self.assertEqual(result, DOCUMENTS[0].page_content)

    def test_empty_retrieval_is_failure_not_claim_of_no_courses(self):
        store = Mock()
        store.as_retriever.return_value.invoke.return_value = []
        with patch.object(schedule, "OpenAIEmbeddings"), \
             patch.object(schedule, "Chroma", return_value=store):
            with self.assertRaises(schedule._ScheduleFailure):
                schedule._retrieve_schedule(DOCUMENTS, "question")
        store.delete_collection.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
