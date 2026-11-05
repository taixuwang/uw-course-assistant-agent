"""Compare browser and context lifecycle costs as a local microbenchmark.

This does not use the application's BrowserPool, load a page, or measure
end-to-end request latency. Install the Playwright Chromium browser first.
"""

import argparse
import asyncio
import statistics
from time import perf_counter

from playwright.async_api import async_playwright


def positive_int(value):
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("iterations must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("iterations must be a positive integer")
    return number


async def use_context(browser):
    """Create and close a context and its page, even if page creation fails."""
    context = await browser.new_context()
    try:
        page = await context.new_page()
        try:
            pass  # No navigation: this measures browser/context setup only.
        finally:
            await page.close()
    finally:
        await context.close()


async def run_benchmark(iterations):
    cold_samples = []
    reused_samples = []
    async with async_playwright() as playwright:
        # Both modes use the same already-started Playwright driver. Cold times
        # include launching and closing a browser; reused times exclude the
        # one-time browser startup and shutdown but include the same context work.
        for _ in range(iterations):
            start = perf_counter()
            browser = await playwright.chromium.launch(headless=True)
            try:
                await use_context(browser)
            finally:
                await browser.close()
            cold_samples.append((perf_counter() - start) * 1000)

        browser = await playwright.chromium.launch(headless=True)
        try:
            for _ in range(iterations):
                start = perf_counter()
                await use_context(browser)
                reused_samples.append((perf_counter() - start) * 1000)
        finally:
            await browser.close()
    return cold_samples, reused_samples


def print_samples(name, samples):
    print(f"{name}: mean={statistics.mean(samples):.2f} ms, "
          f"median={statistics.median(samples):.2f} ms, n={len(samples)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=positive_int, default=5,
                        help="number of samples per mode (default: 5)")
    args = parser.parse_args()
    print("Browser lifecycle microbenchmark only; excludes navigation, network, "
          "and application BrowserPool behavior.")
    cold_samples, reused_samples = asyncio.run(run_benchmark(args.iterations))
    print_samples("Cold browser, context and page lifecycle", cold_samples)
    print_samples("Reused browser, context and page lifecycle", reused_samples)


if __name__ == "__main__":
    main()
