import asyncio
import time
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright

# -------------------------------------------------------------
# 1. Old Method: Cold launch a brand-new Browser process per request
# -------------------------------------------------------------
async def test_old_cold_launch(iterations=5):
    latencies = []
    print("[Testing] Measuring Old Method (Cold Launching Browser per request)...")
    for i in range(iterations):
        start = time.perf_counter()
        p = await async_playwright().start()
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.close()
        await browser.close()
        await p.stop()
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000
        latencies.append(elapsed_ms)
        print(f"  Run {i+1}: {elapsed_ms:.2f} ms")
    return latencies

# -------------------------------------------------------------
# 2. New Method: Reuse long-lived Browser singleton with Async Pool
# -------------------------------------------------------------
async def test_new_browser_pool(iterations=5):
    latencies = []
    print("\n[Testing] Measuring New Method (Async Playwright Pool - Reusing Browser Singleton)...")
    
    # Global cold launch happens once (at app startup)
    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=True)
    
    # Measure latency of creating lightweight tab pages on existing browser
    for i in range(iterations):
        start = time.perf_counter()
        context = await browser.new_context()
        page = await context.new_page()
        await page.close()
        await context.close()
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000
        latencies.append(elapsed_ms)
        print(f"  Run {i+1}: {elapsed_ms:.2f} ms")
        
    await browser.close()
    await p.stop()
    return latencies

# -------------------------------------------------------------
# 3. Run benchmark and output comparative metrics
# -------------------------------------------------------------
async def main():
    iterations = 5
    old_times = await test_old_cold_launch(iterations)
    new_times = await test_new_browser_pool(iterations)
    
    avg_old = sum(old_times) / len(old_times)
    avg_new = sum(new_times) / len(new_times)
    speedup = ((avg_old - avg_new) / avg_old) * 100

    print("\n" + "="*55)
    print("PAGE INITIALIZATION LATENCY BENCHMARK RESULTS")
    print("="*55)
    print(f"Old Method (Cold Launch Browser) Avg:  {avg_old:.2f} ms")
    print(f"New Method (Async Browser Pool) Avg:   {avg_new:.2f} ms")
    print(f"Speedup / Latency Reduction:           {speedup:.2f}%")
    print(f"Time Saved per Page Init:              {(avg_old - avg_new):.2f} ms")
    print("="*55)

if __name__ == "__main__":
    asyncio.run(main())
