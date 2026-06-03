import requests
import re
import json
from bs4 import BeautifulSoup
from tqdm import tqdm
from typing import List, Dict


BASE_URL = "https://www.washington.edu/students/crscat/"
HEADERS = {"User-Agent": "Mozilla/5.0"}


# -----------------------------
# 1. Get all department pages
# -----------------------------
def get_departments():
    html = requests.get(BASE_URL, headers=HEADERS, timeout=10).text
    soup = BeautifulSoup(html, "lxml")

    links = []
    for a in soup.select("a"):
        href = a.get("href", "")
        if re.match(r"^[a-z]+\.html$", href):
            links.append(BASE_URL + href)

    return links


# -----------------------------
# 2. Download department page
# -----------------------------
def fetch_department(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.text


# -----------------------------
# 3. Clean text
# -----------------------------
def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text("\n")
    text = re.sub(r"\n+", "\n", text)
    return text


# -----------------------------
# 4. Parse courses
# -----------------------------
HEADER_PATTERN = re.compile(r"^([A-Z& ]+?\d{3}[A-Z]?)\s+(.*?)\s+\(([^)]+)\)(?:\s+(.*))?$")

def parse_courses(text: str) -> List[Dict]:
    lines = text.split("\n")
    
    courses = []
    current_course = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        match = HEADER_PATTERN.match(line)
        if match:
            # Save the previously parsed course
            if current_course:
                desc_text = "\n".join(current_course['buffer']).strip()
                prereq_match = re.search(r"Prerequisite:\s*(.*?)(?=\n[A-Z]|$)", desc_text)
                current_course['prerequisites'] = prereq_match.group(1).strip() if prereq_match else None
                current_course['description'] = desc_text
                current_course['raw_text'] = current_course['course'] + " " + current_course['title'] + "\n" + desc_text
                del current_course['buffer']
                courses.append(current_course)
            
            # Start a new course
            gened_str = match.group(4)
            current_course = {
                "course": match.group(1).strip(),
                "title": match.group(2).strip(),
                "credits": match.group(3).strip(),
                "gen_ed": [x.strip() for x in gened_str.split(',')] if gened_str else [],
                "prerequisites": None,
                "description": "",
                "buffer": []
            }
        else:
            if current_course:
                # Filter out irrelevant information
                if line.startswith("View course details in MyPlan:"):
                    continue
                current_course['buffer'].append(line)
                
    # Save the last course
    if current_course:
        desc_text = "\n".join(current_course['buffer']).strip()
        prereq_match = re.search(r"Prerequisite:\s*(.*?)(?=\n[A-Z]|$)", desc_text)
        current_course['prerequisites'] = prereq_match.group(1).strip() if prereq_match else None
        current_course['description'] = desc_text
        current_course['raw_text'] = current_course['course'] + " " + current_course['title'] + "\n" + desc_text
        del current_course['buffer']
        courses.append(current_course)
        
    return courses


# -----------------------------
# 7. Build RAG text
# -----------------------------
def build_rag_text(course: Dict) -> str:
    return f"""
Course: {course['course']}
Title: {course['title']}

Description:
{course['description']}

Prerequisites:
{course['prerequisites']}

Gen Ed:
{course['gen_ed']}

Credits:
{course['credits']}
""".strip()


# -----------------------------
# 8. Main process
# -----------------------------
def main():

    print("Fetching departments...")
    dept_pages = get_departments()

    all_courses = []

    for url in tqdm(dept_pages):
        try:
            html = fetch_department(url)
            text = clean_text(html)

            courses = parse_courses(text)
            all_courses.extend(courses)

        except Exception as e:
            print(f"Failed {url}: {e}")

    # -------------------------
    # Save courses.json
    # -------------------------
    with open("courses.json", "w", encoding="utf-8") as f:
        json.dump(all_courses, f, indent=2, ensure_ascii=False)

    # -------------------------
    # Build RAG dataset
    # -------------------------
    rag_data = []

    for c in all_courses:
        rag_data.append({
            "id": c["course"],
            "text": build_rag_text(c)
        })

    with open("rag_documents.jsonl", "w", encoding="utf-8") as f:
        for item in rag_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Done!")
    print(f"Total courses: {len(all_courses)}")


if __name__ == "__main__":
    main()