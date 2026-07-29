import json
import os

def main():
    with open('courses.json', 'r', encoding='utf-8') as f:
        courses = json.load(f)

    # Departments to pick from for wide variety
    target_depts = [
        'CSE', 'MATH', 'STAT', 'PHYS', 'BIOL', 'ECON', 'PSYCH', 'INFO', 
        'ENGL', 'CHEM', 'AFRAM', 'ARCH', 'MUSIC', 'HIST', 'PHIL', 'SOC',
        'ART', 'GEOG', 'LING', 'POLS', 'COM', 'CEE', 'ME', 'EE', 'AA'
    ]
    
    selected_courses = []
    seen_codes = set()

    for c in courses:
        code = c.get('course', '').strip()
        if not code or code in seen_codes:
            continue
        dept = code.split(' ')[0] if ' ' in code else ''
        title = c.get('title', '').strip()
        desc = c.get('description', '').strip()
        prereq = c.get('prerequisites')

        if dept in target_depts and title and desc and len(desc) > 30:
            seen_codes.add(code)
            selected_courses.append(c)
            if len(selected_courses) >= 60:
                break

    # Fallback if target_depts didn't give 60
    if len(selected_courses) < 60:
        for c in courses:
            code = c.get('course', '').strip()
            if not code or code in seen_codes:
                continue
            title = c.get('title', '').strip()
            desc = c.get('description', '').strip()
            if title and desc and len(desc) > 30:
                seen_codes.add(code)
                selected_courses.append(c)
                if len(selected_courses) >= 60:
                    break

    print(f"Loaded {len(selected_courses)} courses from courses.json")

    sample_docs = []
    test_queries = []
    ground_truths = []

    for c in selected_courses:
        code = c['course']
        title = c['title']
        credits_str = str(c.get('credits') or 'N/A')
        gen_ed_list = c.get('gen_ed')
        gen_ed_str = ", ".join(gen_ed_list) if isinstance(gen_ed_list, list) else str(gen_ed_list or 'N/A')
        prereq = c.get('prerequisites')
        prereq_str = str(prereq) if prereq else 'None'
        desc = c['description']

        content = f"Course: {code} - {title}\nCredits: {credits_str}\nGeneral Education: {gen_ed_str}\nPrerequisites: {prereq_str}\nDescription: {desc}"
        sample_docs.append(content)

        query = f"What is {code} ({title}) about and what are its prerequisites?"
        truth = f"{code} ({title}) covers: {desc} Prerequisites: {prereq_str}."
        test_queries.append(query)
        ground_truths.append(truth)

    with open('test_set_60.json', 'w', encoding='utf-8') as out:
        json.dump({
            'documents': sample_docs,
            'queries': test_queries,
            'ground_truths': ground_truths
        }, out, indent=2, ensure_ascii=False)

    print("Successfully generated test_set_60.json with 60 verified Q&A pairs.")

if __name__ == '__main__':
    main()
