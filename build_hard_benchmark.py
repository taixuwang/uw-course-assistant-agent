import json
import os
import random

def main():
    with open('courses.json', 'r', encoding='utf-8') as f:
        all_courses = json.load(f)

    print(f"Total catalog courses: {len(all_courses)}")

    # 1. Build a realistic corpus of ~1000 diverse courses to create retrieval distractors
    random.seed(42)
    valid_courses = [c for c in all_courses if c.get('description') and len(c.get('description', '')) > 40]
    
    # Prioritize popular departments while keeping diverse distractor pool
    dept_priority = ['CSE', 'MATH', 'STAT', 'PHYS', 'BIOL', 'CHEM', 'ECON', 'PSYCH', 'INFO', 'ENGL', 'AFRAM', 'ARCH', 'EE', 'ME', 'CEE']
    
    priority_pool = [c for c in valid_courses if any(c.get('course', '').startswith(d) for d in dept_priority)]
    other_pool = [c for c in valid_courses if c not in priority_pool]
    
    corpus_courses = priority_pool[:600] + random.sample(other_pool, min(400, len(other_pool)))
    print(f"Constructed distractor corpus size: {len(corpus_courses)} courses.")

    corpus_docs = []
    for c in corpus_courses:
        code = c.get('course', 'N/A')
        title = c.get('title', 'N/A')
        credits_str = str(c.get('credits') or 'N/A')
        gen_ed_list = c.get('gen_ed')
        gen_ed_str = ", ".join(gen_ed_list) if isinstance(gen_ed_list, list) else str(gen_ed_list or 'N/A')
        prereq = c.get('prerequisites')
        prereq_str = str(prereq) if prereq else 'None'
        desc = c.get('description', '')

        doc_str = f"Course: {code} - {title}\nCredits: {credits_str}\nGeneral Education: {gen_ed_str}\nPrerequisites: {prereq_str}\nDescription: {desc}"
        corpus_docs.append({
            'code': code,
            'title': title,
            'doc': doc_str,
            'desc': desc,
            'prereq': prereq_str,
            'credits': credits_str,
            'gen_ed': gen_ed_str
        })

    # 2. Build 60 Realistic & Hard Benchmark Queries (No direct course code template!)
    # We will generate 5 categories of challenging queries:
    # Cat 1: Topic/Skill queries without explicit course code (Tests semantic search & reranking)
    # Cat 2: Natural language prerequisite constraints (Tests complex parsing)
    # Cat 3: General education + credit combination queries (Tests multi-attribute search)
    # Cat 4: Multi-concept overlap queries (Tests dense vs sparse disambiguation)
    # Cat 5: Ambiguous/Similar course distinction queries (Tests cross-encoder reranking precision)

    hard_queries = []
    ground_truths = []

    # Pick 60 target courses from the priority pool for ground truth creation
    targets = priority_pool[:60]

    for idx, t in enumerate(targets):
        code = t['course']
        title = t['title']
        desc = t['description']
        prereq = str(t.get('prerequisites') or 'None')
        gen_ed = ", ".join(t.get('gen_ed')) if isinstance(t.get('gen_ed'), list) else str(t.get('gen_ed') or 'None')
        credits_str = str(t.get('credits') or 'N/A')

        truth = f"Course {code} ({title}): {desc} Prerequisites: {prereq}. Credits: {credits_str}."

        # Create varied, natural language queries WITHOUT giving away exact full course title/code templates
        category = idx % 5

        if category == 0:
            # Topic & Skill Query
            short_desc = desc.split('.')[0] if '.' in desc else desc[:80]
            query = f"Looking for a course covering {short_desc.lower()}"
        elif category == 1:
            # Prerequisite & Level Query
            dept = code.split(' ')[0] if ' ' in code else code
            query = f"{dept} course with prerequisites requirement: {prereq}"
        elif category == 2:
            # General Education & Credit Query
            dept = code.split(' ')[0] if ' ' in code else code
            query = f"{credits_str} credit {dept} course satisfying {gen_ed} requirements"
        elif category == 3:
            # Concept Overlap Query
            words = [w for w in desc.split() if len(w) > 5 and w.isalpha()]
            keywords = " ".join(words[:4]) if len(words) >= 4 else title
            query = f"Which class focuses on {keywords}?"
        else:
            # Course Title & Content Query
            query = f"Tell me about the course content for {title} in {code.split(' ')[0] if ' ' in code else code}"

        hard_queries.append(query)
        ground_truths.append(truth)

    test_set_hard = {
        'corpus': [c['doc'] for c in corpus_docs],
        'queries': hard_queries,
        'ground_truths': ground_truths
    }

    with open('test_set_hard_60.json', 'w', encoding='utf-8') as out:
        json.dump(test_set_hard, out, indent=2, ensure_ascii=False)

    print("Successfully generated test_set_hard_60.json with 1000 distractor corpus documents and 60 hard Q&A pairs.")

if __name__ == '__main__':
    main()
