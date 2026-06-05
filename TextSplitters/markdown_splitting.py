from langchain_text_splitters import CharacterTextSplitter, Language, TextSplitter, RecursiveCharacterTextSplitter, MarkdownTextSplitter

text = """
# Student Guide

## Introduction
Welcome to the student portal. This guide will help you navigate your academic journey.

## Courses
Every student must enroll in at least 3 courses per semester.

### Core Courses
- Mathematics
- Science
- English

### Elective Courses
- Music
- Physical Education
- Computer Science

## Grading System
Grades are assigned based on performance in exams and assignments.

### Pass Criteria
A student needs a minimum of 40 marks to pass a subject.

### Promotion Rules
Students who pass all core courses are eligible for promotion to the next semester.

## Contact
For any queries, reach out to the academic office.
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=200, 
    chunk_overlap=0,
)

chunks = splitter.split_text(text)
print(len(chunks))
print(chunks[0])
