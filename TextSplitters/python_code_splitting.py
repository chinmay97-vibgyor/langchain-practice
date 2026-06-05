from langchain_text_splitters import CharacterTextSplitter, Language, TextSplitter, RecursiveCharacterTextSplitter

text = """
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade

    def get_details(self):
        return f"Name: {self.name}, Age: {self.age}, Grade: {self.grade}"

    def is_passing(self):
        return self.grade >= 40

    def promote(self):
        if self.is_passing():
            self.grade += 5
            return f"{self.name} has been promoted!"
        return f"{self.name} cannot be promoted."

    def __str__(self):
        return f"Student({self.name}, {self.age}, {self.grade})"


s1 = Student("Rahul", 20, 75)
s2 = Student("Priya", 22, 35)

print(s1.get_details())
print(s1.is_passing())
print(s2.promote())
print(s1.promote())
print(s1)

"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=200, 
    chunk_overlap=0,
)

chunks = splitter.split_text(text)
print(len(chunks))
print(chunks[3])
