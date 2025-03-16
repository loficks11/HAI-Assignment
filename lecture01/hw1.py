class Student:
    def __init__(self, student_id, name, age):
        self.student_id = student_id
        self.name = name
        self.age = age
        
    def display_info(self):
        print(f"ID: {self.student_id}번 / 이름: {self.name} / 나이: {self.age}살")

class StudentManager:
    def __init__(self):
        self.students = []
        
    def add_student(self, student): # student : Student class
        self.students.append(student)
        
    def display_all_students(self):
        for S in self.students:
            S.display_info()
            
manager = StudentManager()
student1 = Student(1, "김철수", 20)
student2 = Student(2, "이영희", 21)
student3 = Student(3, "박지민", 19)
manager.add_student(student1)
manager.add_student(student2)
manager.add_student(student3)
print("현재 등록된 학생 목록:")
manager.display_all_students()

student4 = Student(4, "한지수", 22)
manager.add_student(student4)
print()
print("학번 4번 학생 추가 후:")
manager.display_all_students()