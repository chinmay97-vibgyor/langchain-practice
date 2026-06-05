import csv, random

with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["UserID","Gender","Age","EstimatedSalary","Purcahses"])
    for i in range(1, 401):
        gender = random.choice(["Male","Female"])
        age = random.randint(18, 60)
        salary = random.randint(30000, 120000)
        purchases = random.randint(1, 10)
        writer.writerow([i, gender, age, salary, purchases])
