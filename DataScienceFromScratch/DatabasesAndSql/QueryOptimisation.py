import CreateTableAndInsert as cr
import Join as join

print("--------------------------------------------------------")
print(join.user_interests \
            .where(lambda row: row["interest"] == "SQL") \
            .join(cr.users) \
            .select(["name"]))

print(join.user_interests \
            .join(cr.users) \
            .where(lambda row: row["interest"] == "SQL") \
            .select(["name"]))