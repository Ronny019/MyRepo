import TableClass
import CreateTableAndInsert as cr

user_interests = TableClass.Table(["user_id", "interest"])
user_interests.insert([0, "SQL"])
user_interests.insert([0, "NoSQL"])
user_interests.insert([2, "SQL"])
user_interests.insert([2, "MySQL"])


sql_users = cr.users \
                .join(user_interests) \
                .where(lambda row: row["interest"] == "SQL") \
                .select(keep_columns=["name"])

print("--------------------------------------------")
print(sql_users)


def count_interests(rows):
    """counts how many rows have non-None interests"""
    return len([row for row in rows if row["interest"] is not None])

user_interest_counts = cr.users \
                            .join(user_interests, left_join=True) \
                            .group_by(group_by_columns=["user_id"],
                            aggregates={"num_interests" : count_interests })

print(user_interest_counts)