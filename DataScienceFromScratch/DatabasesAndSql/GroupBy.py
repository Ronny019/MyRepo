import CreateTableAndInsert as cr
import Select as sel

def min_user_id(rows): return min(row["user_id"] for row in rows)

stats_by_length = cr.users \
                    .select(additional_columns={"name_length" : sel.name_length}) \
                    .group_by(group_by_columns=["name_length"],
                    aggregates={ "min_user_id" : min_user_id,
                    "num_users" : len })

print("-----------------------------------------------------------")
print(stats_by_length)


def first_letter_of_name(row):
    return row["name"][0] if row["name"] else ""
def average_num_friends(rows):
    return sum(row["num_friends"] for row in rows) / len(rows)
def enough_friends(rows):
    return average_num_friends(rows) > 1


avg_friends_by_letter = cr.users \
                            .select(additional_columns={'first_letter' : first_letter_of_name}) \
                            .group_by(group_by_columns=['first_letter'],
                            aggregates={ "avg_num_friends" : average_num_friends },
                            having=enough_friends)

print(avg_friends_by_letter)


def sum_user_ids(rows): return sum(row["user_id"] for row in rows)


user_id_sum = cr.users \
                .where(lambda row: row["user_id"] > 1) \
                .group_by(group_by_columns=[],
                aggregates={ "user_id_sum" : sum_user_ids })

print(user_id_sum)