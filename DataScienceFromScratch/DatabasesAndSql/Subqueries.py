import Join as join
import GroupBy as gr
likes_sql_user_ids = join.user_interests \
                                .where(lambda row: row["interest"] == "SQL") \
                                .select(keep_columns=['user_id'])

print("-------------------------------------------------------------------")
print(likes_sql_user_ids.group_by(group_by_columns=[],
                            aggregates={ "min_user_id" : gr.min_user_id }))
