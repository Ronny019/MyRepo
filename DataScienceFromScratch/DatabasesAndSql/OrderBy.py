import GroupBy as gr

friendliest_letters = gr.avg_friends_by_letter \
                                        .order_by(lambda row: -row["avg_num_friends"]) \
                                        .limit(4)
print("-----------------------------------------------------")
print(friendliest_letters)