import CreateTableAndInsert as cr

cr.users.update({'num_friends' : 3}, # set num_friends = 3
            lambda row: row['user_id'] == 1) # in rows where user_id == 1

print(cr.users)